import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import numpy as np

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.

INITIAL_LOG_LOSS_SCALE = 20.0
NUM_CLASSES = 1000


class Trainer:
    def __init__(
        self,
        opt,
        model,
        diffusion,
        data,
        schedule_sampler=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = opt.batch_size
        self.microbatch = opt.microbatch if opt.microbatch > 0 else opt.batch_size
        self.lr = opt.lr
        self.ema_rate = (
            [opt.ema_rate]
            if isinstance(opt.ema_rate, float)
            else [float(x) for x in opt.ema_rate.split(",")]
        )
        self.log_interval = opt.log_interval
        self.save_interval = opt.save_interval
        self.resume_checkpoint = opt.resume_checkpoint
        self.use_fp16 = opt.use_fp16
        self.fp16_scale_growth = opt.fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = opt.weight_decay
        self.lr_anneal_steps = opt.lr_anneal_steps

        self.step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.resume_step = 0

        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=opt.fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            #if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        print(f'ema ckpt : {ema_checkpoint}')
        if ema_checkpoint:
            #if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        print(f'opt ckpt : {opt_checkpoint}')
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)

            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
    
    def run_epoch_loop(self):
        import time
        total_time = 0
        for i in range(self.data.dataset.num_epochs):
            logger.log('\nEpoch: %d, num iterations %d' % (i, len(self.data)))
            end = time.time()
            for j, (batch, cond) in enumerate(self.data):
                self.run_step(batch, cond)
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                    print(f'Epoch {i} - {j}th iteration')

                self.step += 1
            
            total_time += time.time() - end
            logger.log((f'Training time until {i} epoch : ', total_time))
            
            if i % self.save_interval == 0:
                logger.dumpkvs()
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
        
        # Save the last checkpoint if it wasn't already saved.
        self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            
            noise = None
            if 'noise' in cond:
                noise = cond['noise'][i : i + self.microbatch].to(dist_util.dev())
                del cond['noise']

            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                noise=noise,
                model_kwargs=micro_cond
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            ###### InfoBatch : all scoring/rescaling/getting mean is now conducted here ######
            loss = self.data.dataset.update(losses["loss"] * weights)
            ###################################################
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()
        
        
    def sample(self, opt, logger):
        
        all_images = []
        all_labels = []
        while len(all_images) * opt.batch_size < opt.num_samples:
            model_kwargs = {}
            if opt.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(opt.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                self.diffusion.p_sample_loop if not opt.use_ddim else self.diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                (opt.batch_size, 3, opt.image_size, opt.image_size),
                clip_denoised=opt.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if opt.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * opt.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: opt.num_samples]
        if opt.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: opt.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if opt.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)
                
            import cv2
            import random
            
            vis = np.empty(shape=(arr.shape[1], arr.shape[2]*5, arr.shape[3]), dtype=np.uint8)
            vis_inds = random.sample(range(arr.shape[0]), 5)
            for i in range(5):
                vis[:, arr.shape[2]*i: arr.shape[2]*(i+1), :] = arr[vis_inds[i]]
            cv2.imwrite(os.path.join(logger.get_dir(), "vis.png"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        dist.barrier()
        logger.log("sampling complete")
        
        
    def sample_from_latent(self, opt, logger):
        all_images = []
        all_labels = []
        
        # if latent_dir path is specified, opt.num_samples is ignored
        num_samples = opt.num_samples
        latent_dir = None if opt.latent_dir == '' else opt.latent_dir
        if latent_dir is not None:
            from datasets.diffusion import LatentDataset
            from itertools import cycle
            latent_loader = cycle(th.utils.data.DataLoader(
                LatentDataset(latent_dir),
                batch_size = opt.batch_size,
                shuffle=False,
                num_workers=2,
                drop_last=True
            ))
            num_samples = len(latent_loader)
        
        while len(all_images) * opt.batch_size < num_samples:
            model_kwargs = {}
            if latent_dir is not None:
                latents = next(latent_loader)
            else:
                latents = th.randn(size=(opt.batch_size, opt.latent_dim), device=dist_util.dev())

            model_kwargs["z"] = latents
            sample_fn = (
                self.diffusion.p_sample_loop if not opt.use_ddim else self.diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                (opt.batch_size, 3*opt.num_channels, opt.image_size, opt.image_size),
                clip_denoised=opt.clip_denoised,
                progress=opt.progress,
                model_kwargs=model_kwargs,
            )
            sample = sample.to(th.float32) # Triplane feature
            #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) # Image
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if opt.class_cond:
                gathered_labels = [
                    th.zeros_like(latents) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, latents)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * opt.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: opt.num_samples]
        if opt.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: opt.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if opt.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")
        
        
    def sample_from_low_res(self, opt, logger):
        all_images = []
        all_labels = []
        
        # if latent_dir path is specified, opt.num_samples is ignored
        num_samples = opt.num_samples
        low_res_path = None if opt.low_res_path == '' else opt.low_res_path
        if os.path.isdir(opt.low_res_path):
            from datasets.diffusion import ImageDataset, _list_image_files_recursively
            from itertools import cycle
            lr_img_loader = cycle(th.utils.data.DataLoader(
                ImageDataset(
                    opt.small_size, 
                    _list_image_files_recursively(low_res_path)
                ),
                batch_size = opt.batch_size,
                shuffle=False,
                num_workers=2,
                drop_last=True
            ))
            num_samples = len(lr_img_loader)
        elif os.path.splitext(opt.low_res_path)[-1] == '.npz':
            lr_imgs = np.load(opt.low_res_path)['arr_0']
            num_samples = lr_imgs.shape[0]
        else:
            print(
                "[WARNING] The path for low resolution images does not exist, or type is not correct \n \
                (should be either directory of image files, or single npz file) \n \
                Random normal images will be used for upsample"
            )
        
        while len(all_images) * opt.batch_size < num_samples:
            model_kwargs = {}
            if os.path.isdir(opt.low_res_path):
                lr_img = next(lr_img_loader)
            elif os.path.splitext(opt.low_res_path)[-1] == '.npz':
                lr_img = lr_imgs[len(all_images) : min(len(all_images) * opt.batch_size, num_samples)]
                lr_img = th.FloatTensor(lr_img).to(dist_util.dev())
            else:
                lr_img = th.randn(size=(opt.batch_size, 3*opt.num_channels, opt.small_size, opt.small_size), device=dist_util.dev())

            model_kwargs["low_res"] = lr_img
            sample_fn = (
                self.diffusion.p_sample_loop if not opt.use_ddim else self.diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                (opt.batch_size, 3*opt.num_channels, opt.large_size, opt.large_size),
                clip_denoised=opt.clip_denoised,
                progress=opt.progress,
                model_kwargs=model_kwargs,
            )
            sample = sample.to(th.float32) # Triplane feature
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) # Image
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log(f"created {len(all_images) * opt.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: opt.num_samples]
        if opt.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: opt.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if opt.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")
        
        
    def sample_latent(self, opt, logger):
        all_zs = []
        all_labels = []
        
        # if latent_dir path is specified, opt.num_samples is ignored
        num_samples = opt.num_samples
        
        while len(all_zs) * opt.batch_size < num_samples:
            model_kwargs = {}
            if opt.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(opt.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                self.diffusion.p_sample_loop if not opt.use_ddim else self.diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                (opt.batch_size, opt.num_channels),
                clip_denoised=opt.clip_denoised,
                progress=opt.progress,
                model_kwargs=model_kwargs
            )
            sample = sample.to(th.float32) # Latent feature
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_zs.extend([sample.cpu().numpy() for sample in gathered_samples])
            
            if opt.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                
            logger.log(f"created {len(all_zs) * opt.batch_size} samples")

        arr = np.concatenate(all_zs, axis=0)
        arr = arr[: opt.num_samples]
        if opt.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: opt.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if opt.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
            
