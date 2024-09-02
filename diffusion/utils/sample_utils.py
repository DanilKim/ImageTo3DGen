import os
import blobfile as bf
import numpy as np
import torch as th

import torch.distributed as dist
from diffusion.utils import dist_util
from diffusion.utils.train_utils import parse_resume_step_from_filename
from datasets.diffusion import load_data

NUM_CLASSES = 1000

import pdb


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []
                


def sample(opt, model, diffusion, logger):
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
            diffusion.p_sample_loop if not opt.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (opt.batch_size, opt.in_channels, opt.image_size, opt.image_size),
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
        step = parse_resume_step_from_filename(opt.model_path)
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_step{step}.npz")
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if opt.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            
        import random


        # vis = np.empty(shape=(arr.shape[1], arr.shape[2]*5, arr.shape[3]), dtype=np.uint8)
        # vis_inds = random.sample(range(arr.shape[0]), 5)
        # for i in range(5):
        #     vis[:, arr.shape[2]*i: arr.shape[2]*(i+1), :] = arr[vis_inds[i]]
        # cv2.imwrite(os.path.join(logger.get_dir(), "vis.png"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        #vis = np.empty(shape=(arr.shape[1]*4, arr.shape[2]*8, arr.shape[3]), dtype=np.uint8)
        #for i in range(4):
        #    for j in range(8):
        #        vis[arr.shape[1]*i: arr.shape[1]*(i+1), arr.shape[2]*j: arr.shape[2]*(j+1)] = arr[i*8+j]
        #cv2.imwrite(os.path.join(logger.get_dir(), "vis_all.png"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))


    dist.barrier()
    logger.log("sampling complete")
    


def sample_image(opt, model, diffusion, logger):
    import cv2
    if dist.get_rank() == 0:
        shape_str = f'{opt.num_samples}x{opt.image_size}x{opt.image_size}x3'
        step = parse_resume_step_from_filename(opt.model_path)
        save_dir = os.path.join(logger.get_dir(), f"samples_{shape_str}_step{step}")
        os.makedirs(save_dir, exist_ok=True)

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
            diffusion.p_sample_loop if not opt.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (opt.batch_size, opt.in_channels, opt.image_size, opt.image_size),
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
        #### Save Individual Sampled Images ####
        if dist.get_rank() == 0:
            for i, img in enumerate(all_images[-1]):
                img_id = (len(all_images)-1) * opt.batch_size + i
                cv2.imwrite(os.path.join(save_dir, f'{img_id:05d}.jpg'), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: opt.num_samples]
    if opt.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: opt.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        step = parse_resume_step_from_filename(opt.model_path)
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_step{step}.npz")
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if opt.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            
        import random


        # vis = np.empty(shape=(arr.shape[1], arr.shape[2]*5, arr.shape[3]), dtype=np.uint8)
        # vis_inds = random.sample(range(arr.shape[0]), 5)
        # for i in range(5):
        #     vis[:, arr.shape[2]*i: arr.shape[2]*(i+1), :] = arr[vis_inds[i]]
        # cv2.imwrite(os.path.join(logger.get_dir(), "vis.png"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        #vis = np.empty(shape=(arr.shape[1]*4, arr.shape[2]*8, arr.shape[3]), dtype=np.uint8)
        #for i in range(4):
        #    for j in range(8):
        #        vis[arr.shape[1]*i: arr.shape[1]*(i+1), arr.shape[2]*j: arr.shape[2]*(j+1)] = arr[i*8+j]
        #cv2.imwrite(os.path.join(logger.get_dir(), "vis_all.png"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))


    dist.barrier()
    logger.log("sampling complete")


    
def sample_from_latent(opt, model, diffusion, logger):
    all_images = []
    all_labels = []
    all_paths = []
    
    # if latent_dir path is specified, opt.num_samples is ignored
    num_samples = opt.num_samples
    latent_dir = None if opt.latent_dir == '' else opt.latent_dir
    noise_dir = None if opt.noise_dir == '' else opt.noise_dir
    if latent_dir is not None:
        from datasets.diffusion import LatentDataset, LatentNoiseDataset
        from itertools import cycle
        latent_loader = cycle(th.utils.data.DataLoader(
            LatentDataset(latent_dir) if noise_dir is None else LatentNoiseDataset(latent_dir, noise_dir),
            batch_size = opt.batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=True #False
        ))
        #num_samples = len(latent_loader)
    
    while len(all_images) * opt.batch_size < num_samples:
        model_kwargs = {}
        if latent_dir is not None:
            if noise_dir is not None:
                latents, noises, paths = next(latent_loader)
                noises = noises.to(dist_util.dev())
            else:
                noises = None
                latents, paths = next(latent_loader)
        else:
            latents = th.randn(size=(opt.batch_size, opt.latent_dim), device=dist_util.dev())
            paths, noises = None, None
        all_paths.extend(paths)
        
        model_kwargs["z"] = latents.to(dist_util.dev())
        sample_fn = (
            diffusion.p_sample_loop if not opt.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (opt.batch_size, opt.in_channels, opt.image_size, opt.image_size),
            clip_denoised=opt.clip_denoised,
            progress=opt.progress,
            model_kwargs=model_kwargs,
            noise=noises
        )
        sample = sample.to(th.float32) # Triplane feature
        #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) # Image
        
        # Dynamic Thresholding
        if opt.dynamic_threshold > 0:
            s = np.percentile(sample.view(-1).cpu().numpy(), int(100*opt.dynamic_threshold))
            sample = th.clip(sample, -s, s) / s
        else:
            sample = th.clip(sample, -1.0, 1.0)
        
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
    paths = all_paths[: opt.num_samples]
    
    if opt.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: opt.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        step = parse_resume_step_from_filename(opt.model_path)
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_step{step}.npz")
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        
        #flipped_out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_step{step}_flipped.npz")
        
        logger.log(f"saving to {out_path}")
        if opt.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            #np.savez(flipped_out_path, arr[:,:,::-1])
        
        #import cv2
        #import random
        
        #vis = np.empty(shape=(arr.shape[1], arr.shape[2]*5, arr.shape[3]), dtype=np.uint8)
        ##vis_inds = sorted(random.sample(range(arr.shape[0]), 5))
        #for i in range(5):
        #    vis[:, arr.shape[2]*i: arr.shape[2]*(i+1), :] = arr[i] #arr[vis_inds[i]]
        #cv2.imwrite(os.path.join(logger.get_dir(), "vis.png"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        
        # print(paths)
        # vis = np.empty(shape=(arr.shape[1]*2, arr.shape[2]*4, arr.shape[3]), dtype=np.uint8)
        # for i in range(2):
        #     for j in range(4):
        #         vis[arr.shape[1]*i: arr.shape[1]*(i+1), arr.shape[2]*j: arr.shape[2]*(j+1)] = arr[i*4+j]
        # cv2.imwrite(os.path.join(logger.get_dir(), "vis_all.png"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    dist.barrier()
    logger.log("sampling complete")
    
    
    
def sample_from_low_res(opt, model, diffusion, logger):
    all_images = []
    
    # if latent_dir path is specified, opt.num_samples is ignored
    num_samples = opt.num_samples
    if os.path.isdir(opt.low_res_path):
        lr_img_loader = load_data(
            data_dir=opt.low_res_path,
            batch_size=opt.batch_size,
            image_size=opt.small_size,
            latent_dir=opt.latent_dir,
            class_cond=opt.class_cond
        )
        '''
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
        '''
        num_samples = len(lr_img_loader)
        
    while len(all_images) * opt.batch_size < num_samples:
        model_kwargs = {}
        if os.path.isdir(opt.low_res_path):
            model_kwargs["low_res"] = next(lr_img_loader)
        elif os.path.splitext(opt.low_res_path)[-1] == '.npz':
            data = load_data_for_worker(opt.low_res_path, opt.batch_size, opt.class_cond)
            model_kwargs = next(data)
            # 'low_res', 'r'
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        else:
            model_kwargs["low_res"] = th.randn(size=(opt.batch_size, 3*opt.num_channels, opt.small_size, opt.small_size), device=dist_util.dev())

        sample_fn = (
            diffusion.p_sample_loop if not opt.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (opt.batch_size, opt.in_channels, opt.image_size, opt.image_size),
            clip_denoised=opt.clip_denoised,
            progress=opt.progress,
            model_kwargs=model_kwargs,
        )
        #sample = sample.to(th.float32) # Triplane feature
        #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) # Image
        
        sample = th.clip(sample, -1.0, 1.0)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * opt.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: opt.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        step = parse_resume_step_from_filename(opt.model_path)
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_step{step}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        
        '''
        import cv2
        import random

        vis = np.empty(shape=(arr.shape[1]*4, arr.shape[2]*8, arr.shape[3]), dtype=np.uint8)
        for i in range(4):
            for j in range(8):
                vis[arr.shape[1]*i: arr.shape[1]*(i+1):, arr.shape[2]*j: arr.shape[2]*(j+1), :] = arr[i*8+j]
        
        #vis = np.empty(shape=(arr.shape[1], arr.shape[2]*5, arr.shape[3]), dtype=np.uint8)
        #vis_inds = random.sample(range(arr.shape[0]), 5)
        #for i in range(5):
        #    vis[:, arr.shape[2]*i: arr.shape[2]*(i+1), :] = arr[vis_inds[i]]
        cv2.imwrite(os.path.join(logger.get_dir(), "vis.png"), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        '''
        
    dist.barrier()
    logger.log("sampling complete")
    
    
    
def sample_latent(opt, model, diffusion, logger):
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
            diffusion.p_sample_loop if not opt.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
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
        step = parse_resume_step_from_filename(opt.model_path)
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_step{step}.npz")
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if opt.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")