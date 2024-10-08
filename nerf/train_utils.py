import os
import glob
import tqdm
import math
import imageio
import random
import tensorboardX

import numpy as np

import time

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from .general_utils import get_rays, visualize_depth

from torch_efficient_distloss import eff_distloss

from kornia.filters import gaussian_blur2d

import pdb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class Trainer(object):
    def __init__(self, 
            name, # name of this experiment
            opt, # extra conf
            model, # network 
            optimizer=None, # optimizer
            lr_scheduler=None, # scheduler
            criterion=nn.MSELoss(), # loss function, if None, assume inline implementation in train_step
            ema_decay=None, # if use EMA, set the decay
            metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
            local_rank=0, # which GPU am I
            world_size=1, # total num of GPUs
            device=None, # device to use, usually setting to None is OK. (auto choose device)
            mute=False, # whether to mute all print
            fp16=False, # amp optimize level
            eval_interval=1, # eval once every $ epoch
            max_keep_ckpt=2, # max num of saved ckpts in disk
            workspace='workspace', # workspace to save logs & ckpts
            best_mode='min', # the smaller/larger result, the better
            use_loss_as_metric=True, # use loss as the first metric
            report_metric_at_train=False, # also report metrics at training
            use_checkpoint="latest", # which ckpt to use at init time
            use_tensorboardX=True, # whether to use tensorboard for logging
            scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
        ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
    
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model
    
        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        #self.perc_loss = VGGPerceptualLoss()

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer #optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler #lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None


        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        self.gaussian_blur = T.GaussianBlur(15, sigma=(0.1, 10))
    
    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'].to(self.device) # [B, N, 3]
        rays_d = data['rays_d'].to(self.device) # [B, N, 3]
        
        image = data['image'].to(self.device) # [B, N, 3]
        poses = data['poses'].to(self.device) # [B, 4, 4]
        fov = data['fov']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.global_step < self.opt.albedo_iters:
            shading = 'albedo'
            ambient_ratio = 1.0
            l_p = 0
            l_a = 1
        else: 
            rand = random.random()
            if rand < self.opt.p_albedo:
                shading = 'albedo'
                ambient_ratio = 1.0
                l_a = torch.ones(3, device=rays_o.device, dtype=torch.float)
                l_p = torch.zeros(3, device=rays_o.device, dtype=torch.float)
            else:
                # re-sample pose for normal (low resolution)
                focal_x = self.opt.normal_shape / (2 * torch.tan(fov[:,0] / 2))
                focal_y = self.opt.normal_shape / (2 * torch.tan(fov[:,1] / 2))
                normal_shape = torch.ones(B) * self.opt.normal_shape / 2
                intrinsics = torch.stack([focal_x, focal_y, normal_shape, normal_shape], dim=1).to(self.device)
                #intrinsics = np.array([focal_x, focal_y, self.opt.normal_shape / 2, self.opt.normal_shape / 2])
                rays = get_rays(poses, intrinsics, self.opt.normal_shape, self.opt.normal_shape, self.opt.num_rays_per_image)
                rays_o = rays['rays_o'].to(self.device) # [B, N, 3]
                rays_d = rays['rays_d'].to(self.device) # [B, N, 3]

                H, W = self.opt.normal_shape, self.opt.normal_shape
                # shading is on
                l_a = torch.zeros(3, device=rays_o.device, dtype=torch.float) + 0.1
                l_p = torch.zeros(3, device=rays_o.device, dtype=torch.float) + 0.9
                if random.random() > self.opt.p_textureless:
                    shading = 'lambertian_df'
                    ambient_ratio = random.random() * 0.6 + 0.1
                else:
                    shading = 'textureless'
                    ambient_ratio = 0


        bg_color = None #torch.rand((B, N, 3), device=rays_o.device) # pixel-wise random

        # original light_d is None
        light_d = None
        if 'subject_idx' in data:
            outputs = self.model.render(data['subject_idx'].to(self.device), rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, light_d=light_d, l_a=l_a, l_p=l_p, **vars(self.opt))
        elif 'triplane' in data:
            outputs = self.model.render(data['triplane'].to(self.device), rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, light_d=light_d, l_a=l_a, l_p=l_p, **vars(self.opt))
        else:
            raise NotImplementedError("data should contain either [ learnable triplane subject idex || loaded triplane ]")
        
        pred_rgb = outputs['image'] # [B, N, 3] # outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        pred_depth = outputs['depth'] # [B, N] # outputs['depth'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()    
        
        loss = 0
        ww = {}
        
        # MSE loss

        loss_mse = self.criterion(pred_rgb, image)
        loss = loss + loss_mse
        
        ww['mse'] = loss_mse.item()
        
        # perceptual loss
        #loss_perceptual = self.perc_loss(pred_rgb, data['image'].cuda(), feature_layers=[2], style_layers=[0, 1, 2, 3])
        #loss = loss + loss_perceptual * self.opt.lambda_perceptual

        # occupancy loss
        pred_ws = outputs['weights_sum'].unsqueeze(-1) # [B, N, 1]

        if (np.random.random() < self.opt.p_randbg and shading != 'textureless'):
            # use rand bg
            bg_color = torch.ones_like(pred_rgb) * (torch.rand((B, 3, 1), device=rays_o.device) * 0.6 + 0.2)
            pred_rgb = pred_rgb * pred_ws + bg_color * (1 - pred_ws) # [B, N, 3]


        if self.epoch > self.opt.warmup_epoch:
            
            if self.opt.lambda_opacity > 0:
                loss_opacity = (pred_ws ** 2).mean()
                ww['opacity'] = loss_opacity.item()
                if loss_opacity >= 0.5:
                    loss = loss + self.opt.lambda_opacity * 10 * loss_opacity
                else:
                    loss = loss + self.opt.lambda_opacity * loss_opacity

            if self.opt.lambda_entropy > 0:
                alphas = (pred_ws).clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
                ww['entropy'] = loss_entropy.item()
                        
                loss = loss + self.opt.lambda_entropy * loss_entropy

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
                loss_orient = outputs['loss_orient']
                ww['orient'] = loss_orient.item()
                if self.global_step < 3000:
                    orient_weight = self.global_step / 3000 * (self.opt.lambda_orient - 1e-4) + 1e-4
                else:
                    orient_weight = self.opt.lambda_orient      
                if loss_orient.item() > 1e-2:
                    orient_weight *= 5
                loss = loss + orient_weight * loss_orient

            if self.opt.lambda_blur > 0 and 'normals' in outputs:
                normals = outputs['normals'].reshape(B, 3, self.opt.normal_shape, self.opt.normal_shape)
                with torch.no_grad():
                    normals_blur = gaussian_blur2d(normals, (9, 9), (3, 3))
                loss_blur = (normals - normals_blur).square().mean()
                ww['normals_blur'] = loss_blur.item()
                loss = loss + self.opt.lambda_blur * loss_blur


        if False: #self.global_step % 10 == 0:
            pred_depth = outputs['depth'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                im = pred_rgb
                self.writer.add_image('train/img', im[0], self.global_step)
                depth = pred_depth.squeeze()
                depth = visualize_depth(depth)
                self.writer.add_image('train/depth', depth, self.global_step)

        # Regularization
        idx = outputs['weights_sum'] > 1e-4
        loss_distortion = eff_distloss(outputs['weights'][idx], outputs['midpoint'][idx], outputs['deltas'][idx])
        loss_smoothness = F.l1_loss(outputs['sigmas'], outputs['sigmas_perturbed'])
        loss_sparsity = torch.norm(outputs['weights_sum'], p=1) / (B * N)
        
        ww['distortion'] = loss_distortion.item()
        ww['smoothness'] = loss_smoothness.item()
        ww['sparsity'] = loss_sparsity.item()
        
        loss = loss \
            + loss_distortion * self.opt.lambda_dist \
            + loss_smoothness * self.opt.lambda_smooth \
            + loss_sparsity * self.opt.lambda_sparse
        

        return pred_rgb, ww, loss

    def post_train_step(self):

        if self.opt.backbone == 'grid':

            lambda_tv = min(1.0, self.global_step / 1000) * self.opt.lambda_tv
            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)
            self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        # loss = self.opt.lambda_entropy * loss_entropy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, loss

    def test_step(self, data, bg_color=None, perturb=False):  
        rays_o = data['rays_o'].to(self.device) # [B, N, 3]
        rays_d = data['rays_d'].to(self.device) # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        assert N == H * W

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device) # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None
        
        outputs = self.model.render(data['subject_idx'].to(self.device), rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, bg_color=bg_color, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, H, W) > 0.95

        return pred_rgb, pred_depth, pred_mask


    def save_mesh(self, save_path=None, resolution=128):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=resolution)

        self.log(f"==> Finished saving mesh.")
        
    
    def save_triplane(self, subject_list, save_path=None):
        assert len(subject_list) == self.model.triplane_features.num_embeddings, \
            "length of subject list & triplane feature number should match!!"
    
        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')
        
        self.log(f"==> Saving Leanred Triplanes to {save_path}")
        
        os.makedirs(save_path, exist_ok=True)
        
        dim, size = self.opt.feature_dim, self.opt.feature_size
        for i, subject in enumerate(subject_list):
            np.save(f'{save_path}/{subject}.npy', self.model.triplane_features.weight[i].detach().cpu().numpy().reshape(3, dim, size, size))
        
        #for idx, triplane in enumerate(self.model.triplane_features):
        #    subject = subject_list[idx]
        #    print(f'[{idx+1}/{len(subject_list)}] Saving to {save_path}/{subject}.npy')
        #    np.save(f'{save_path}/{subject}.npy', triplane.weight.detach().numpy().reshape(3, dim, size, size))

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)
                
                # Save learned triplanes if triplane fitting mode
                if hasattr(self.model, 'triplane_features'):
                    self.save_triplane(train_loader._data.subjects, self.workspace)

            if valid_loader is not None:
                if self.epoch % self.eval_interval == 0:
                    self.evaluate_one_epoch(valid_loader)
                    self.save_checkpoint(full=False, best=True)
                    
                    # Save learned triplanes if triplane fitting mode
                    if hasattr(self.model, 'triplane_features'):
                        self.save_triplane(valid_loader.dataset.subjects, self.workspace)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_mask = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                # pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = visualize_depth(preds_depth[0])
                pred_depth = (pred_depth * 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
        
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, ww, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    for k, v in ww.items():
                        if k == 'tot':
                            continue
                        if k == 'sd_component':
                            continue
                        self.writer.add_scalar(f"train/loss_{k}", v, self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                    #pbar.set_description(f"loss={loss_val:.4f} (mes={ww['mse']:.4f} / dist={ww['distortion']:.3f} / smooth={ww['smoothness']:.3f} / sparse={ww['sparsity']:.3f} avg ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
