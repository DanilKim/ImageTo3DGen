import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T
import numpy as np
import cv2

import tqdm
import random
import glob
import tensorboardX
import imageio
from datetime import datetime

import time
from rich.console import Console
from torch_ema import ExponentialMovingAverage


from nerf.general_utils import rand_poses, get_rays, visualize_depth
# from nerf.diffaug import DiffAugment

from torch_efficient_distloss import eff_distloss

from kornia.losses import ssim_loss, inverse_depth_smoothness_loss, total_variation
from kornia.filters import gaussian_blur2d


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    import argparse
    import os, shutil
    import yaml, json, types
    
    from models.mlpdecoder import MLPDecoder
    from datasets.mlpdecoder import TriplaneImagePair
    from nerf.train_utils import Trainer
    #from utils.losses import VGGPerceptualLoss
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mlpdecoder_train.yaml', help='load config')
    args = parser.parse_args()
    
    with open(args.config, "r") as stream:
        try:
            opt = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
    
    def load_object(dct):
        return types.SimpleNamespace(**dct)
    opt = json.loads(json.dumps(opt), object_hook=load_object)
    print(opt)
    
    opt.workspace = os.path.basename(args.config).replace('.yaml', '')
    opt.workspace = os.path.join(
                        'log', 
                        os.path.basename(args.config).split('.')[0],
                        datetime.datetime.now().strftime("tmax-%Y-%m-%d-%H-%M-%S-%f")
                    )

    os.makedirs(opt.workspace, exist_ok=True)
    shutil.copy(args.config, os.path.join(opt.workspace, os.path.basename(args.config)))
    
    seed_everything(opt.seed)
    
    model = MLPDecoder(opt, hidden_dim=128, num_sigma_layers=2, num_bg_layers=2)
    
    print(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.test:
        trainer = Trainer('lift', opt, model, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)
        test_loader = TriplaneImagePair(opt, device=device, type='test', H=opt.H, W=opt.W, size=100, shading=opt.test_shading).dataloader()
        trainer.test(test_loader)
        
        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
    else:
        optimizer = lambda model: torch.optim.AdamW(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    
        train_loader = TriplaneImagePair(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
        
        opt.max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        
        trainer = Trainer('mlp_decoder', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
        valid_loader = TriplaneImagePair(opt, device=device, type='val',  H=opt.H, W=opt.W, size=5).dataloader()
        
        opt.max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    
        if True:
            trainer.train(train_loader, valid_loader, opt.max_epoch)

        test_loader = TriplaneImagePair(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
        trainer.test(test_loader)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)