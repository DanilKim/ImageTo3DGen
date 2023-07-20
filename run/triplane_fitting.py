import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.optim as optim
import numpy as np

import shutil
import random
from datetime import datetime

import argparse
import yaml, json, types

from nerf.train_utils import Trainer
from models.mlpdecoder import TriplaneLearner

#from utils.losses import VGGPerceptualLoss
#from torch_ema import ExponentialMovingAverage

# from nerf.diffaug import DiffAugment


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/triplane_fitting.yaml', help='load config')
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
                        datetime.now().strftime("tmax-%Y-%m-%d-%H-%M-%S-%f")
                    )

    os.makedirs(opt.workspace, exist_ok=True)
    shutil.copy(args.config, os.path.join(opt.workspace, os.path.basename(args.config)))
    
    seed_everything(opt.seed)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.test:
        from datasets.mlpdecoder import TriplaneDataset
        from torch.utils.data import DataLoader
        
        model = None # MLPDecoder!
        
        trainer = Trainer('lift', opt, model, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)
        test_loader = DataLoader(
            TriplaneDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100, shading=opt.test_shading),
            batch_size=opt.batch_size, shuffle=False, num_workers=0
        )
        
        trainer.test(test_loader)
        
        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
            
    else:
        from datasets.mlpdecoder import MultiviewImages
        
        train_loader = MultiviewImages(opt, data_dir=opt.data_dir, device=device).dataloader()
        
        num_objects = len(train_loader)
        
        model = TriplaneLearner(opt, num_objects=num_objects, triplane_size=opt.feature_size, triplane_dim=opt.feature_dim, 
                                triplane_cpu_intermediate=opt.triplane_cpu_intermediate,
                                hidden_dim=128, num_sigma_layers=2, num_bg_layers=2)
        
        print(model)
        
        #params = [{'params': embedding.parameters()} for embedding in model.triplane_features]
        if opt.freeze_decoder:
            params = [{'params': model.triplane_features.parameters()}]
        else:
            params = [{'params': model.parameters()}]
            
        optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        
        trainer = Trainer('mlp_decoder', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
        
        opt.max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    
        if True:
            trainer.train(train_loader, valid_loader=None, max_epochs=opt.max_epoch)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)