import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import yaml, json, types
import datetime

from datasets.diffusion import load_data
from diffusion.utils import dist_util, logger
from diffusion.utils.train_utils import Trainer
from diffusion.utils.resample import create_named_schedule_sampler

from diffusion.utils.script_utils import (
    create_triplane_model,
    create_sr_triplane_model,
    create_gaussian_diffusion
)  

if __name__ == '__main__':
    """
    Train a diffusion model on images.
    """
    
    DEBUG = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/basediffusion_train.yaml', help='load config')
    args = parser.parse_args()
    
    isSR = args.config.split('/')[-1].startswith('base')
    
    with open(args.config, "r") as stream:
        try:
            opt = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
    
    def load_object(dct):
        return types.SimpleNamespace(**dct)
    opt = json.loads(json.dumps(opt), object_hook=load_object)
    print(opt)
    
    try:
        gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    except:
        gpu_list = []
        
    dist_util.setup_dist(gpu_list)
    logger.configure(dir = 'log/{}/{}'.format(
            os.path.basename(args.config).split('.')[0],
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")
    ))

    logger.log('loading diffusion model...')

    model = create_triplane_model(opt) if not isSR else create_sr_triplane_model(opt)
    diffusion = create_gaussian_diffusion(opt)

    
    if opt.train:
        
        logger.log("creating data loader...")
        
        if not isSR:
            opt.small_size = 0
            
        if opt.data_dir != '' and not DEBUG:
            data = load_data(
                data_dir=opt.data_dir,
                batch_size=opt.batch_size,
                image_size=opt.image_size,
                latent_dir=opt.latent_dir,
            )
        else:
            from torch.utils.data import DataLoader
            from datasets.diffusion import DummyDataset
            from itertools import cycle
            data = cycle(DataLoader(
                DummyDataset(
                    total_size=1000, 
                    image_size=opt.image_size
                ),
                batch_size=opt.batch_size, 
                shuffle=False, 
                num_workers=1, 
                drop_last=True
            ))
            
        logger.log("training...")
        
        model.train()
        
        model.to(dist_util.dev())
        if opt.use_fp16:
            model.convert_to_fp16()
        
        schedule_sampler = create_named_schedule_sampler(opt.schedule_sampler, diffusion)
        Trainer(
            opt,
            model=model,
            diffusion=diffusion,
            data=data,
            schedule_sampler=schedule_sampler,
        ).run_loop()
        
    else:
        logger.log("sampling...")
        
        model.load_state_dict(
            dist_util.load_state_dict(
                opt.model_path, map_location=dist_util.dev()
            )
        )
        
        model.eval()
        
        model.to(dist_util.dev())
        if opt.use_fp16:
            model.convert_to_fp16()
        
        Trainer(
            opt,
            model=model,
            diffusion=diffusion,
            data=None,
        ).sample_from_latent(opt, logger)
