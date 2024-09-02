import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn

import argparse
import yaml, json, types
import datetime

from models.imgencoder import ImageEncoder

from diffusion.utils import dist_util, logger
from diffusion.utils.train_utils import Trainer
from diffusion.utils.resample import create_named_schedule_sampler

from diffusion.utils.script_utils import create_gaussian_diffusion

def seed_everything(seed):
    import random
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def create_model(config_fn, opt):
    
    if config_fn.startswith('base'):
        from diffusion.utils.script_utils import create_triplane_model
        return create_triplane_model(opt)
    
    
    elif config_fn.startswith('image'):
        from diffusion.utils.script_utils import create_image_model
        return create_image_model(opt)
    
    else:
        raise NotImplementedError("config filename should start with [ base | upsample | image | imageupsample ]")


class DiffAE(nn.Module):
    def __init__(self, config_fn, opt):
        super().__init__()
        
        self.encoder = ImageEncoder("cuda", 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K').clip_model.vision_model
        self.projection = ImageEncoder("cuda", 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K').clip_model.visual_projection
        
        self.diffusion_unet = create_model(config_fn, opt)
        
        self.output_attentions = False
        self.output_hidden_states = False
        self.return_dict = True
    
    def forward(self, x, temb, img):
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        
        vision_outputs = self.encoder(
            pixel_values=img,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=self.return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.projection(pooled_output)
        
        diffusion_out = self.diffusion_unet(x, temb, image_features)

        return diffusion_out


def get_dataloader(opt, config_fn, random_mode=False):
    from datasets.diffusion import load_data, load_superres_data, load_random_data
    opt.bound = tuple(float(a) for a in opt.bound.split(',')) if opt.bound is not None else (-1.0, 1.0)
    
    if random_mode:
        return load_random_data(opt.image_size, opt.batch_size, 'upsample' in config_fn)
    if 'upsample' in config_fn:
        return load_superres_data(
            data_dir=opt.data_dir,
            batch_size=opt.batch_size,
            large_size=opt.image_size,
            small_size=opt.small_size,
            latent_dir=opt.latent_dir,
            class_cond=opt.class_cond,
            bound=opt.bound,
            guidance_strength=opt.guidance_strength
        )
    else:
        return load_data(
            data_dir=opt.data_dir,
            batch_size=opt.batch_size,
            image_size=opt.image_size,
            latent_dir=opt.latent_dir,
            noise_dir=opt.noise_dir,
            class_cond=opt.class_cond,
            bound=opt.bound,
            guidance_strength=opt.guidance_strength
        )


def get_sample_fn(config_fn, latent_dir):
    from diffusion.utils.sample_utils import sample, sample_from_latent, sample_from_low_res
    if 'upsample' in config_fn:  
        return sample_from_low_res
    elif latent_dir != '':
        return sample_from_latent
    else:
        return sample


def update_opt(opt, config_fn):
    if 'image' in config_fn:
        opt.in_channels = 3
    else:
        opt.in_channels = 3*opt.num_channels
    return opt


def create_log_dir(opt, config_path):
    config_name = os.path.basename(config_path).split('.')[0]
    words = config_name.split('_')
    split = words[1]
    if split == 'train' and opt.resume_checkpoint != '':
        assert opt.train
        log_dir = os.path.dirname(opt.resume_checkpoint)
    elif split == 'sample':
        assert not opt.train
        log_dir = os.path.dirname(opt.model_path)
    else:
        config_name = '_'.join(words[:1] + words[2:])
        log_dir = 'log/{}/{}'.format(
            config_name,
            datetime.datetime.now().strftime("tmax-%Y-%m-%d-%H-%M-%S-%f")
        )
    logger.configure(dir = log_dir, log_suffix='_' + split)


if __name__ == '__main__':
    """
    Train a diffusion model on images.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/basediffusion_train.yaml', help='load config')
    args = parser.parse_args()
    
    isSR = args.config.split('/')[-1].startswith('upsample')
    
    with open(args.config, "r") as stream:
        try:
            opt = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
    
    def load_object(dct):
        return types.SimpleNamespace(**dct)
    opt = json.loads(json.dumps(opt), object_hook=load_object)
    #opt = update_opt(opt, args.config)
            
    try:
        gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    except:
        gpu_list = []
        
    dist_util.setup_dist(gpu_list)
    
    create_log_dir(opt, args.config)

    logger.log(opt)
    logger.log('loading diffusion model...')
    
    model = create_model(args.config.split('/')[-1], opt)
    diffusion = create_gaussian_diffusion(opt)

    seed_everything(seed=0)
    
    if opt.train:
        
        logger.log("creating data loader...")
            
        data = get_dataloader(opt, args.config, opt.data_dir == '')
            
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
        
        missing_keys, unexpected_keys = model.load_state_dict(
            dist_util.load_state_dict(
                opt.model_path, map_location=dist_util.dev()
            ), strict=False
        )
        logger.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            logger.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.log(f"[WARN] unexpected keys: {unexpected_keys}")  
        
        model.eval()
        
        model.to(dist_util.dev())
        if opt.use_fp16:
            model.convert_to_fp16()
        
        sample_fn = get_sample_fn(args.config.split('/')[-1], opt.latent_dir)
        sample_fn(opt, model, diffusion, logger)
            
        
        '''
        Trainer(
            opt,
            model=model,
            diffusion=diffusion,
            data=None,
        ).sample(opt, logger) #sample_from_latent(opt, logger)
        '''