import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import yaml, json, types
import datetime
from mpi4py import MPI

from diffusion.utils import dist_util, logger
from diffusion.utils.train_utils_infobatch import Trainer
from diffusion.utils.resample import create_named_schedule_sampler

from diffusion.utils.script_utils import create_gaussian_diffusion

from torch.utils.data import DataLoader
from datasets.utils import list_image_files_recursively
from datasets.diffusion import ImageDataset
from ifbtest import InfoBatch

def seed_everything(seed):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def create_model(config_fn, opt):
    
    if config_fn.startswith('base'):
        from diffusion.utils.script_utils import create_triplane_model
        return create_triplane_model(opt)
    
    elif config_fn.startswith('upsample'):
        from diffusion.utils.script_utils import create_sr_triplane_model
        return create_sr_triplane_model(opt)
    
    elif config_fn.startswith('imageupsample'):
        from diffusion.utils.script_utils import create_sr_image_model
        return create_sr_image_model(opt)
    
    elif config_fn.startswith('image'):
        from diffusion.utils.script_utils import create_image_model
        return create_image_model(opt)
    
    else:
        raise NotImplementedError("config filename should start with [ base | upsample | image | imageupsample ]")
    

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
            random_flip=opt.random_flip,
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
            random_flip=opt.random_flip,
            bound=opt.bound,
            guidance_strength=opt.guidance_strength
        )


def get_sample_fn(config_fn, latent_dir):
    from diffusion.utils.sample_utils import sample_image, sample_from_latent, sample_from_low_res
    if 'upsample' in config_fn:  
        return sample_from_low_res
    elif latent_dir != '':
        return sample_from_latent
    else:
        return sample_image


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
        log_dir = 'log/{}/num_epoch{}_ratio{}_delta{}_{}rescale'.format(
            config_name,
            opt.num_epoch,
            opt.ratio,
            opt.delta,
            '' if opt.rescale else 'no'
            #datetime.datetime.now().strftime("tmax-%Y-%m-%d-%H-%M-%S-%f")
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

        image_paths = list_image_files_recursively(opt.data_dir)

        dataset = ImageDataset(
            opt.image_size,
            image_paths,
            classes=None,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=False,
            random_flip=opt.random_flip,
            bound=(-1.0,1.0),
        )
        
        logger.log(f'InfoBatch : num epoch {opt.num_epoch} ratio {opt.ratio} delta {opt.delta} rescale {opt.rescale}')
        dataset = InfoBatch(dataset, opt.num_epoch, opt.ratio, opt.delta, opt.rescale)
        sampler = dataset.sampler

        data = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, sampler=sampler)
            
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
        ).run_epoch_loop()
        
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