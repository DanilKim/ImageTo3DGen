# Source from DALLE-2 
# (https://github.com/lucidrains/DALLE2-pytorch/tree/main)
# (https://github.com/LAION-AI/conditioned-prior/tree/main)

import numpy as np
import torch
import torch.nn.functional as F
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, DiffusionPriorTrainer, OpenAIClipAdapter
from dalle2_pytorch.tokenizer import tokenizer


def load_prior(opt):
    clip = OpenAIClipAdapter("ViT-L/14")
    '''
    clip = CLIP(
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 49408,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        visual_enc_depth = 6,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_heads = 8
    ).cuda()
    '''
    prior_network = DiffusionPriorNetwork(
        dim=opt.embed_dim,
        depth=opt.depth,
        dim_head=opt.dim_head,
        heads=opt.heads,
        normformer=opt.normformer,
        attn_dropout=opt.attn_dropout,
        ff_dropout=opt.ff_dropout,
        num_time_embeds=opt.num_time_embeds,
        num_image_embeds=opt.num_image_embeds,
        num_text_embeds=opt.num_text_embeds,
        num_timesteps=opt.diffusion_steps,
        ff_mult=4,
    ).cuda()

    diffusion_prior = DiffusionPrior(
        net = prior_network,
        clip = clip,
        image_embed_dim=opt.embed_dim,
        timesteps = opt.diffusion_steps,
        cond_drop_prob = opt.cond_drop_prob,
        predict_x_start=opt.predict_x_start,
        loss_type=opt.loss_type,
        condition_on_text_encodings=opt.condition_on_text_encodings
    ).cuda()
    
    if opt.model_path != '':
        model_state_dict = torch.load(opt.model_path)
        if "ema_model" in model_state_dict:
            print("Loading EMA Model")
            diffusion_prior.load_state_dict(model_state_dict["ema_model"], strict=True)
        elif "ema_prior_aes_finetune.pth" in opt.model_path:
            diffusion_prior.load_state_dict(model_state_dict, strict=False)
        elif "lionai_ema" in opt.model_path:
            diffusion_prior.load_state_dict(model_state_dict, strict=False)
        else:
            print("Loading Standard Model")
            diffusion_prior.load_state_dict(model_state_dict["model"], strict=False)
        del model_state_dict
    
    return diffusion_prior, clip
    

def train(opt, dataset, save_path):
    diffusion_prior, _ = load_prior(opt)

    # prior networks (with transformer)

    diffusion_prior_trainer = DiffusionPriorTrainer(
        diffusion_prior,
        lr = opt.lr,
        wd = opt.weight_decay,
        ema_beta = opt.ema_rate,
        ema_update_after_step = 1000,
        ema_update_every = 10,
    )
    
    
    # mock data

    text = torch.randint(0, 49408, (4, 256)).cuda()
    images = torch.randn(4, 3, 256, 256).cuda()

    loss = diffusion_prior_trainer(text, images, max_batch_size = 4)
    diffusion_prior_trainer.update()  # this will update the optimizer as well as the exponential moving averaged diffusion prior
    diffusion_prior_trainer.save(save_path, overwrite=False)

    # after much of the above three lines in a loop
    # you can sample from the exponential moving average of the diffusion prior identically to how you do so for DiffusionPrior

    image_embeds = diffusion_prior_trainer.sample(text, max_batch_size = 4)
    

# Text promt -> CLIP image embedding
def sample(opt, prompt, img_emb_path):
    diffusion_prior, clip = load_prior(opt)
    
    text_tokens = tokenizer.tokenize(prompt).cuda() #, truncate=True).cuda()

    image_embed = diffusion_prior.sample(
        text=text_tokens,
        num_samples_per_batch=opt.candidates,
        cond_scale=opt.cond_scale,
        timesteps=opt.diffusion_steps,
    )
    
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    image_embed_numpy = image_embed.cpu().detach().numpy().astype("float32")
    np.save(img_emb_path, image_embed_numpy)
    print(f"Saved image embedding to {img_emb_path}")
    
    
if __name__ == '__main__':
    import argparse
    import os, shutil
    import yaml, json, types
    import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/textencoder_train.yaml', help='load config')
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
    
    dataset = None
    
    save_path = os.path.join(
        'logs', 
        os.path.basename(args.config).split('.')[0],
        datetime.datetime.now().strftime("tmax-%Y-%m-%d-%H-%M-%S-%f")
    )

    if 'train' in args.config:
        train(opt, dataset, save_path)
    elif 'sample' in args.config:
        assert opt.model_path != '', 'Model path must be specified in your config file!'
        prompts = [
            'A street sign that reads "Latent Diffusion"', 
            'A painting of a squarrel eating a burger', 
            'An image of an animal half mouse half octopus',
            'A zombie in the style of Picasso'
        ]
        img_emb_path = os.path.join(save_path, opt.model_path.split('/')[-1].split('.')[-1])
        os.makedirs(img_emb_path, exist_ok=True)
        sample(opt, prompts, img_emb_path)