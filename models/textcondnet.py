# Source from DALLE-2 
# (https://github.com/lucidrains/DALLE2-pytorch/tree/main)
# (https://github.com/LAION-AI/conditioned-prior/tree/main)

import numpy as np
import torch
import torch.nn.functional as F
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, DiffusionPriorTrainer, CLIP, OpenAIClipAdapter


def load_prior(model_path=None, simulated_steps=1000):
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
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8
    ).cuda()

    diffusion_prior = DiffusionPrior(
        net = prior_network,
        clip = clip,
        timesteps = 100,
        cond_drop_prob = 0.2
    ).cuda()
    
    if model_path is not None:
        model_state_dict = torch.load(model_path)
        if "ema_model" in model_state_dict:
            print("Loading EMA Model")
            diffusion_prior.load_state_dict(model_state_dict["ema_model"], strict=True)
        elif "ema_prior_aes_finetune.pth" in model_path:
            diffusion_prior.load_state_dict(model_state_dict, strict=False)
        else:
            print("Loading Standard Model")
            diffusion_prior.load_state_dict(model_state_dict["model"], strict=False)
        del model_state_dict
    
    return diffusion_prior, clip
    

def train(dataset, save_path, batch_size=1):
    diffusion_prior, _ = load_prior()

    # prior networks (with transformer)

    diffusion_prior_trainer = DiffusionPriorTrainer(
        diffusion_prior,
        lr = 3e-4,
        wd = 1e-2,
        ema_beta = 0.99,
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
def sample(prompt, prior_path, img_emb_path, sample_timesteps=250, candidates=2, cond_scale=1.0):
    diffusion_prior, clip = load_prior(prior_path)
    
    text_tokens = clip.tokenize([prompt], truncate=True).cuda()

    image_embed = diffusion_prior.sample(
        text=text_tokens,
        num_samples_per_batch=candidates,
        cond_scale=cond_scale,
        timesteps=sample_timesteps,
    )
    
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    image_embed_numpy = image_embed.cpu().detach().numpy().astype("float32")
    np.save(img_emb_path, image_embed_numpy)
    print(f"Saved image embedding to {img_emb_path}")