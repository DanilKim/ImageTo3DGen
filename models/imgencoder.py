# Source from NeuralLift-360 (https://github.com/VITA-Group/NeuralLift-360/blob/main/nerf/clip.py)

import torch
import torch.nn as nn

import torchvision.transforms as T
import torchvision.transforms.functional as TF

# import clip
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer
from torchvision import transforms

import torch.nn.functional as F


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    # print(x.shape, y.shape)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class ImageEncoder(nn.Module):
    def __init__(self, device, clip_name = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'):
        super().__init__()

        self.device = device

        clip_name = clip_name

        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name).cuda()
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    
        self.normalize = transforms.Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)

        self.resize = transforms.Resize(224)

         # image augmentation
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    
    def get_text_embeds(self, prompt, neg_prompt=None, dir=None):

        clip_text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        text_z = self.clip_model.get_text_features(clip_text_input)
        # text = clip.tokenize(prompt).to(self.device)
        # text_z = self.clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)

        return text_z

    def set_epoch(self, epoch):
        pass

    def get_img_embeds(self, img):
        img = self.aug(img)
        image_z = self.clip_model.get_image_features(img)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features
        return image_z

    
    def train_step(self, text_z, pred_rgb, image_ref_clip, **kwargs):

        pred_rgb = self.resize(pred_rgb)
        pred_rgb = self.normalize(pred_rgb)

        image_z = self.clip_model.get_image_features(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        # print(image_z.shape, text_z.shape)
        loss = spherical_dist_loss(image_z, image_ref_clip)

        # loss = - (image_z * text_z).sum(-1).mean()

        return loss
    
    def text_loss(self, text_z, pred_rgb):

        pred_rgb = self.resize(pred_rgb)
        pred_rgb = self.normalize(pred_rgb)

        image_z = self.clip_model.get_image_features(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        # print(image_z.shape, text_z.shape)
        loss = spherical_dist_loss(image_z, text_z)

        # loss = - (image_z * text_z).sum(-1).mean()

        return loss
    
    def img_loss(self, img_ref_z, pred_rgb):
        # pred_rgb = self.aug(pred_rgb)
        pred_rgb = self.resize(pred_rgb)
        pred_rgb = self.normalize(pred_rgb)

        image_z = self.clip_model.get_image_features(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        # loss = - (image_z * img_ref_z).sum(-1).mean()
        loss = spherical_dist_loss(image_z, img_ref_z)

        return loss



if __name__ == '__main__':
    """
    Extract CLIP encodings for images
    
    ex) ${image_dir}/${image_fn}.png -> ${embed_dir}/${image_fn}.npy
    """
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    import argparse
    import numpy as np
    from PIL import Image
    from datasets.utils import list_image_files_recursively
    from torch.utils.data import DataLoader, Dataset
    
    from tqdm.auto import tqdm
    
    import pdb
    
    encoder = ImageEncoder("cuda", 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    
    #pdb.set_trace()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True)
    parser.add_argument('-e', '--embed', type=str, default='')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    if args.embed == '':
        args.embed = '/'.join(args.image.split('/')[:-1].append('CLIP_embeds'))
    os.makedirs(args.embed, exist_ok=True)
    
    image_list = list_image_files_recursively(args.image)
    toTensor = T.ToTensor()
    
    
    class ImageDataset(Dataset):
        def __init__(self, img_dir, emb_dir):
            super().__init__()
            self.img_dir = img_dir
            self.image_list = list_image_files_recursively(img_dir)
            self.emb_dir = emb_dir

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, idx):
            img_fn = self.image_list[idx]
            pil_image = Image.open(img_fn)
            pil_image = pil_image.convert("RGB")
            arr = np.array(pil_image) / 255.
            
            save_path = os.path.join(self.emb_dir, os.path.relpath(img_fn, self.img_dir))
            save_path = save_path[:-4] + '.npy'
            return np.transpose(arr, [2, 0, 1]), save_path
        
    dataset = ImageDataset(args.image, args.embed)
    
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False
    )

    for i, batch in tqdm(enumerate(loader)):
        img_arr, path_arr = batch
        
        with torch.no_grad():
            embeds = encoder.get_img_embeds(img_arr.float().cuda())
        
        for embed, save_path in zip(embeds, path_arr):
            if os.path.exists(save_path):
                continue
            np.save(save_path, embed.cpu().numpy())

    '''
    for i, img_fn in tqdm(enumerate(image_list)):
        save_path = os.path.join(args.embed, os.path.relpath(img_fn, args.image))
        save_path = save_path[:-4] + '.npy'
        if os.path.exists(save_path):
            continue
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        pil_image = Image.open(img_fn)
        pil_image = pil_image.convert("RGB")
        image = toTensor(pil_image)[None,:].to("cuda")
        
        with torch.no_grad():
            embed = encoder.get_img_embeds(image)
        
        np.save(save_path, embed[0].cpu().numpy())
        
        if i % 50 == 0:
            print(f'[{i+1} / {len(image_list)}] saved...')
    '''