import os
import json
import blobfile as bf
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
toTensor = ToTensor()

from nerf.general_utils import get_rays, rand_poses, circle_poses

from .utils import list_image_files_recursively

class TriplaneGTDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class TriplaneDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_paths = list_image_files_recursively(data_dir)

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        lat_path = self.latent_paths[idx]
        with bf.BlobFile(lat_path, "rb") as f:
            if lat_path.split(".")[-1] == "npy":
                arr = np.load(lat_path)
            else:
                NotImplementedError("Latent vector must be numpy")
        return arr


class TriplaneImagePair(Dataset):
    def __init__(self, opt, device, type='train', H=256, W=256, size=100, shading=False):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.size = size

        self.training = self.type in ['train', 'all']
        
        self.cx = self.H / 2
        self.cy = self.W / 2
        self.shading = shading

        # [debug] visualize poses
        # poses, dirs = rand_poses(100, self.device, return_dirs=self.opt.dir_text, radius_range=self.radius_range)
        # visualize_poses(poses.detach().cpu().numpy())


    def collate(self, index):

        B = len(index) # always 1

        if self.training:
            # random pose on the fly
            poses, dirs = rand_poses(B, self.device, radius_range=self.radius_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=True)
            # poses, dirs = rand_poses(B, self.device, radius_range=self.radius_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose)

            # random focal
            fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])
        else:
            # circle pose
            phi = (index[0] / self.size) * 360
            poses, dirs = circle_poses(self.device, radius=self.radius_range[1] * 1.2, theta=self.opt.init_theta, phi=phi, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # fixed focal
            fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])


        # sample a low-resolution but full image for CLIP
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        if self.shading:
            data = {
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'dir': dirs,
                'fov': fov,
                'poses': poses,
                'intrinsics': intrinsics,
                'shading': 'lambertian_df',
                'light_dir': rays['rays_o']
            }
        else:
            data = {
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'dir': dirs,
                'fov': fov,
                'poses': poses,
                'intrinsics': intrinsics,
            }

        return data


    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access dataset in trainer.
        return loader
    
    

class MultiviewImages(Dataset):
    def __init__(self, opt, data_dir, device, size=100, training=True, shading=False):
    #def __init__(self, opt, device, type='train', H=256, W=256, size=100, shading=False):
        super().__init__()
        
        # image path : {data_dir}/{subject}/{name}.[png,jpg] 
        # pose path : {data_dir}/{subject}/{name}.npy 
        # Assume image & corresponding pose has equal file name
        
        self.subjects = os.listdir(data_dir)
        self.image_pose_pairs = []
        self.all_images = {}
        self.camera_info = {}
        self.name2idx = {}
        
        self.H, self.W = None, None
        
        for i, subject in enumerate(self.subjects):
            self.all_images[subject] = []
            subject_path = os.path.join(data_dir, subject)
            with open(os.path.join(subject_path, 'transforms_train.json'), 'r') as f:
                camera_info = json.load(f)
            poses = {
                fn['file_path'].split('/')[-1] : fn['transform_matrix'] for fn in camera_info['frames']
            }
            self.camera_info[subject] = {
                k : v for k, v in camera_info.items() if k != 'frames'
            }
            
            if self.H is not None:
                assert self.H == self.camera_info[subject]['h'] and self.W == self.camera_info[subject]['w'], \
                    "Images of all subjects should have equal sizes"
                
            for im in os.listdir(subject_path + '/train'):
                self.image_pose_pairs.append({
                    'subject': subject,
                    'subject_idx': i,
                    'image_path': f'{data_dir}/{subject}/train/{im}',
                    'pose_matrix': poses[im],
                })
                    
        self.num_total_images = sum([len(self.all_images[s]) for s in self.all_images])
            
        self.opt = opt
        self.device = device

        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range

        self.training = training

        # Assume every image has equal H, W
        self.H = int(self.camera_info[subject]['h'])
        self.W = int(self.camera_info[subject]['w'])

        self.shading = shading

        # [debug] visualize poses
        # poses, dirs = rand_poses(100, self.device, return_dirs=self.opt.dir_text, radius_range=self.radius_range)
        # visualize_poses(poses.detach().cpu().numpy())


    def collate(self, index):

        B = len(index)
        
        # random pose on the fly
        image_pose_pairs = [self.image_pose_pairs[ind] for ind in index]
        poses = torch.stack([torch.FloatTensor(ipp['pose_matrix']) for ipp in image_pose_pairs], dim=0)
        subjects = [ipp['subject'] for ipp in image_pose_pairs]
        subjects_idx = torch.LongTensor([ipp['subject_idx'] for ipp in image_pose_pairs])
        
        
        camera_infos = [self.camera_info[subject] for subject in subjects]
        
        # focal
        fov = np.array([[ci['camera_angle_x'], ci['camera_angle_y']] for ci in camera_infos])
        intrinsics = torch.stack([torch.FloatTensor([
            ci['fl_x'],
            ci['fl_y'],
            ci['cx'],
            ci['cy'],
        ]) for ci in camera_infos], dim=0)

        rays = get_rays(poses, intrinsics, self.H, self.W, self.opt.num_rays_per_image)

        image_sampled = []
        for ipp in image_pose_pairs:
            image = toTensor(Image.open(ipp['image_path']).convert('RGB'))
            c, h, w = image.size()
            assert c == 3 and h == self.H and w == self.W
            image = image.reshape(3, self.H*self.W)
            image_sampled.append(image[:,rays['inds']])
        image_sampled = torch.stack(image_sampled, dim=0)
        
        if self.shading:
            data = {
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'inds': rays['inds'],
                'fov': fov,
                'image': image_sampled,
                'poses': poses,
                'intrinsics': intrinsics,
                'shading': 'lambertian_df',
                'light_dir': rays['rays_o'],
                'subject_idx': subjects_idx
            }
        else:
            data = {
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'inds': rays['inds'],
                'fov': fov,
                'image': image_sampled,
                'poses': poses,
                'intrinsics': intrinsics,
                'subject_idx': subjects_idx
            }

        return data
    
    
    def dataloader(self):
        loader = DataLoader(list(range(len(self.subjects))), batch_size=self.opt.batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access dataset in trainer.
        return loader