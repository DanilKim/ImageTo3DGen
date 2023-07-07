import numpy as np
from PIL import Image
import random

from torch.utils.data import Dataset, DataLoader

from nerf.general_utils import get_rays, rand_poses, circle_poses

class TriplaneGTDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    

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