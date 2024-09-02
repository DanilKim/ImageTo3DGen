import os
import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .utils import list_image_files_recursively

DEFAULT_LATENT_DIM = 768

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    latent_dir='',
    noise_dir='',
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    bound=(-1.0,1.0),
    guidance_strength=1.0
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory. (Triplane GT)
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param latent_dir: If latent vector directory specified, include a "z" key in 
                       returned dicts for latent vector. The latent
                       has equal file name with its corresponding image (triplane).
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    image_paths = list_image_files_recursively(data_dir)

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in image_paths]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    if noise_dir != '':
        noise_paths = list_image_files_recursively(noise_dir)
        # Assume each noise file has same file name with its corresponding image (triplane)
        # Ex) data/images/32455.jpg <==> data/noises/32455.npy
        # before an underscore.
        noise_path_dict = {os.path.basename(fn)[:-4] : fn for fn in noise_paths}
    
    if latent_dir != '':
        latent_paths = list_image_files_recursively(latent_dir)
        # Assume each latent file has same file name with its corresponding image (triplane)
        # Ex) data/images/32455.jpg <==> data/triplanes/32455.npy
        # before an underscore.
        latent_path_dict = {os.path.basename(fn)[:-4] : fn for fn in latent_paths}
        
        
        if noise_dir != '':
            
            img_lat_noi_triplets = [
                (
                    img_path, 
                    latent_path_dict[os.path.basename(img_path)[:-4]], 
                    noise_path_dict[os.path.basename(img_path)[:-4]]
                )
                for img_path in image_paths
            ]
            
            dataset = ImageLatentNoiseDataset(
                image_size,
                img_lat_noi_triplets,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
                random_crop=random_crop,
                random_flip=random_flip,
                bound=bound,
                cond_prop=guidance_strength
            )
        
        else:
            
            img_lat_pairs = [
                (
                    img_path, 
                    latent_path_dict[os.path.basename(img_path)[:-4]]
                )
                #if os.path.basename(img_path)[:-4] in latent_path_dict else (img_path, None) \
                for img_path in image_paths 
            ]
            
            dataset = ImageLatentDataset(
                image_size,
                img_lat_pairs,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
                random_crop=random_crop,
                random_flip=random_flip,
                bound=bound,
                cond_prop=guidance_strength
            )
            
    else:
        
        dataset = ImageDataset(
            image_size,
            image_paths,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            bound=bound,
        )
        
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_superres_data(data_dir, latent_dir, batch_size, large_size, small_size, class_cond=False, bound=(-1.0,1.0), guidance_strength=1.0):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        latent_dir=latent_dir,
        class_cond=class_cond,
        bound=None,
        guidance_strength=guidance_strength
    )
    
    for large_batch, model_kwargs in data:
        low_res = F.interpolate(large_batch, small_size, mode="area")
        
        large_batch = torch.clamp(large_batch, bound[0], bound[1])
        large_batch = ( large_batch - bound[0] ) / ( (bound[1] - bound[0]) / 2 ) - 1
        
        low_res = torch.clamp(low_res, bound[0], bound[1])
        model_kwargs["low_res"] = ( low_res - bound[0] ) / ( (bound[1] - bound[0]) / 2 ) - 1
        
        yield large_batch, model_kwargs


def load_random_data(image_size, batch_size, super_res=False):
    from torch.utils.data import DataLoader
    from datasets.diffusion import DummyDataset
    from itertools import cycle
    return cycle(DataLoader(
        DummyDataset(
            total_size=200, 
            image_size=image_size,
            sr=super_res
        ),
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=1, 
        drop_last=True
    ))


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        bound=(-1.0,1.0)
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.bound = bound

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            if path.split(".")[-1] == "npy":
                image = np.load(path)   # [ 3*feat_dim X H X W ]
                image = np.transpose(image, [1, 2, 0])
            else:
                image = Image.open(f)
                image.load()
                image = image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        if path.split(".")[-1] == "npy" and self.bound is not None:
            arr = np.clip(arr, self.bound[0], self.bound[1])
            arr = ( arr - self.bound[0] ) / ( (self.bound[1] - self.bound[0]) / 2 ) - 1
        elif path.split(".")[-1] in ['jpg', 'png', 'jpeg']:
            arr = arr.astype(np.float32) / 127.5 - 1
        else:
            arr = arr.astype(np.float32)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return np.transpose(arr, [2, 0, 1]), out_dict


class ImageLatentDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_latent_paths,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        bound=(-1.0,1.0),
        train=True,
        cond_prop=1.0
    ):
        super().__init__()
        self.train = train
        self.resolution = resolution
        self.local_image_latents = image_latent_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.bound = bound
        self.cond_prop = cond_prop

    def __len__(self):
        return len(self.local_image_latents)

    def __getitem__(self, idx):
        img_path, lat_path = self.local_image_latents[idx]
        with bf.BlobFile(img_path, "rb") as f:
            if img_path.split(".")[-1] == "npy":
                image = np.load(img_path)  # [ 3*feat_dim X H X W ]
                image = np.transpose(image, [1, 2, 0]) # H x W x C
            else:
                image = Image.open(f)
                image.load()
                image = image.convert("RGB")
            
        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1] # H x W x C

        # normalize and move range from -self.bound[0] ~ self.bound[1] to -1.0 ~ 1.0
        if img_path.split(".")[-1] == "npy" and self.bound is not None:
            arr = np.clip(arr, self.bound[0], self.bound[1])
            arr = ( arr - self.bound[0] ) / ( (self.bound[1] - self.bound[0]) / 2 ) - 1
        elif img_path.split(".")[-1] in ['jpg', 'png', 'jpeg']:
            arr = arr.astype(np.float32) / 127.5 - 1
        else:
            arr = arr.astype(np.float32)

        out_dict = {}
        
        # TODO: Also apply random_crop / center_crop to latent vector?
        if os.path.exists(lat_path):
            out_dict["z"] = np.load(lat_path).astype(np.float32)
        else:
            out_dict["z"] = np.random.normal(size=(768,))
        
        if self.train:
            # randomly drop for condition for 1 - cond_prop
            cond_on = float(np.random.binomial(1, self.cond_prop))
            if not cond_on:
                out_dict["z"] = np.zeros(shape=(out_dict["z"].shape)).astype(np.float32)
            
        return np.transpose(arr, [2, 0, 1]), out_dict


class ImageLatentNoiseDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_latent_noise_paths,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        bound=(-1.0,1.0),
        train=True,
        cond_prop=1.0
    ):
        super().__init__()
        self.train = train
        self.resolution = resolution
        self.local_image_latent_noises = image_latent_noise_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.bound = bound
        self.cond_prop = cond_prop

    def __len__(self):
        return len(self.local_image_latent_noises)

    def __getitem__(self, idx):
        img_path, lat_path, noise_path = self.local_image_latent_noises[idx]
        with bf.BlobFile(img_path, "rb") as f:
            if img_path.split(".")[-1] == "npy":
                image = np.load(img_path)  # [ 3*feat_dim X H X W ]
                image = np.transpose(image, [1, 2, 0]) 
            else:
                image = Image.open(f)
                image.load()
                image = image.convert("RGB")
            
        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        if img_path.split(".")[-1] == "npy" and self.bound is not None:
            arr = np.clip(arr, self.bound[0], self.bound[1])
            arr = ( arr - self.bound[0] ) / ( (self.bound[1] - self.bound[0]) / 2 ) - 1
        elif img_path.split(".")[-1] in ['jpg', 'png', 'jpeg']:
            arr = arr.astype(np.float32) / 127.5 - 1
        else:
            arr = arr.astype(np.float32)

        out_dict = {}
        
        # TODO: Also apply random_crop / center_crop to latent vector?
        if os.path.exists(lat_path):
            out_dict["z"] = np.load(lat_path).astype(np.float32)
        else:
            out_dict["z"] = np.random.normal(size=(768,))
        
        if self.train:
            # randomly drop for condition for 1 - cond_prop
            cond_on = float(np.random.binomial(1, self.cond_prop))
            if not cond_on:
                out_dict["z"] = np.zeros(size=(out_dict["z"].shape))

        out_dict["noise"] = np.load(noise_path).astype(np.float32)
        
        return np.transpose(arr, [2, 0, 1]), out_dict


class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        super().__init__()
        self.latent_paths = list_image_files_recursively(latent_dir)

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        lat_path = self.latent_paths[idx]
        with bf.BlobFile(lat_path, "rb") as f:
            if lat_path.split(".")[-1] == "npy":
                arr = np.load(lat_path).astype(np.float32)
            else:
                NotImplementedError("Latent vector must be numpy")
        return arr, lat_path

class LatentNoiseDataset(LatentDataset):
    def __init__(self, latent_dir, noise_dir):
        super().__init__(latent_dir)
        self.noise_paths = list_image_files_recursively(noise_dir)

    def __getitem__(self, idx):
        lat_arr, lat_path = super().__getitem__(idx)
        noi_path = self.noise_paths[idx]
        with bf.BlobFile(noi_path, "rb") as f:
            if lat_path.split(".")[-1] == "npy":
                noi_arr = np.load(noi_path).astype(np.float32)
            else:
                NotImplementedError("Noise vector must be numpy")
        return lat_arr, noi_arr, lat_path
        

class DummyDataset(Dataset):
    def __init__(self, total_size, image_size=256, latent_dim=768, sr=False):
        super().__init__()
        self.size = total_size
        self.images = np.random.normal(size=(total_size,96,image_size,image_size)).astype(np.float32)
        if not sr:
            self.latents = np.random.normal(size=(total_size,latent_dim)).astype(np.float32)
        else:
            self.latents = np.zeros(shape=(total_size,latent_dim)).astype(np.float32)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.images[idx], {"z" : self.latents[idx]}
    

class LRDataset(Dataset):
    def __init__(self, lr_img_dir):
        super().__init__()
        self.lr_paths = list_image_files_recursively(lr_img_dir)

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_path = self.lr_paths[idx]
        with bf.BlobFile(lr_path, "rb") as f:
            if lr_path.split(".")[-1] in ['jpg', 'png', 'jpeg', 'gif']:
                arr = np.load(lr_path)
            else:
                NotImplementedError("data must be image...")
        return arr


def random_latent_gen(size, dim, bs):
    data = torch.randn(size=(size, dim))
    indices = list(range(size))
    
    cnt = 0
    while True:
        cnt += bs
        if cnt > size:
            random.shuffle(indices)
            cnt = 0
            yield data[indices[:bs]], {}
        else:
            yield data[indices[cnt-bs:cnt]], {}
            
            

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    if isinstance(pil_image, np.ndarray):
        arr = pil_image
    else:
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    
    if isinstance(pil_image, np.ndarray):
        arr = pil_image
    else:
        while min(*pil_image.size) >= 2 * smaller_dim_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = smaller_dim_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    
    