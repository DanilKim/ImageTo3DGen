import os
import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

DEFAULT_LATENT_DIM = 768

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    latent_dir='',
    deterministic=False,
    random_crop=False,
    random_flip=True,
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
                       returned dicts for latent vector. The corresponding latent
                       has equal file name with its corresponding image (triplane).
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    image_paths = _list_image_files_recursively(data_dir)
    if latent_dir != '':
        latent_paths = _list_image_files_recursively(latent_dir)
        # Assume each latent file has same file name with its corresponding image (triplane)
        # Ex) data/images/32455.jpg <==> data/triplanes/32455.npy
        # before an underscore.
        latent_path_dict = {os.path.basename(fn) : fn for fn in latent_paths}
        img_lat_pairs = [
            (img_path, latent_path_dict[os.path.basename(img_path)])  \
                if os.path.basename(img_path) in latent_path_dict else (img_path, None) \
                for img_path in image_paths 
        ]
    dataset = TriplaneLatentDataset(
        image_size,
        img_lat_pairs,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
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


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


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
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class TriplaneLatentDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_latent_paths,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        train=True,
        cond_prop=1.0
    ):
        super().__init__()
        self.train = train
        self.resolution = resolution
        self.local_image_latents = image_latent_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.cond_prop = cond_prop

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        img_path, lat_path = self.local_image_latents[idx]
        with bf.BlobFile(img_path, "rb") as f:
            if img_path.split(".")[-1] == "npy":
                image = np.load(img_path)
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

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        
        if os.path.exists(lat_path):
            out_dict["z"] = np.array(lat_path, dtype=np.float32)
        else:
            out_dict["z"] = np.random.normal(size=(768,))
        
        if self.train:
            cond_prob = float(np.random.binomial(1, self.cond_prop))
            if not cond_prob:
                out_dict["z"] = np.zeros(size=(out_dict["z"].shape))
            
        return np.transpose(arr, [2, 0, 1]), out_dict


class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        super().__init__()
        self.latent_paths = _list_image_files_recursively(latent_dir)

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        lat_path = self.latent_paths[idx]
        with bf.BlobFile(lat_path, "rb") as f:
            if lat_path.split(".")[-1] == "npy":
                arr = np.load(lat_path)
            else:
                NotImplementedError("Latent vector must be numpy")
        return arr


class DummyDataset(Dataset):
    def __init__(self, total_size, image_size=256):
        super().__init__()
        self.size = total_size
        self.images = np.random.normal(size=(total_size,96,image_size,image_size)).astype(np.float32)
        self.latents = np.random.normal(size=(total_size,768)).astype(np.float32)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.images[idx], {"z" : self.latents[idx]}


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
    
    