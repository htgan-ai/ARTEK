import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from PIL import Image

from .utils import cvtColor, preprocess_input, resize_image

def compute_image_mean_color(img: Image.Image):

    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:                      
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:                
        arr = arr[..., :3]
    mean = arr.reshape(-1, 3).mean(axis=0) # (3,)
    mean_rgb = tuple(int(round(x)) for x in mean)
    return mean_rgb

class FacenetDataset(data.Dataset):
    def __init__(self,
                 input_shape,
                 lines,
                 random,
                 k90_prob: float = 0.5,              
                 small_rotate_prob: float = 0.5,     
                 small_rotate_max_deg: float = 15.0  
                 ):
        self.input_shape          = input_shape
        self.lines                = lines
        self.random               = random
        self.k90_prob             = float(k90_prob)
        self.small_rotate_prob    = float(small_rotate_prob)
        self.small_rotate_max_deg = float(small_rotate_max_deg)

    def __len__(self):
        return len(self.lines)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self, index):
        annotation_path = self.lines[index].split(';')[1].split()[0]
        y               = int(self.lines[index].split(';')[0])

        image = cvtColor(Image.open(annotation_path))

        if self.random:

            fill = compute_image_mean_color(image)

            if self.rand() < self.k90_prob:
                k = np.random.choice([1, 2, 3])
                image = image.rotate(90 * k, resample=Image.BILINEAR, expand=True, fillcolor=fill)


            if self.rand() < self.small_rotate_prob:
                angle = np.random.uniform(-self.small_rotate_max_deg, self.small_rotate_max_deg)
                image = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=fill)

        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)

        image = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))
        return image, y

def dataset_collate(batch):
    images  = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).long()
    return images, targets

class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(LFWDataset, self).__init__(dir,transform)
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = self.get_lfw_paths(dir)


    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()

            if len(parts) in (3,4):
                pairs.append(parts)
        return pairs   

    def get_lfw_paths(self,lfw_dir,file_ext="jpg"):

        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []


        for pair in pairs:
            if len(pair) == 3:
                name = pair[0]; i1 = pair[1]; i2 = pair[2]
                p0 = try_resolve(lfw_dir, name, i1)
                p1 = try_resolve(lfw_dir, name, i2)
                issame = True
            else:  # len == 4
                n1, i1, n2, i2 = pair
                p0 = try_resolve(lfw_dir, n1, i1)
                p1 = try_resolve(lfw_dir, n2, i2)
                issame = False

            if p0 and p1:
                path_list.append((p0, p1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def __getitem__(self, index):
        (path_1, path_2, issame)    = self.validation_images[index]
        image1, image2              = Image.open(path_1), Image.open(path_2)

        image1 = resize_image(image1, [self.image_size[1], self.image_size[0]], letterbox_image = True)
        image2 = resize_image(image2, [self.image_size[1], self.image_size[0]], letterbox_image = True)
        
        image1, image2 = np.transpose(preprocess_input(np.array(image1, np.float32)),[2, 0, 1]), np.transpose(preprocess_input(np.array(image2, np.float32)),[2, 0, 1])

        return image1, image2, issame

    def __len__(self):
        return len(self.validation_images)
    
def try_resolve(lfw_dir, name, idx, exts=(".jpg",".jpeg",".png",".bmp",".BMP")):
    idx = int(idx)
    candidates = []

    for ext in exts:
        candidates.append(os.path.join(lfw_dir, name, f"{name}_{idx}{ext}"))

    for ext in exts:
        candidates.append(os.path.join(lfw_dir, name, f"{name}_{idx:04d}{ext}"))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None