import torch, time, torch, torchvision
import numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob

# get nearest multiple of given number
def get_nearest_multiple(num:int, multiple:int) -> int:
    upper = int(np.ceil(num / multiple) * multiple)
    lower = int(np.floor(num / multiple) * multiple)
    
    return upper if (upper - num) <= (num - lower) else lower

class LakeDataset(Dataset):
    def __init__(self, dataset_name, transforms=None, resize_dims=None, train=True):
        
        self.transforms = transforms
        self.resize_dims = resize_dims
        ttstr = 'train'
        if not train:
            ttstr = 'test'
        self.files = sorted(glob(f'data/{dataset_name}/{ttstr}/*.png'))
        print(f'Loaded {len(self.files)} images from {dataset_name}/{ttstr} dataset')
        
    def __len__(self):
        return len(self.files) - 1
    
    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.files[idx], mode=torchvision.io.ImageReadMode.GRAY)
        label = torchvision.io.read_image(self.files[idx+1], mode=torchvision.io.ImageReadMode.GRAY)
        if self.resize_dims:
            img = torchvision.transforms.functional.resize(img, self.resize_dims)
            label = torchvision.transforms.functional.resize(label, self.resize_dims)
        if self.transforms:
            img = self.transforms(img)
            label = self.transforms(label)
        return img/255, label/255
    
    
if __name__ == '__main__':
    
    # C, H, W
    resize_dims = (get_nearest_multiple(419, 16), get_nearest_multiple(385, 16))
    print(f'resize_dims: {resize_dims}')
    sawa = LakeDataset('sawa', resize_dims=resize_dims)
    
    img0, label0 = sawa[0]
    print(f'img0.shape: {img0.shape}')
    print(f'label0.shape: {label0.shape}')
    print(f"Min: {torch.min(img0)}, Max: {torch.max(img0)}")
    
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img0[0])
    plt.subplot(1, 2, 2)
    plt.imshow(label0[0])
    plt.show()