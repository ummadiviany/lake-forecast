import torch, time, torch, torchvision, monai
import numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
torch.manual_seed(2000)
from monai.transforms import RandSpatialCrop, ToTensor, AddChannel
import torchvision.transforms.functional as TF
from einops import rearrange

# get nearest multiple of given number
def get_nearest_multiple(num:int, multiple:int) -> int:
    upper = int(np.ceil(num / multiple) * multiple)
    lower = int(np.floor(num / multiple) * multiple)
    
    return upper if (upper - num) <= (num - lower) else lower

class LakeDataset(Dataset):
    def __init__(self, dataset_path, img_transforms=None, label_transforms=None, resize_dims=None, train=True, time_steps=3):
        
        self.img_transforms = img_transforms
        self.time_steps = time_steps
        self.label_transfroms = label_transforms
        self.resize_dims = resize_dims
        self.train = train
        
        self.files = sorted(glob(f'data/{dataset_path}/*.png'))
        print(f'Loaded {len(self.files)} images from {dataset_path} dataset')
        
    def __len__(self):
        return len(self.files) - self.time_steps - 1
    
    def img_label_transform(self, images, label):
        # Resize
        resize = transforms.Resize(size=self.resize_dims)
        images = [resize(image) for image in images]
        label = resize(label)

        # if self.train:
        #     # Random crop
        #     i, j, h, w = transforms.RandomCrop.get_params(label, output_size=(160, 160))
        #     images = [TF.crop(image, i, j, h, w) for image in images]
        #     label = TF.crop(label, i, j, h, w)

        return images, label

    def __getitem__(self, idx):
        images = [
            torchvision.io.read_image(self.files[idx+i], mode=torchvision.io.ImageReadMode.GRAY) for i in range(self.time_steps)
        ]
        images = torch.stack(images, dim=0)
        label = torchvision.io.read_image(self.files[idx+self.time_steps+1], mode=torchvision.io.ImageReadMode.GRAY)
            

        images, label = self.img_label_transform(images, label)
        images = rearrange(images, 't c h w -> c t h w')
        label = rearrange(label, 'c h w -> c 1 h w')
            
        # print('-'*50)
        # print(f'images.shape: {images.shape}')
        # print(f'label.shape: {label.shape}')
        # print('-'*50)
        # B C T H W
        # B C 1 H W
        return images/255, label/255
    
    
if __name__ == '__main__':
    
    # C, H, W
    resize_dims = (get_nearest_multiple(140, 16), get_nearest_multiple(129, 16))
    print(f'resize_dims: {resize_dims}')
    
    # sawa = LakeDataset('sawa/train', resize_dims=resize_dims, train=True, time_steps=5)
    # loader = DataLoader(sawa, batch_size=2, shuffle=False)
    
    # # for img, label in enumerate(loader):
    # #     print()
    # # img0, label0 = next(iter(loader))
    # # print(f'img0.shape: {img0.shape}')
    # # print(f'label0.shape: {label0.shape}')
    # # print(f"Min: {torch.min(img0)}, Max: {torch.max(img0)}")
    
    # for i, (img, label) in enumerate(loader):
    #     print(f'Step {i}, img.shape: {img.shape}, label.shape: {label.shape}')
    
    # plt.figure(figsize= (4*2, 1*2))
    # plt.subplot(1, 4, 1)
    # plt.imshow(img0[0, 0, :, :])
    # plt.axis('off')
    # plt.subplot(1, 4, 2)
    # plt.imshow(img0[0, 1, :, :])
    # plt.axis('off')
    # plt.subplot(1, 4, 3)
    # plt.imshow(img0[0, 2, :, :])
    # plt.axis('off')
    # plt.subplot(1, 4, 4)
    # plt.imshow(label0[0, 0, :, :])
    # plt.axis('off')
    # plt.show()

    
    