import torch, time, torch, torchvision, monai
import numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
torch.manual_seed(2000)
from monai.transforms import RandSpatialCrop, ToTensor, AddChannel
import torchvision.transforms.functional as TF

# get nearest multiple of given number
def get_nearest_multiple(num:int, multiple:int) -> int:
    upper = int(np.ceil(num / multiple) * multiple)
    lower = int(np.floor(num / multiple) * multiple)
    
    return upper if (upper - num) <= (num - lower) else lower

class LakeDataset(Dataset):
    def __init__(self, dataset_path, img_transforms=None, label_transforms=None, resize_dims=None, train=True):
        
        self.img_transforms = img_transforms
        self.label_transfroms = label_transforms
        self.resize_dims = resize_dims
        self.train = train
        
        self.files = sorted(glob(f'data/{dataset_path}/*.png'))
        print(f'Loaded {len(self.files)} images from {dataset_path} dataset')
        
    def __len__(self):
        return len(self.files) - 1
    
    def img_label_transform(self, image, label):
        # Resize
        resize = transforms.Resize(size=self.resize_dims)
        image = resize(image)
        label = resize(label)

        if self.train:
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(160, 160))
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)

        return image, label

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.files[idx], mode=torchvision.io.ImageReadMode.GRAY)
        label = torchvision.io.read_image(self.files[idx+1], mode=torchvision.io.ImageReadMode.GRAY)
        
        # if self.resize_dims:
        #     img = torchvision.transforms.functional.resize(img, self.resize_dims)
        #     label = torchvision.transforms.functional.resize(label, self.resize_dims)
            
        
        # if self.img_transforms:
        #     img = self.img_transforms(img)
        # if self.label_transfroms:
        #     label = self.label_transfroms(label)

        img, label = self.img_label_transform(img, label)
            
        return img/255, label/255
    
    
if __name__ == '__main__':
    
    # C, H, W
    resize_dims = (get_nearest_multiple(419, 16), get_nearest_multiple(385, 16))
    print(f'resize_dims: {resize_dims}')
    
    sawa = LakeDataset('sawa/train', resize_dims=resize_dims, train=True)
    loader = DataLoader(sawa, batch_size=1, shuffle=True)
    
    # for img, label in enumerate(loader):
    #     print()
    img0, label0 = sawa[0]
    print(f'img0.shape: {img0.shape}')
    print(f'label0.shape: {label0.shape}')
    print(f"Min: {torch.min(img0)}, Max: {torch.max(img0)}")
    
    
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(img0[0])
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(label0[0])
    # plt.axis('off')
    # # plt.show()
    # plt.savefig('test.png')

    # imgs = torch.zeros(size=(len(sawa), 1, resize_dims[0], resize_dims[1]))

    
    