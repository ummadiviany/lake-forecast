import torch, torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from glob import glob
torch.manual_seed(2000)
import torchvision.transforms.functional as TF
import pandas as pd
from einops import rearrange

# class SeqImageDataset(Dataset):
#     def __init__(self, dataset_path, label_path, img_transforms=None, sequence_length=3):
        
#         self.img_transforms = img_transforms
#         self.sequence_length = sequence_length
#         self.labels = pd.read_csv(label_path)
#         self.files = sorted(glob(f'{dataset_path}/*.png'))
#         print(f'Loaded {len(self.files)} images from {dataset_path} dataset')
        
#     def __len__(self):
#         return len(self.files) - self.sequence_length
    

#     def __getitem__(self, idx):
#         images = [
#             torchvision.io.read_image(self.files[idx+i], mode=torchvision.io.ImageReadMode.GRAY) for i in range(self.sequence_length)
#         ]
#         images = torch.stack(images, dim=0)
#         labels = self.labels.iloc[idx : idx + self.sequence_length]            

#         if self.img_transforms:
#             images = self.img_transforms(images)
            
#         return images, labels
    


# trainset = SeqImageDataset('data/train', 'data/train.csv')
# train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
    
if __name__ == '__main__':
    x = torch.randn(4, 3, 5, 5)
    print(rearrange(x, 'b c h w -> c h (w b)').shape)
    