import torch, os, time, torch, monai, torchvision
# import numpy as np, matplotlib.pyplot as plt
# import torch.nn as nn
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from glob import glob
# from skimage.metrics import *
from src import model_ts
# from monai.inferers import sliding_window_inference
# from src.model_ts import get_model
from einops import rearrange

# from src.monai_mod.utils import sliding_window_inference

# imgs = torch.randn(1, 1, 8, 416, 384)
# imgs = rearrange(imgs, 'b c t h w -> b c h w t')
# print(f'imgs.shape: {imgs.shape}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = get_model().to(device)
# preds = sliding_window_inference(inputs=imgs, roi_size=(160, 160, 8), sw_batch_size=1, predictor=model, overlap=0.5, device=device)
# print(f'preds.shape: {preds.shape}')


from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
imgs = torch.randn(1, 1, 4, 416, 384)
imgs = rearrange(imgs, 'b c t h w -> b c h w t')
print(f'imgs.shape: {imgs.shape}')
# B x C x H x W x D
# model = UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(2, 4, 8,),
#     strides=(2, 2),
# )
model = model_ts.get_model().to(device)
preds = sliding_window_inference(inputs=imgs, roi_size=(160, 160,-1), sw_batch_size=1, predictor=model, overlap=0.5, device=device)
print(f'preds.shape: {preds.shape}')