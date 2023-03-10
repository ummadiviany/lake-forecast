import torch, os, time, torch, monai, torchvision
import numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
from skimage.metrics import *
from monai.inferers import sliding_window_inference
torch.manual_seed(2000)

# Timing
start = time.time()

# Take the last image from the training set and use it as the first image in the test set
# Perform auto-regressive prediction on the using the model output as the next input

from src.dataloader import LakeDataset, get_nearest_multiple
resize_dims = (get_nearest_multiple(419, 32), get_nearest_multiple(385, 32))
sawa_train = LakeDataset('sawa/train', resize_dims=resize_dims, train=False)

last_img = sawa_train[-1][0]
# print(f'last_img.shape: {last_img.shape}, dtype: {last_img.dtype}')

# Get the length of the test set
sawa_test = LakeDataset('sawa/test', resize_dims=resize_dims, train=False)
test_len = len(sawa_test)

# Create a empty arrays to store the predictions and ground truth
predictions = torch.zeros((test_len, 1, resize_dims[0], resize_dims[1]))
ground_truth = torch.zeros((test_len, 1, resize_dims[0], resize_dims[1]))
for i in range(test_len):
    img, label = sawa_test[i]
    ground_truth[i] = img

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
from src.model import get_model
model = get_model().to(device)

# Load the model weights
model_name = 'unet_sawa_patch_sli'
model.load_state_dict(torch.load(f'artifacts/models/{model_name}.pth', map_location=device))

# Pass the last image to the model and get the predictions
for i in range(test_len):
    img = torch.unsqueeze(last_img, dim=0).to(device)
    # pred = model(img)
    pred = sliding_window_inference(inputs=img.to(device), roi_size=(160, 160), sw_batch_size=4,
                                                predictor=model, overlap = 0.5, mode = 'gaussian', device=device)
    predictions[i] = pred.clip(0, 1)
    last_img = predictions[i]

# Calculate the NRMSE, PSNR and SSIM for the predictions and ground truth
nrmse, psnr, ssim = [], [], []
predictions = predictions.cpu().detach().numpy()
ground_truth = ground_truth.cpu().detach().numpy()
for i in range(test_len):
    nrmse.append(normalized_root_mse(predictions[i], ground_truth[i]))
    psnr.append(peak_signal_noise_ratio(predictions[i], ground_truth[i]))
    ssim.append(structural_similarity(predictions[i][0], ground_truth[i][0]))

    
# Plot the predictions on first 10 images and the ground truth
c = 5
plt.figure(figsize=(2*c, 1*c))
plt.suptitle('Predictions vs Ground Truth')
for i in range(1, c+1):
    plt.subplot(2, c, i)
    plt.imshow(predictions[i][0])
    plt.axis('off')
    plt.title(f'SSIM: {ssim[i]:.3f}')
    plt.subplot(2, c, i+c)
    plt.imshow(ground_truth[i][0])
    plt.axis('off')
    
plt.tight_layout()
plt.savefig(f'artifacts/predictions/{model_name}.png')
print(f'Predictions saved sucessfully at artifacts/predictions/{model_name}.png')
# plt.show()