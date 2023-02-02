import torch, os, time, torch, monai, torchvision
import numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
from skimage.metrics import *

# Timing
start = time.time()

# Take the last image from the training set and use it as the first image in the test set
# Perform auto-regressive prediction on the using the model output as the next input

from src.dataloader import LakeDataset, get_nearest_multiple
resize_dims = (get_nearest_multiple(419, 16), get_nearest_multiple(385, 16))
sawa_train = LakeDataset('sawa', resize_dims=resize_dims, train=True)

last_img = sawa_train[-1][0]

# Get the length of the test set
sawa_test = LakeDataset('sawa', resize_dims=resize_dims, train=False)
test_len = len(sawa_test)

# Create a empty arrays to store the predictions and ground truth
predictions = np.zeros((test_len, 1, resize_dims[0], resize_dims[1]))
ground_truth = np.zeros((test_len, 1, resize_dims[0], resize_dims[1]))
for i in range(test_len):
    img, label = sawa_test[i]
    ground_truth[i] = img

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
from src.model import get_model
model = get_model().to(device)

# Load the model weights
model_name = 'base_unet'
model.load_state_dict(torch.load(f'artifacts/models/{model_name}.pth'))

# Pass the last image to the model and get the predictions
for i in range(test_len):
    img = torch.from_numpy(last_img).unsqueeze(0).to(device)
    pred = model(img)
    predictions[i] = pred.detach().cpu().numpy()
    last_img = pred.detach().cpu().numpy()

# Calculate the NRMSE, PSNR and SSIM for the predictions and ground truth
nrmse, psnr, ssim = [], [], []
for i in range(test_len):
    nrmse.append(normalized_root_mse(predictions[i], ground_truth[i]))
    psnr.append(peak_signal_noise_ratio(predictions[i], ground_truth[i]))
    ssim.append(structural_similarity(predictions[i], ground_truth[i]))

    
# Plot the predictions on first 10 images and the ground truth
c = 10
plt.figure(figsize=(2*c, 1*c))
plt.suptitle('Predictions vs Ground Truth')
for i in range(1, c+1):
    plt.subplot(2, c, i+1)
    plt.imshow(predictions[i][0])
    plt.title(f'SSIM: {ssim[i]:.3f}')
    plt.subplot(2, c, i+1+c)
    plt.imshow(ground_truth[i][0])

plt.show()