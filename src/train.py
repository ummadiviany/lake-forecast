import torch, os, time, torch, monai, torchvision
import numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
from skimage.metrics import *
torch.manual_seed(2000)

# Timing
start = time.time()

# Dataloader
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=3)
    ]
)
from src.dataloader import LakeDataset, get_nearest_multiple
resize_dims = (get_nearest_multiple(419, 16), get_nearest_multiple(385, 16))
sawa_train = LakeDataset('sawa', resize_dims=resize_dims, train=True)
sawa_trainloader = DataLoader(sawa_train, batch_size=1, shuffle=True)
sawa_test = LakeDataset('sawa', resize_dims=resize_dims, train=False)
sawa_testloader = DataLoader(sawa_train, batch_size=1, shuffle=False)

# Hyperparameters
epochs = 10
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.model import get_model
model = get_model().to(device)

mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-5)

# Training loop
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(sawa_trainloader):
        optimizer.zero_grad()
        outputs = model(imgs.to(device))
        loss = mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Step {i+1}/{len(sawa_trainloader)}, Loss: {loss.item():.4f}')
            
    
    # Test the model every 5 epochs
    nrmse, psnr, ssim, loss = 0, 0, 0, 0
    for i, (imgs, labels) in enumerate(sawa_testloader):
        outputs = model(imgs)
        loss += mse_loss(outputs, labels)
        nrmse += normalized_root_mse(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
        psnr += peak_signal_noise_ratio(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
        ssim += structural_similarity(labels.squeeze().detach().cpu().numpy(), outputs.squeeze().detach().cpu().numpy())
        
    nrmse /= len(sawa_testloader)
    psnr /= len(sawa_testloader)
    ssim /= len(sawa_testloader)
    print(f'Epoch {epoch+1}/{epochs}, Test Loss: {loss/len(sawa_testloader):.4f}, NRMSE: {nrmse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')
    
    
    scheduler.step()

# Save the model checkpoint 
model_name = 'base_unet'
torch.save(model.state_dict(), f'artifacts/models/{model_name}.pth')

# Timing
print(f'Time elapsed: {time.time() - start:.2f} seconds')