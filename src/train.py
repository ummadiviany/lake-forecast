import torch, os, time, torch, monai, torchvision, wandb
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

# Hyperparameters
wandb_log = True
epochs = 100
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4

# Dataloader
img_transforms = transforms.Compose(
    [
        # transfroms.Normalize()
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=3)
    ]
)
from src.dataloader import LakeDataset, get_nearest_multiple
resize_dims = (get_nearest_multiple(419, 32), get_nearest_multiple(385, 32))
sawa_train = LakeDataset('sawa', resize_dims=resize_dims, train=True)
sawa_trainloader = DataLoader(sawa_train, batch_size=batch_size, shuffle=True)
sawa_test = LakeDataset('sawa', resize_dims=resize_dims, train=False)
sawa_testloader = DataLoader(sawa_test, batch_size=1, shuffle=False)

if wandb_log:
    config = {
        'epochs' : epochs,
        'loss' : 'L1',
        'Augumentations' : None,
        'batch_size' : batch_size,
    }
    wandb.login()
    wandb.init(project="lake_forecast", entity="vinayu", config = config)

from src.model import get_model
model = get_model().to(device)

# mse_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-5)

# Training loop
for epoch in range(epochs):
    model.train()
    for i, (imgs, labels) in enumerate(sawa_trainloader):
        optimizer.zero_grad()
        outputs = model(imgs.to(device))
        # loss = mse_loss(outputs, labels.to(device))
        loss = mae_loss(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Step {i+1}/{len(sawa_trainloader)}, Loss: {loss.item():.4f}')
            

    # predict next image prediction every 2 epochs
    if epoch % 5 == 0:
        model.eval()
        nrmse, psnr, ssim, loss = 0, 0, 0, 0
        test_len = len(sawa_testloader)
        
        img_stack, out_stack, label_stack  = [], [], []

        for i, (imgs, labels) in enumerate(sawa_testloader):
            # outputs = model(imgs.to(device))
            outputs = sliding_window_inference(inputs=imgs.to(device), roi_size=(160, 160), sw_batch_size=4,
                                                predictor=model, overlap = 0.5, mode = 'gaussian', device=device)
            outputs.clip(0,1)
            loss += mae_loss(outputs, labels.to(device))
            
            nrmse += normalized_root_mse(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            psnr += peak_signal_noise_ratio(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            ssim += structural_similarity(labels.squeeze().detach().cpu().numpy(), outputs.squeeze().detach().cpu().numpy())
            
            img_stack.append(imgs), out_stack.append(outputs.cpu()), label_stack.append(labels)

        nrmse /= test_len
        psnr /= test_len
        ssim /= test_len
        print(f'Epoch {epoch+1}/{epochs}, Test Loss: {loss/len(sawa_testloader):.4f}, NRMSE: {nrmse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')

        wandb.log({
            'epoch' : epoch, 'nrmse_n' : nrmse,
            'psnr_n' : psnr,   'ssim_n' : ssim 
        })

        
        if epoch % 10 == 0:
            img_grid = torchvision.utils.make_grid(
                torch.cat([torch.cat(img_stack, dim=0), torch.cat(label_stack, dim=0), torch.cat(out_stack, dim=0)], dim = 3) , nrow =2, padding = 20, pad_value = 1
            )
            images = wandb.Image(transforms.ToPILImage()(img_grid.cpu()), caption="Left: Input, Middle : Ground Truth, Right: Prediction")
            wandb.log({f"Test Predictions": images, "epoch" : epoch})
            print(f'Logged predictions to wandb')
    
    
    scheduler.step()

# Save the model checkpoint 
model_name = 'unet_sawa_patch_sli'
torch.save(model.state_dict(), f'artifacts/models/{model_name}.pth')

# Timing
print(f'Time elapsed: {time.time() - start:.2f} seconds')