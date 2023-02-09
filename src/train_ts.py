import torch, os, time, torch, monai, torchvision, wandb
from torchvision.utils import make_grid
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
wandb_log = False
epochs = 100
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1

# Dataloader
img_transforms = transforms.Compose(
    [
        # transfroms.Normalize()
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=3)
    ]
)
from src.dataloader_ts import LakeDataset, get_nearest_multiple
# Spatial dims : H x W
resize_dims = (get_nearest_multiple(140, 16), get_nearest_multiple(129, 16))
sawa_train = LakeDataset('sawa/landsat8/train', resize_dims=resize_dims, train=True)
sawa_trainloader = DataLoader(sawa_train, batch_size=batch_size, shuffle=True)
sawa_test = LakeDataset('sawa/landsat8/test', resize_dims=resize_dims, train=False)
sawa_testloader = DataLoader(sawa_test, batch_size=1, shuffle=False)

if wandb_log:
    config = {
        'epochs' : epochs,
        'loss' : 'L1',
        'Augumentations' : None,
        'batch_size' : batch_size,
    }
    wandb.login()
    wandb.init(project="lake_forecast", config = config)

from src.model_ts import get_model
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
        # print('training :',imgs.shape,labels.shape)
        outputs = model(imgs.to(device))
        # loss = mse_loss(outputs, labels.to(device))
        loss = mae_loss(outputs.unsqueeze(1), labels.to(device))
        loss.backward()
        optimizer.step()
        # break
        if i % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Step {i+1}/{len(sawa_trainloader)}, Loss: {loss.item():.4f}')
            

    # predict next image prediction every 2 epochs
    if epoch % 5 == 0:
        model.eval()
        nrmse, psnr, ssim, loss = 0, 0, 0, 0
        test_len = len(sawa_testloader)
        
        img_stack, out_stack, label_stack  = [], [], []

        for i, (imgs, labels) in enumerate(sawa_testloader):
            outputs = model(imgs.to(device))
            # outputs = sliding_window_inference(inputs=imgs.to(device), roi_size=(160, 160), sw_batch_size=4,
                                                # predictor=model, overlap = 0.5, mode = 'gaussian', device=device)
            outputs=outputs.clip(0,1).unsqueeze(1)
            # print(f'output shape {outputs.shape} and labels shape is {labels.shape}')
            loss += mae_loss(outputs, labels.to(device))
            
            nrmse += normalized_root_mse(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            psnr += peak_signal_noise_ratio(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            ssim += structural_similarity(labels.squeeze().detach().cpu().numpy(), outputs.squeeze().detach().cpu().numpy())
            
            img_stack.append(imgs), out_stack.append(outputs.cpu()), label_stack.append(labels)

        nrmse /= test_len
        psnr /= test_len
        ssim /= test_len
        print(f'Epoch {epoch+1}/{epochs}, Test Loss: {loss/len(sawa_testloader):.4f}, NRMSE: {nrmse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')

        if wandb_log:
            wandb.log({
                'epoch' : epoch, 'nrmse_n' : nrmse,
                'psnr_n' : psnr,   'ssim_n' : ssim 
            })

        # print(f'img_stack : {img_stack[0].shape,len(img_stack)} label_stack : {label_stack[0].shape,len(label_stack)} out_stack: {out_stack[0].shape,len(label_stack)}')

        # print(f'torch.cat(img_stack, dim=0):{torch.cat(img_stack, dim=0).shape} ,torch.cat(label_stack, dim=0):{torch.cat(label_stack, dim=0).shape},torch.cat(out_stack, dim=0):{torch.cat(out_stack, dim=0).shape}')
        
        # imgs :  1, 1, 3, 416, 384
        # label : B 1 1 H W
        # pred :  B 1 1 H W
        if epoch % 10 == 0 and wandb_log:

          img_stack = torch.cat(img_stack, dim=0)
          label_stack = torch.cat(label_stack, dim=0).squeeze(dim=1)
          out_stack = torch.cat(out_stack, dim=0).squeeze(dim=1)
        
          
          img0_stack = img_stack[:,:,0]
          img1_stack = img_stack[:,:,1]
          img2_stack = img_stack[:,:,2]
        
          f = make_grid(
            torch.cat(
              [img0_stack, img1_stack, img2_stack, label_stack, out_stack], dim=3
            ), nrow=1, padding=15, pad_value=1
          )
          images = wandb.Image(f, caption="First three : Input, Fourth : Ground Truth, Last: Prediction")
          wandb.log({f"Test Predictions": images, "epoch" : epoch})
          print(f'Logged predictions to wandb')
    
    # break
    scheduler.step()

# Save the model checkpoint 
model_name = 'unet_sawa_patch_sli'
torch.save(model.state_dict(), f'artifacts/models/{model_name}.pth')

# Timing
print(f'Time elapsed: {time.time() - start:.2f} seconds')