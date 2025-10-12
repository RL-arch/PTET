'''
coding: utf-8 
        Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
        torch 2.2.2+cu121
Created on 2024-08-16
@title: finetune_ldm.py
@description: finetune the LDM (Latent Diffusion Model) model
@author: <Ronald B Liu>
@github: <https://github.com/RL-arch>
@version: 0.1.0
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.mae import MAE_Encoder
from einops import rearrange
from models.vae import DeepVAE_Encoder, DeepVAE_Decoder, DeepVAE
from models.ldm import UNet
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_noise(features, noise, noise_step, betas):
    alpha_cumprod = torch.cumprod(1 - betas, dim=0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

    alpha_t = sqrt_alpha_cumprod[noise_step].view(-1, 1, 1, 1)
    one_minus_alpha_t = sqrt_one_minus_alpha_cumprod[noise_step].view(-1, 1, 1, 1)
    
    return alpha_t * features + one_minus_alpha_t * noise

def get_loss(x, y, encoder, vae, unet, criterion, device, epoch, batch_size, betas):
    if epoch < 0:
        raise ValueError("Epoch must be non-negative")
    # extract features from the MAE encoder
    features, backward_indexes = encoder(x)
    # [65, b, 128] -> [b, 65, 128]
    out_encoder = rearrange(features, 't b c -> b t c')

    out_vae = vae.encoder(y)
    #0.18215 = vae.config.scaling_factor
    out_vae = out_vae * 0.18215
    
    # random number, the calculate target 
    noise = torch.randn_like(out_vae)

    # add noise to the VAE output
    #1000 = scheduler.num_train_timesteps
    #1 = batch size
    noise_step = torch.randint(0, len(betas), (out_vae.size(0), )).long().to(device)
    # noise_step = torch.randint(0, 1000, (1, )).long().to(device)
    out_vae_noise = add_noise(out_vae, noise, noise_step, betas)

    # Calulate the output of the UNet using the voltage information 
    out_unet = unet(out_vae=out_vae_noise,
                    out_encoder=out_encoder,
                    time=noise_step)
    
    #[1, 128, 6, 6],[1, 128, 6, 6]

    return criterion(out_unet, noise)

def get_img_loss(x, y, encoder, vae_enc, vae_dec, unet, criterion, device, epoch, batch_size, betas):
    if epoch < 0:
        raise ValueError("Epoch must be non-negative")
    
    features, backward_indexes = encoder(x)
    out_encoder = rearrange(features, 't b c -> b t c')
    out_vae = vae_enc(y)
    # Generate **pure noise**
    noise = torch.randn_like(out_vae)
    # print('noise shape:', out_vae.shape)[8, 128, 6, 6]
    # Get random noise step
    noise_step = torch.randint(0, len(betas), (out_vae.size(0), )).long().to(device)
    # Add noise to the VAE output
    out_vae_noise = add_noise(out_vae, noise, noise_step, betas)
    # Pass through the UNet
    out_unet = unet(out_vae=out_vae_noise,
                    out_encoder=out_encoder,
                    time=noise_step)
    # Decode the output
    out_img = vae_dec(out_unet)

    return criterion(out_img, y)

    

def data_normalize(scale, data):
    min_val = data.min()
    max_val = data.max()
    return scale * (data - min_val) / (max_val - min_val)

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = 1. / warmup_epochs
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) * self.warmup_factor for base_lr in self.base_lrs]
        else:
            epoch = np.clip(self.last_epoch - self.warmup_epochs, 0, self.max_epochs - self.warmup_epochs)
            return [base_lr * (0.5 * (1. + np.cos(np.pi * epoch / (self.max_epochs - self.warmup_epochs)))) for base_lr in self.base_lrs]

if __name__ == '__main__':
    # set random seed for reproducibility
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (default: 0.01)')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps (default: 1e-8)')
    parser.add_argument('--num_epochs', type=int, default=4000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--gpu', type=lambda x: [int(gpu.strip()) for gpu in x.split(',')], default=[0], help='GPU IDs for training, separated by comma (e.g., --gpu 0,1,2)')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--beta_start', type=float, default=0.00085, help='Starting value of beta for diffusion process')
    parser.add_argument('--beta_end', type=float, default=0.012, help='Ending value of beta for diffusion process')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of timesteps for diffusion process')
    
    args = parser.parse_args()
    
    V0_tensor = torch.tensor(np.load('./data/V/V_EIM0_64.npy')).float()
    V1_tensor = torch.tensor(np.load('./data/V/V_EIM1_64.npy')).float()
    V2_tensor = torch.tensor(np.load('./data/V/V_EIM2_64.npy')).float()
    V3_tensor = torch.tensor(np.load('./data/V/V_EIM3_64.npy')).float()
    V4_tensor = torch.tensor(np.load('./data/V/V_EIM4_64.npy')).float()
    V5_tensor = torch.tensor(np.load('./data/V/V_EIM5_64.npy')).float()

    GT0_tensor = torch.tensor(np.load('./data/GT/GT0.npy')).float()
    GT1_tensor = torch.tensor(np.load('./data/GT/GT1.npy')).float()
    GT2_tensor = torch.tensor(np.load('./data/GT/GT2.npy')).float()
    GT3_tensor = torch.tensor(np.load('./data/GT/GT3.npy')).float()
    GT4_tensor = torch.tensor(np.load('./data/GT/GT4.npy')).float()
    GT5_tensor = torch.tensor(np.load('./data/GT/GT5.npy')).float()
    # print('GT5:', GT5_tensor.shape)

    x_train = torch.cat((
            V1_tensor[0:500]
            , V2_tensor[0:500]
            , V3_tensor[0:500]
            , V4_tensor[0:500]
            , V5_tensor[0:500]
        ), dim=0)
    x_val = torch.cat((V1_tensor[-1500:-500]
                       , V2_tensor[-1500:-500]
                       , V3_tensor[-1500:-500]
                       , V4_tensor[-1500:-500]
                       , V5_tensor[-1500:-500]
                       ), dim=0)
    x_test = torch.cat((V1_tensor[-500:]
                        , V2_tensor[-500:]
                        , V3_tensor[-500:]
                        , V4_tensor[-500:]
                        , V5_tensor[-500:]
                        ), dim=0)
    y_train = torch.cat((GT1_tensor[0:500]
                         , GT2_tensor[0:500]
                         , GT3_tensor[0:500]
                         , GT4_tensor[0:500]
                         , GT5_tensor[0:500]
                         ), dim=0)
    y_val = torch.cat((GT1_tensor[-1500:-500]
                       , GT2_tensor[-1500:-500]
                       , GT3_tensor[-1500:-500]
                       , GT4_tensor[-1500:-500]
                       , GT5_tensor[-1500:-500]
                       ), dim=0)
    y_test = torch.cat((GT1_tensor[-500:]
                        , GT2_tensor[-500:]
                        , GT3_tensor[-500:]
                        , GT4_tensor[-500:]
                        , GT5_tensor[-500:]
                        ), dim=0)
    
    y_train = data_normalize(scale=1, data=y_train)
    y_val = data_normalize(scale=1, data=y_val)
    y_test = data_normalize(scale=1, data=y_test)
    
    # random shuffle the data
    indices_train = np.arange(len(y_train))
    indices_val = np.arange(len(y_val))
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_val)
    x_train = x_train[indices_train]
    y_train = y_train[indices_train]
    x_val = x_val[indices_val]
    y_val = y_val[indices_val]
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, num_workers=args.num_workers)

    encoder = MAE_Encoder()
    vae_encoder = DeepVAE_Encoder()
    vae_decoder = DeepVAE_Decoder()
    unet = UNet()
    
    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using GPUs:", args.gpu)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder, device_ids=[i for i in range(torch.cuda.device_count())]).to(device)
        vae_encoder = nn.DataParallel(vae_encoder, device_ids=[i for i in range(torch.cuda.device_count())]).to(device)
        vae_decoder = nn.DataParallel(vae_decoder, device_ids=[i for i in range(torch.cuda.device_count())]).to(device)
        unet = nn.DataParallel(unet, device_ids=[i for i in range(torch.cuda.device_count())]).to(device)
    else:
        encoder = encoder.to(device)
        vae_encoder = vae_encoder.to(device)
        vae_decoder = vae_decoder.to(device)
        unet = unet.to(device)


    # encoder.to(device)
    # vae.to(device)
    # unet.to(device)
    #  Initialize betas
    # betas = torch.linspace(args.beta_start, args.beta_end, args.num_timesteps).to(device)
    ''' B: diffusion noise param, control noise step '''
    betas = (
                torch.linspace(
                                args.beta_start**0.5,
                                args.beta_end**0.5,
                                args.num_timesteps,
                                dtype=torch.float32
                                ) ** 2
            ).to(device)
    
    encoder.load_state_dict(torch.load('./results/Model/V_pretrain/MAE_img64_encoder_mr0.75_eb256_data492500.pth'), strict=False)
    # vae.load_state_dict(torch.load('Model/GT_pretrain/GT_DeepVAE_50_new.pth'), strict=False)
    vae_encoder.load_state_dict(torch.load('./results/Model/GT_pretrain/GT_DeepVAE_encoder_2000_MSE.pth'), strict=False)
    vae_decoder.load_state_dict(torch.load('./results/Model/GT_pretrain/GT_DeepVAE_decoder_2000_MSE.pth'), strict=False)
    
    # Frozen = False
    for param in encoder.parameters():
        param.requires_grad = False

    for param in vae_encoder.parameters():
        param.requires_grad = False
    
    for param in vae_decoder.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(unet.parameters(),
                            lr=args.lr,
                            betas=(0.9, 0.999),
                            weight_decay=args.weight_decay,
                            eps=args.eps)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    # optimizer = optim.Adam(unet.parameters(), lr=0.001)
    # scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=2000)
    # optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    img_rec_losses = []
    patience = args.patience
    num_epoch = args.num_epochs
    num_epochs_no_improvement = 0
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epoch):
        unet.train()
        train_loss = 0

        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            # loss = get_loss(
            loss = get_img_loss(
                x=x,
                y=y,
                encoder=encoder, 
                vae_enc=vae_encoder,
                vae_dec=vae_decoder, 
                unet=unet,
                criterion=criterion,
                device=device,
                epoch=epoch,
                batch_size=args.batch_size,
                betas=betas
            )
            loss.backward()
            nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        # current_lr = optimizer.param_groups[0]['lr']
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epoch}], Train Loss: {train_loss:.8f}, LR: {current_lr:.8f}')

        unet.eval()
        encoder.eval()
        vae_encoder.eval()
        vae_decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc='Validation', unit='batch'):
                x, y = x.to(device), y.to(device)
                # loss = get_loss(
                loss = get_img_loss(
                    x=x,
                    y=y,
                    encoder=encoder, 
                    vae_enc=vae_encoder,
                    vae_dec=vae_decoder, 
                    unet=unet,
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    batch_size=args.batch_size,
                    betas=betas
                )
                val_loss += loss.item()
                end_time = time.time()
        elapsed_time = end_time - start_time
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'\n LDM Val Loss: {val_loss:.8f}, Time elapsed: {elapsed_time/60:.2f} min.\n')

        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num_epochs_no_improvement = 0
            torch.save(unet.state_dict(), f'./results/Model/Finetune/ldm_{x_train.shape[0]}data.pth')
            print(f'------> Model saved. Best val loss is {best_val_loss:.8f} <---------------------------------------------')

            np.savetxt(f'./results/Loss/finetune_ldm_train_loss_{x_train.shape[0]}data.txt', train_losses, fmt='%.8f')
            np.savetxt(f'./results/Loss/finetune_ldm_val_loss_{x_train.shape[0]}data.txt', val_losses, fmt='%.8f')

            plt.plot(train_losses, label='train loss')
            plt.plot(val_losses, label='val loss')
            plt.legend()
            plt.savefig(f'./results/Loss/finetune_ldm_loss_{x_train.shape[0]}data.png')
            plt.clf()
        else:
            num_epochs_no_improvement += 1
            print(f'No improvement for {num_epochs_no_improvement} epochs.')
            if num_epochs_no_improvement >= patience:
                print(f'Early stopping at epoch {epoch + 1} with loss {best_val_loss}')
                break
        
    if num_epochs_no_improvement < patience:
        print("Training completed without early stopping.")
