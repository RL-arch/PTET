'''
coding: utf-8 
        Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
        torch 2.2.2+cu121
Created on 2024-08-16
@title: finetune_ldm.py
@description: finetune the VAE (variational autoencoder with Resnet) model
@author: <|Ronald B Liu|liu.ronald@icloud.com|>
@github: <https://github.com/RL-arch>
@version: 0.1.0
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import scipy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
from tqdm import tqdm
import math

# class GaussianNoise(nn.Module):
#     def __init__(self, sigma=0.1, is_relative_detach=True):
#         super(GaussianNoise, self).__init__()
#         self.sigma = sigma
#         self.is_relative_detach = is_relative_detach

#     def forward(self, x):
#         if self.training and self.sigma != 0:
#             scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
#             noise = torch.randn_like(x) * scale
#             x = x + noise
#         return x


class Resnet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.s = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
        )
        self.res = None
        if dim_in != dim_out:
            self.res = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        res = x
        if self.res:
            res = self.res(x)
        return res + self.s(x)
    
class Atten(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=128,
                                       num_groups=32,
                                       eps=1e-6,
                                       affine=True)

        self.q = torch.nn.Linear(128, 128)
        self.k = torch.nn.Linear(128, 128)
        self.v = torch.nn.Linear(128, 128)
        self.out = torch.nn.Linear(128, 128)

    def forward(self, x):
        res = x
        # print('atten input:', res.shape) #[1, 128, 12, 12]

        #norm, dimension unchanged
        #[1, 512, 64, 64]
        x = self.norm(x)

        x = x.flatten(start_dim=2).transpose(1, 2)
        # print('qkv input:', x.shape) #[1, 144, 128]

        # linear calculation, dimension unchanged
        #[1, 4096, 512]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        #[1, 144, 128] -> [1, 128, 144]
        k = k.transpose(1, 2)

        #[1, 4096, 512] * [1, 512, 4096] -> [1, 4096, 4096]
        #0.044194173824159216 = 1 / 512**0.5
        # 1 / 128**0.5 = 0.08838834764831843
        #atten = q.bmm(k) * (1/ dim**0.5)

        # equal to the equation above, but more precise
        atten = torch.baddbmm(torch.empty(1, 144, 144, device=q.device),
                              q,
                              k,
                              beta=0,
                              alpha=0.08838834764831843) # ! to change

        atten = torch.softmax(atten, dim=2)

        atten = atten.bmm(v)

        # linear calculation, dimension unchanged
        #[1, 4096, 512]
        atten = self.out(atten)
        # print('atten output:', atten.shape) #[1, 144, 128]

        #[1, 144, 128] -> [1, 128, 144] -> [1, 128, 12, 12]
        atten = atten.transpose(1, 2).reshape(-1, 128, 12, 12)

        # residual connection, dimension unchanged
        atten = atten + res
        # print('atten output:', atten.shape) #[1, 128, 12, 12]

        return atten

class DeepVAE_Encoder(nn.Module):
    def __init__(self):
        super(DeepVAE_Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Sequential(
                Resnet(32, 32),
                Resnet(32, 32),
                nn.Conv2d(32, 32, 3, stride=2, padding=1),
            ),
            nn.Sequential(
                Resnet(32, 64),
                Resnet(64, 64),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
            ),
            nn.Sequential(
                Resnet(64, 128),
                Resnet(128, 128),
                nn.Conv2d(128, 128, 3, stride=2, padding=1),
            ),
        )
    
    def forward(self, x):
        return self.model(x)

class DeepVAE_Decoder(nn.Module):
    def __init__(self):
        super(DeepVAE_Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Sequential(
                Resnet(128, 128),
                Atten(),
                Resnet(128, 128),
            ),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Sequential(
                Resnet(64, 64),
                Resnet(64, 64),
            ),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Sequential(
                Resnet(32, 32),
                Resnet(32, 32),
            ),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        # print('decoder input:', x.shape)
        return self.model(x)

class DeepVAE(nn.Module):
    def __init__(self):
        super(DeepVAE, self).__init__()
        self.encoder = DeepVAE_Encoder()
        self.decoder = DeepVAE_Decoder()
        # self.noise = GaussianNoise(0.05)  # add noise layer

    def forward(self, x):
        # x = self.noise(x)
        h = self.encoder(x)
        h = self.decoder(h)
        return h


if __name__ == '__main__':
    ## debug
    x = torch.randn(1, 1, 48, 48)
    encoder = DeepVAE_Encoder()
    decoder = DeepVAE_Decoder()
    encoded = encoder(x)
    decoded = decoder(encoded)
    print('decoded', decoded.shape)  # ([1, 1, 48, 48])
    
    # model = DeepVAE()
    # output = model(x)
    print('encoded', encoded.shape)  # ([1, 128, 6, 6])
    
    # #### main
    # # Command-line arguments
    # parser = argparse.ArgumentParser(description='Train ViT CAE Projection Model')
    # parser.add_argument('--gpu', type=lambda x: [int(gpu.strip()) for gpu in x.split(',')], default=[0], help='GPU IDs for training, separated by comma (e.g., --gpu 0,1,2)')
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--max_device_batch_size', type=int, default=49152)
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    # parser.add_argument('--weight_decay', type=float, default=0.05)
    # parser.add_argument('--mask_ratio', type=float, default=0.6)
    # parser.add_argument('--total_epoch', type=int, default=2000)
    # parser.add_argument('--warmup_epoch', type=int, default=200)
    
    
    # args = parser.parse_args()
    
    
    # # load data
    # '''all data'''
    # V0_tensor = torch.tensor(np.load('Data/V/V_EIM0.npy')).float()
    # V1_tensor = torch.tensor(np.load('Data/V/V_EIM1.npy')).float()
    # V2_tensor = torch.tensor(np.load('Data/V/V_EIM2.npy')).float()
    # V3_tensor = torch.tensor(np.load('Data/V/V_EIM3.npy')).float()
    # V4_tensor = torch.tensor(np.load('Data/V/V_EIM4.npy')).float()
    # V5_tensor = torch.tensor(np.load('Data/V/V_EIM5.npy')).float()

    # GT0_tensor = torch.tensor(np.load('Data/GT/GT0.npy')).float()
    # GT1_tensor = torch.tensor(np.load('Data/GT/GT1.npy')).float()
    # GT2_tensor = torch.tensor(np.load('Data/GT/GT2.npy')).float()
    # GT3_tensor = torch.tensor(np.load('Data/GT/GT3.npy')).float()
    # GT4_tensor = torch.tensor(np.load('Data/GT/GT4.npy')).float()
    # GT5_tensor = torch.tensor(np.load('Data/GT/GT5.npy')).float()
    # print('GT5:',GT5_tensor.shape)

    # x_train = torch.cat((
    #                     V1_tensor[0:-1500], 
    #                     V2_tensor[0:-1500], 
    #                     V3_tensor[0:-1500], 
    #                     V4_tensor[0:-1500],
    #                     V5_tensor[0:-1500]), 
    #                     dim=0)
    # x_val = torch.cat((
    #                     V1_tensor[-1500:-500],
    #                     V2_tensor[-1500:-500],
    #                     V3_tensor[-1500:-500],
    #                     V4_tensor[-1500:-500],
    #                     V5_tensor[-1500:-500]), 
    #                     dim=0)
    # x_test = torch.cat((
    #                     V1_tensor[-500:],
    #                     V2_tensor[-500:],
    #                     V3_tensor[-500:],
    #                     V4_tensor[-500:],
    #                     V5_tensor[-500:]), 
    #                     dim=0)
    # y_train = torch.cat((
    #                     GT1_tensor[0:-1500], 
    #                     GT2_tensor[0:-1500], 
    #                     GT3_tensor[0:-1500], 
    #                     GT4_tensor[0:-1500],
    #                     GT5_tensor[0:-1500]), 
    #                     dim=0)
    # y_val = torch.cat((
    #                     GT1_tensor[-1500:-500],
    #                     GT2_tensor[-1500:-500],
    #                     GT3_tensor[-1500:-500],
    #                     GT4_tensor[-1500:-500],
    #                     GT5_tensor[-1500:-500]), 
    #                     dim=0)
    # y_test = torch.cat((
    #                     GT1_tensor[-500:], 
    #                     GT2_tensor[-500:], 
    #                     GT3_tensor[-500:], 
    #                     GT4_tensor[-500:],
    #                     GT5_tensor[-500:]), 
    #                     dim=0)

    # data_num = x_train.shape[0]
    # batch_size = args.batch_size
    # load_batch_size = min(args.max_device_batch_size, batch_size)

    # assert batch_size % load_batch_size == 0
    # steps_per_update = batch_size // load_batch_size

    #  # prepare data loaders
    # train_loader = DataLoader(TensorDataset(y_train, y_train), load_batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(TensorDataset(y_val, y_val), load_batch_size, shuffle=False, num_workers=4)

    # # Set the CUDA_VISIBLE_DEVICES environment variable
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))

    # # Training setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DeepVAE()

    # print("Using GPUs:", args.gpu)

    # if torch.cuda.is_available() and len(args.gpu) > 1:
    #     model = nn.DataParallel(model, device_ids=args.gpu).to(device)
    # else:
    #     model = model.to(device)

    # # Model and optimizer setup
    # optimizer = torch.optim.AdamW(model.parameters(), 
    #                           lr=args.base_learning_rate * args.batch_size / 256, 
    #                           betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 
    #                             0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    # step_count = 0
    # optimizer.zero_grad()
    # criterion = nn.MSELoss()
    # best_loss = float("inf")
    # train_losses = []
    # val_losses = []

    # # Use tqdm in the training loop
    # for e in range(args.total_epoch):
    #     start_time = time.time()
    #     model.train()
    #     train_loss = 0.0
    #     # Apply tqdm on the data loader
    #     for data, target in tqdm(train_loader, desc=f'Training Epoch {e+1}/{args.total_epoch}'):
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #     lr_scheduler.step()
    #     current_lr = lr_scheduler.get_last_lr()
    #     train_loss /= len(train_loader)
    #     train_losses.append(train_loss)

    #     model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         # Apply tqdm on the validation data loader
    #         for data, target in tqdm(val_loader, desc=f'Validation Epoch {e+1}/{args.total_epoch}'):
    #             data, target = data.to(device), target.to(device)
    #             output = model(data)
    #             val_loss += criterion(output, target).item()
    #     val_loss /= len(val_loader)
    #     val_losses.append(val_loss)

    #     epoch_time = time.time() - start_time

    #     print(  f'Epoch [{e+1}/{args.total_epoch}], '
    #             f'Training Loss: {train_loss:.8f}, '
    #             f'Validation Loss: {val_loss:.8f}, '
    #             f'Current learning rate: {current_lr}, '
    #             f'Time: {epoch_time:.2f} s')

        
    #     ''' save model '''
    #     # Suppose model is a DataParallel object
    #     if isinstance(model, torch.nn.DataParallel):
    #         # Get the model instance inside DataParallel
    #         model_instance = model.module
    #     else:
    #         # If model is not a DataParallel object, use it directly
    #         model_instance = model

    #     if val_loss < best_loss:
    #         best_loss = val_loss
    #         file_name = f'Model/GT_pretrain/GT_DeepVAE_encoder.pth'
    #         torch.save(model_instance.encoder.state_dict(), file_name)
    #         file_name = f'Model/GT_pretrain/GT_DeepVAE_decoder.pth'
    #         torch.save(model_instance.decoder.state_dict(), file_name)
    #         file_name = f'Model/GT_pretrain/GT_DeepVAE.pth'
    #         torch.save(model_instance.state_dict(), file_name)
    #         print('Model saved.')
    

    # # Use string formatting to add variables to the file name
    # train_loss_file = f'Loss/GT_DeepVAE_train_loss.txt'
    # val_loss_file = f'Loss/GT_DeepVAE_val_loss.txt'

    # # Save the losses to text files
    # np.savetxt(train_loss_file, train_losses, fmt='%.8f')
    # np.savetxt(val_loss_file, val_losses, fmt='%.8f')

    # # plot the loss
    # plt.plot(train_losses, label='train loss')
    # plt.plot(val_losses, label='val loss')
    # plt.legend()
    # plt.savefig(f'Loss/GT_DeepVAE_loss.png')
    