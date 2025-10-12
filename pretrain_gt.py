'''
coding: utf-8 
        Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
        torch 2.2.2+cu121
Created on 2024-08-16
@title: pretrained_gt.py
@description: Pretrain the VAE model with GT data
@author: <Ronald B Liu>
@github: <https://github.com/RL-arch>
@version: 0.1.0
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
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
from models.vae import *


# class PerceptualLoss(nn.Module):
#     def __init__(self, input_channels, layer_ids, layer_weights):
#         super(PerceptualLoss, self).__init__()
#         self.layer_ids = layer_ids
#         self.layer_weights = layer_weights
#         self.vgg_layers = models.vgg19(pretrained=True).features

#         if input_channels != 3:
#             self.channel_reducer = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=1)
#         else:
#             self.channel_reducer = None  # No need to reduce channels if input is already 3 channels

#     def forward(self, input, target):
#         # Adjust input and target channels if necessary
#         if self.channel_reducer is not None:
#             input = self.channel_reducer(input)
#             target = self.channel_reducer(target)

#         # Now, process through VGG
#         loss = 0.0
#         x = input
#         y = target
#         for i, (layer_id, weight) in enumerate(zip(self.layer_ids, self.layer_weights)):
#             x = self.vgg_layers[:layer_id + 1](x)
#             y = self.vgg_layers[:layer_id + 1](y)
#             loss += weight * F.mse_loss(x, y)
#         return loss

def data_normalize(scale, data):
    min_val = data.min()
    max_val = data.max()
    return scale * (data - min_val) / (max_val - min_val)

def save_model(model_instance, epoch=None, prefix='GT_DeepVAE'):
    """Helper function to save the model."""
    if epoch is not None:
        suffix = f'_epoch_{epoch}'
    else:
        suffix = ''
    
    file_name = f'./results/Model/GT_pretrain/{prefix}_encoder{suffix}_MSE.pth'
    torch.save(model_instance.encoder.state_dict(), file_name)
    
    file_name = f'./results/Model/GT_pretrain/{prefix}_decoder{suffix}_MSE.pth'
    torch.save(model_instance.decoder.state_dict(), file_name)
    
    file_name = f'./results/Model/GT_pretrain/{prefix}{suffix}_MSE.pth'
    torch.save(model_instance.state_dict(), file_name)
    

if __name__ == '__main__':
    # ## debug
    # x = torch.randn(1, 1, 48, 48)
    # encoder = DeepVAE_Encoder()
    # decoder = DeepVAE_Decoder()
    # encoded = encoder(x)
    
    # model = DeepVAE()
    # output = model(x)
    # print(encoded.shape)  # 应输出torch.Size([1, 1, 48, 48])
    
    #### main
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Pretrain VAE Model')
    parser.add_argument('--gpu', type=lambda x: [int(gpu.strip()) for gpu in x.split(',')], default=[0], help='GPU IDs for training, separated by comma (e.g., --gpu 0,1,2)')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_device_batch_size', type=int, default=49152)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--patience', type=int, default=200)
    
    args = parser.parse_args()
    
    # Load data
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
    print('GT5:', GT5_tensor.shape)

    '''all data'''
    x_train = torch.cat((
            V1_tensor[0:-1500]
            ,V2_tensor[0:-1500]
            ,V3_tensor[0:-1500]
            ,V4_tensor[0:-1500]
            ,V5_tensor[0:-1500]
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
    y_train = torch.cat((GT1_tensor[0:-1500]
                         , GT2_tensor[0:-1500]
                         , GT3_tensor[0:-1500]
                         , GT4_tensor[0:-1500]
                         , GT5_tensor[0:-1500]
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

    data_num = x_train.shape[0]
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    
    # random shuffle 
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    print('y_train shape:', y_train.shape)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size
    
     # load the data 
    train_loader = DataLoader(TensorDataset(y_train, y_train), load_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(y_val, y_val), load_batch_size, shuffle=False, num_workers=4)

    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepVAE()

    print("Using GPUs:", args.gpu)

    if torch.cuda.is_available() and len(args.gpu) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu).to(device)
    else:
        model = model.to(device)

    ##  model and optimizer setup
    # optimizer = torch.optim.AdamW(model.parameters(), 
    #                           lr=args.base_learning_rate * args.batch_size / 256, 
    #                           betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 
    #                             0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    # Model and optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.base_learning_rate,
                                  betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # Reduces LR by half
            patience=10,  # Patience before reducing LR
            verbose=True
        )

    step_count = 0
    optimizer.zero_grad()
    criterion = nn.MSELoss()
    # criterion = PerceptualLoss(input_channels=1, layer_ids=[3, 8, 15, 22], layer_weights=[1.0, 0.8, 0.6, 0.4]).to(device)
    best_loss = float("inf")
    train_losses = []
    val_losses = []
    # patience = 200
    patience = args.patience
    num_epochs_no_improvement = 0
    # use tdqm to visualize the training process
    
    # Training loop
    for e in range(args.total_epoch):
        start_time = time.time()

        model.train()
        train_loss = 0.0

        for data, target in tqdm(train_loader, desc=f'Training Epoch {e+1}/{args.total_epoch}'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f'Validation Epoch {e+1}/{args.total_epoch}'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Update learning rate scheduler with validation loss
        lr_scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        print(f'Epoch [{e+1}/{args.total_epoch}], Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}, Current learning rate: {optimizer.param_groups[0]["lr"]:.8f}, Time: {epoch_time:.2f} s')

        # save loss
        train_loss_file = f'./results/Loss/GT_DeepVAE_train_loss_MSE.txt'
        val_loss_file = f'./results/Loss/GT_DeepVAE_val_loss_MSE.txt'
        np.savetxt(train_loss_file, train_losses, fmt='%.8f')
        np.savetxt(val_loss_file, val_losses, fmt='%.8f')
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='val loss')
        plt.legend()
        plt.savefig(f'./results/Loss/GT_DeepVAE_loss_MSE.png')
        plt.clf()

        # Save model
        if isinstance(model, torch.nn.DataParallel):
            model_instance = model.module
        else:
            model_instance = model

        # Save the best model overall
        if val_loss < best_loss:
            best_loss = val_loss
            num_epochs_no_improvement = 0
            save_model(model_instance)
            print(f'Model saved, with overall best loss {best_loss}.')
        else:
            num_epochs_no_improvement += 1
            print(f'No improvement for {num_epochs_no_improvement} epochs.')
            if num_epochs_no_improvement == patience:
                print(f"Early stopping! Best loss: {best_loss}")
                break

        # Every 10 epochs, save the best model up to this period
        if (e + 1) % 10 == 0:
            save_model(model_instance, epoch=e+1)
            print(f'Model saved for 0-epoch {e+1}.')

    if num_epochs_no_improvement < patience:
        print(f"Training completed without early stopping.")