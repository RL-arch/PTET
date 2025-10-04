'''
coding: utf-8 
        Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
        torch 2.2.2+cu121
Created on 2024-08-16
@title: finetune_mlp_S.py
@description: finetune the linear projection model with MLP
@author: <|Ronald B Liu|liu.ronald@icloud.com|>
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
from models.mae import *
from models.vae import *
import random

'''with hid layer'''
# class Linear(nn.Module):
#     def __init__(self, input_dim=16640, hidden_dim1=4096, output_dim=128 * 6 * 6, dropout_rate=0.2):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim1, output_dim),
#         )
'''direct projection'''
class Linear(nn.Module):
    def __init__(self, input_dim=16640, output_dim=128 * 6 * 6, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.layers(x)

class MLP_Projection(nn.Module):
    def __init__(
                    self, 
                    encoder,
                    decoder,
                    i_dim=16640,
                    hidden_dim1=4096,
                    o_dim=128 * 6 * 6,
                    dropout_rate=0.5,
                ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection = Linear(
                                    input_dim=i_dim,
                                    output_dim=o_dim,
                                    dropout_rate=dropout_rate
                                )
        self.flatten = nn.Flatten()

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        new_features = rearrange(features, 't b c -> b t c')
        # print('new_features:', new_features.shape)
        encoded = self.flatten(new_features)
        projected = self.projection(encoded)
        projected = projected.view(-1, 128, 6, 6)
        decoded = self.decoder(projected)
        return decoded

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def data_normalize(scale, data):
    """
    Normalize data to a specified range.

    Args:
        scale (float): The range of the normalized data, e.g., 255 means normalization to 0-255.
        data (numpy.ndarray): The data to be normalized, should be a numpy array.

    Returns:
        numpy.ndarray: The normalized data, with the same shape as the input.
    """
    min_val = data.min()
    max_val = data.max()
    return scale * (data - min_val) / (max_val - min_val)

import torch.nn.functional as F

def sensitivity_loss(device, sensitivity_matrix, output, target):
    # print('output:', output.shape)
    # print('target:', target.shape)
    
    # Step 1: Flatten the output and target to match sensitivity matrix
    output_flattened = output.view(output.size(0), -1)  # Shape: (b, 2304)
    target_flattened = target.view(target.size(0), -1)  # Shape: (b, 2304)
    
    # Step 3: Expand sensitivity_matrix to match the batch size
    sensitivity_matrix_expanded = sensitivity_matrix.unsqueeze(0).expand(output_flattened.size(0), -1, -1)  # [b, 104, 2304]

    # Step 4: Perform the matrix multiplication and compute the loss
    output_transformed = torch.matmul(sensitivity_matrix_expanded, output_flattened.unsqueeze(2))  # [b, 104, 1]
    target_transformed = torch.matmul(sensitivity_matrix_expanded, target_flattened.unsqueeze(2))  # [b, 104, 1]
    
    # Step 5: Normalize the transformed output and target to 0-1
    output_transformed = F.normalize(output_transformed, p=2, dim=1)
    target_transformed = F.normalize(target_transformed, p=2, dim=1)
    
    # Step 6: Calculate the loss
    loss = torch.mean(torch.abs(output_transformed - target_transformed))
    # print('sens loss:', loss)
    return loss
    

if __name__ == "__main__":
    ###### Sensitivity matrix laod ######
    sensitivity_matrix = np.loadtxt('./Sensitivity16_48.txt').astype(np.float32)

    # set random seed for reproducibility
    set_seed(42)

    parser = argparse.ArgumentParser(description='Train Projection Model')
    ####### model-define related params ###############################
    parser.add_argument('--input_dim', 
                        type=int, 
                        default=16640, 
                        help='if mask_ratio=0, input_dim=(256+1)*256=65792; else if mask_ratio=0.75, input_dim=(256+1)*256*(1-0.75)=16640')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='mask ratio set for finetuning')
    parser.add_argument('--hidden_dim1', type=int, default=4096, help='hidden layer 1 dimension')
    parser.add_argument('--ep_num', type=int, default=250, help='best saved model epoch of GT pretrain')
    ####### training hyp params ##############################
    parser.add_argument('--s_ratio', type=float, default=0.3, help='sensitivity loss ratio')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--patience', type=int, default=200, help='early stopping patience (default: 200)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    '''
    if mask_ratio=0, 
    input_dim=(256+1)*256=65792
    
    elif mask_ratio=0.75, 
    input_dim=(256+1)*256*(1-0.75)=16640
    '''
    ae_encoder = MAE_Encoder(mask_ratio=args.mask_ratio) 
    ae_decoder = DeepVAE_Decoder()
    
    
    # ep_num = 250 # ! test best epoch of GT pretrain 
    ep_num = args.ep_num
    # Load pre-trained weights
    ae_encoder.load_state_dict(torch.load("./results/Model/V_pretrain/MAE_img64_encoder_mr0.75_eb256_data492500.pth"), strict=False)
    # ae_decoder.load_state_dict(torch.load("./results/Model/GT_pretrain/GT_DeepVAE_decoder.pth"), strict=False)
    ae_decoder.load_state_dict(torch.load(f"./results/Model/GT_pretrain/GT_DeepVAE_decoder_{ep_num}_MSE.pth"), strict=False)

    # Freeze the MAE encoder parameters
    for param in ae_encoder.parameters():
        param.requires_grad = False

    # Unfreeze the DeepVAE decoder parameters
    for param in ae_decoder.parameters():
        param.requires_grad = True

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

    '''500*5 data'''
    x_train = torch.cat((
            V1_tensor[0:500]
            ,V2_tensor[0:500]
            ,V3_tensor[0:500]
            ,V4_tensor[0:500]
            ,V5_tensor[0:500]
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Projection(
                            encoder=ae_encoder,
                            decoder=ae_decoder,
                            i_dim =args.input_dim
                        ).to(device)
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=2000)
    
    # keep ReduceLROnPlateau for further adjusting learning rate when validation loss stagnates
    scheduler_plateau = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=True)

    train_losses = []
    val_losses = []
    num_epochs = 2000
    # threshold = 0.5
    # patience = 200 # early stopping patience
    patience = args.patience
    num_epochs_no_improvement = 0
    best_loss = float('inf')

    start_time = time.time()

    lambda_s = args.s_ratio
    # beta = 0.2
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            recon_loss = criterion(output, target)
            sens_loss = sensitivity_loss(device, torch.tensor(sensitivity_matrix).to(device), output, target)
            # print('sens_loss: ',sens_loss)
            # total_loss = lambda_s * sens_loss + (1 - lambda_s - beta) * recon_loss + beta * nn.SmoothL1Loss()(output, target)
            total_loss = lambda_s * sens_loss + (1 - lambda_s) * recon_loss
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        # scheduler_plateau.step()
        scheduler.step()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        end_time = time.time()
        elapsed_time = end_time - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Time elapsed: {elapsed_time:.2f} seconds, Learning Rate: {current_lr:.8f}')

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation', unit='batch'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                # binary_val_outputs = (output > threshold).float()

                recon_loss = criterion(output, target)
                val_loss += recon_loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}')

        # scheduler.step(val_loss)
        scheduler_plateau.step(val_loss)
        start_time = time.time()
        
        np.savetxt(f'./results/Loss/finetune_img64_S_mlp_dvae_train_loss_mr{args.mask_ratio}_{x_train.shape[0]}data_ep{ep_num}_bch{args.batch_size}.txt', train_losses, fmt='%.8f')
        np.savetxt(f'./results/Loss/finetune_img64_S_mlp_dvae_val_loss_mr{args.mask_ratio}_{x_train.shape[0]}data_ep{ep_num}_bch{args.batch_size}.txt', val_losses, fmt='%.8f')

        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='val loss')
        plt.legend()
        plt.savefig(f'./results/Loss/finetune_img64_S_mlp_dvae_loss_mr{args.mask_ratio}_{x_train.shape[0]}data_ep{ep_num}_bch{args.batch_size}.png')
        plt.clf()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'./results/Model/Finetune/finetune_img64_S_mlp_dvae_mr{args.mask_ratio}_{x_train.shape[0]}data_ep{ep_num}_bch{args.batch_size}.pth')
            print(f'------> Model saved. Best val loss is {best_loss:.8f} <---------------------------------------------')
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1
            print(f'No improvement for {num_epochs_no_improvement} epochs.')
            if num_epochs_no_improvement == patience:
                print(f"Early stopping! with best loss {best_loss}")
                break

    if num_epochs_no_improvement < patience:
        print("Training completed without early stopping.")
