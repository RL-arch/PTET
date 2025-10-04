'''
coding: utf-8 
        Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
        torch 2.2.2+cu121
Created on 2024-08-16
@title: finetune_mlp.py
@description: finetune the linear projection model with MLP
@author: <|Ronald B Liu|liu.ronald@icloud.com|>
@github: <https://github.com/RL-arch>
@version: 0.1.0
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                    encoder,     # Teacher model (MAE encoder)
                    decoder,     # Student model (DeepVAE decoder)
                    i_dim=16640,
                    hidden_dim1=4096,
                    o_dim=128 * 6 * 6,
                    dropout_rate=0.5,
                ):
        super().__init__()
        self.encoder = encoder   # Pre-trained MAE encoder (teacher)
        self.decoder = decoder   # DeepVAE decoder (student)
        self.projection = Linear(
                                    input_dim=i_dim,
                                    output_dim=o_dim,
                                    dropout_rate=dropout_rate
                                )
        self.flatten = nn.Flatten()

    def forward(self, img):
        # Teacher model (frozen, no gradients) output
        with torch.no_grad():
            teacher_features, _ = self.encoder(img)
            teacher_output = rearrange(teacher_features, 't b c -> b t c') 

        # Student model forward pass
        features, _ = self.encoder(img)
        new_features = rearrange(features, 't b c -> b t c')
        encoded = self.flatten(new_features)
        projected = self.projection(encoded)
        projected = projected.view(-1, 128, 6, 6)
        decoded = self.decoder(projected)

        return decoded, teacher_output


def distillation_loss(student_output, teacher_output, temperature=1.0):
    """
    Calculates the KL divergence between the student and teacher model outputs.
    Resizes teacher output to match the student's spatial dimensions.
    """
    # print('student_output:', student_output.shape)  # [1, 1, 48, 48]
    # print('teacher_output:', teacher_output.shape)  # [1, 65, 256]

    # Resize teacher output to match student's spatial dimensions (48x48)
    # First, we need to reshape the teacher output if it's not spatial (depends on your teacher model architecture)
    teacher_output = teacher_output.view(teacher_output.shape[0], teacher_output.shape[1], int(teacher_output.shape[2] ** 0.5), -1)
    
    # If needed, interpolate the teacher output to match student output's spatial dimensions (48x48)
    teacher_output_resized = F.interpolate(teacher_output, size=(student_output.shape[2], student_output.shape[3]), mode='bilinear', align_corners=False)

    # Make sure the number of channels match (this may depend on your model design)
    # You can project the teacher output channels down to match the student if necessary
    teacher_output_resized = teacher_output_resized[:, :student_output.shape[1], :, :]

    # Calculate distillation loss (KL divergence)
    return F.kl_div(
        F.log_softmax(student_output / temperature, dim=1),
        F.softmax(teacher_output_resized / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)


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
    min_val = data.min()
    max_val = data.max()
    return scale * (data - min_val) / (max_val - min_val)

if __name__ == "__main__":
    # set random seed for reproducibility
    set_seed(42)

    parser = argparse.ArgumentParser(description='Train Projection Model')
    ####### model-define related params ###############################
    parser.add_argument('--input_dim', 
                        type=int, 
                        default=16640, 
                        help='if mask_ratio=0, input_dim=(256+1)*256=65792; else if mask_ratio=0.75, input_dim=(256+1)*256*(1-0.75)=16640')
    parser.add_argument('--mask_ratio', type=int, default=0.75, help='mask ratio set for finetuning')
    parser.add_argument('--hidden_dim1', type=int, default=4096, help='hidden layer 1 dimension')
    parser.add_argument('--ep_num', type=int, default=2000, help='best saved model epoch of GT pretrain')
    ####### knowledge distillation related params ###############################
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for distillation, higher, teacher softer')
    parser.add_argument('--alpha', type=float, default=0.7, help='weight for distillation loss vs. reconstruction loss')
    ####### training hyp params ##############################
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size (default: 1)')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience (default: 200)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    '''
    if mask_ratio=0, 
    input_dim=(256+1)*256=65792
    
    elif mask_ratio=0.75, 
    input_dim=(256+1)*256*(1-0.75)=16640
    '''
    # Load pre-trained MAE encoder and DeepVAE decoder
    ae_encoder = MAE_Encoder(mask_ratio=args.mask_ratio)  # Teacher
    vae_decoder = DeepVAE_Decoder()  # Student

    
    # ep_num = 250 # ! test best epoch of GT pretrain 
    ep_num = args.ep_num
    # Load pre-trained weights
    ae_encoder.load_state_dict(torch.load("./results/Model/V_pretrain/MAE_img64_encoder_mr0.75_eb256_data492500.pth"), strict=False)
    # ae_decoder.load_state_dict(torch.load("./results/Model/GT_pretrain/GT_DeepVAE_decoder.pth"), strict=False)
    vae_decoder.load_state_dict(torch.load(f"./results/Model/GT_pretrain/GT_DeepVAE_decoder_{ep_num}_MSE.pth"), strict=False)

    # Freeze the MAE encoder parameters
    for param in ae_encoder.parameters():
        param.requires_grad = False

    # Unfreeze the DeepVAE decoder parameters
    for param in vae_decoder.parameters():
        param.requires_grad = True
    
    # Hyperparameters for distillation
    temperature = args.temperature  # Temperature for distillation
    alpha = args.alpha  # Weight for distillation loss vs. reconstruction loss

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
                            decoder=vae_decoder,
                            i_dim =args.input_dim
                        ).to(device)
    criterion = nn.MSELoss() # Reconstruction loss
    # criterion = nn.SmoothL1Loss()
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=2000)
    # 保持ReduceLROnPlateau用于在验证损失停滞时进一步调整学习率
    # keep ReduceLROnPlateau for further adjusting learning rate when validation loss stagnates
    scheduler_plateau = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    num_epochs = 2000
    # threshold = 0.5
    # patience = 200 # early stopping patience
    patience = args.patience
    num_epochs_no_improvement = 0
    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            student_output, teacher_output = model(data)
            # Reconstruction loss
            reconstruction_loss = criterion(student_output, target)
            # Distillation loss (between student and teacher outputs)
            distill_loss = distillation_loss(student_output, teacher_output, temperature)
            # Total loss (combination of distillation and reconstruction losses)
            total_loss = alpha * distill_loss + (1 - alpha) * reconstruction_loss 
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
                student_output, teacher_output = model(data)
                reconstruction_loss = criterion(student_output, target)
                val_loss += reconstruction_loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}')

        # scheduler.step(val_loss)
        scheduler_plateau.step(val_loss)
        start_time = time.time()
        
        np.savetxt(f'./results/Loss/finetune_img64_dist_dvae_train_loss_{x_train.shape[0]}data_ep{ep_num}_bch{args.batch_size}_temp{args.temperature}.txt', train_losses, fmt='%.8f')
        np.savetxt(f'./results/Loss/finetune_img64_dist_dvae_val_loss_{x_train.shape[0]}data_ep{ep_num}_bch{args.batch_size}_temp{args.temperature}.txt', val_losses, fmt='%.8f')

        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='val loss')
        plt.legend()
        plt.savefig(f'./results/Loss/finetune_img64_dist_dvae_loss_{x_train.shape[0]}data_ep{ep_num}_bch{args.batch_size}_temp{args.temperature}.png')
        plt.clf()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'./results/Model/Finetune/finetune_img64_dist_dvae_{x_train.shape[0]}data_ep{ep_num}_bch{args.batch_size}_temp{args.temperature}.pth')
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
