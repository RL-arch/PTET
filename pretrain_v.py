"""
coding: utf-8 
        Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
        torch 2.2.2+cu121
Created on 2024-08-16
@title: finetune_ldm.py
@description: finetune the LDM (Latent Diffusion Model) model
              MAE img: random mask blocks on image (square matrix)
@author: <|Ronald B Liu|liu.ronald@icloud.com|>
@github: <https://github.com/RL-arch>
@version: 0.1.0
"""
import os
import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
import scipy
import os
import matplotlib.pyplot as plt
from models.mae import *
from torch.utils.data import DataLoader, TensorDataset
import time

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ############ for MAE_ViT params ##############################
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=64)
    ############## training hypparams ##############################
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_device_batch_size', type=int, default=49152)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='mae-pretrain')
    parser.add_argument('--gpu', type=str, default='0', help='Comma separated list of GPU ids to use (default: 0)')

    args = parser.parse_args()
    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # set random seed for reproducibility
    set_random_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    '''for .mat format'''
    # V0_mat = scipy.io.loadmat('./data/V/V_EIM0.mat')['V_EIM0']
    # V1_mat = scipy.io.loadmat('./data/V/V_EIM1.mat')['V_EIM1']
    # V2_mat = scipy.io.loadmat('./data/V/V_EIM2.mat')['V_EIM2']
    # V3_mat = scipy.io.loadmat('./data/V/V_EIM3.mat')['V_EIM3']
    # V4_mat = scipy.io.loadmat('./data/V/V_EIM4.mat')['V_EIM4']
    
    # V0_tensor = mat2tensor64(V0_mat)
    # V1_tensor = mat2tensor64(V1_mat)
    # V2_tensor = mat2tensor64(V2_mat)
    # V3_tensor = mat2tensor64(V3_mat)
    # V4_tensor = mat2tensor64(V4_mat)
    
    '''for numpy format'''
    V0_tensor = torch.tensor(np.load('./data/V/V_EIM0_64.npy'))
    V1_tensor = torch.tensor(np.load('./data/V/V_EIM1_64.npy'))
    V2_tensor = torch.tensor(np.load('./data/V/V_EIM2_64.npy'))
    V3_tensor = torch.tensor(np.load('./data/V/V_EIM3_64.npy'))
    V4_tensor = torch.tensor(np.load('./data/V/V_EIM4_64.npy'))
    V5_tensor = torch.tensor(np.load('./data/V/V_EIM5_64.npy'))

    x_train = torch.cat((
                        # V0_tensor, 
                        V1_tensor[0:-1500] 
                        , V2_tensor[0:-1500] 
                        , V3_tensor[0:-1500] 
                        , V4_tensor[0:-1500]
                        , V5_tensor[0:-1500]
                        ), dim=0)
    x_val = torch.cat((
                        V1_tensor[-1500:-500]
                        , V2_tensor[-1500:-500]
                        , V3_tensor[-1500:-500]
                        , V4_tensor[-1500:-500]
                        , V5_tensor[-1500:-500]
                        ), dim=0)
    x_test = torch.cat((
                        V1_tensor[-500:]
                        , V2_tensor[-500:]
                        , V3_tensor[-500:]
                        , V4_tensor[-500:]
                        , V5_tensor[-500:]), 
                        dim=0)
    
    
    # random shuffle the data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    
    x_train = x_train[indices]
    print(x_train.shape)
    data_num = x_train.shape[0]
    
    train_loader = DataLoader(TensorDataset(x_train), load_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TensorDataset(x_val), load_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    '''
    patch_size = 4
    mask_ratio = 0.75
    embedding_dim = 256
    img_size = 64
    '''
    # define a model 
    model = MAE_ViT(
                        image_size=args.img_size, 
                        patch_size=args.patch_size,
                        emb_dim=args.embedding_dim,
                        mask_ratio=args.mask_ratio, 
                    )
    
    # Enable multi-GPU support
    if torch.cuda.device_count() > 1:
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"  
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Using {torch.cuda.device_count()} GPUs, ids {args.gpu}")
        model = torch.nn.DataParallel(model)

    # load model to device 
    model = model.to(device)
    
    # customized learning rate setting
    optim = torch.optim.AdamW(model.parameters(), 
                              lr=args.base_learning_rate * args.batch_size / 256, 
                              betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.total_epoch, eta_min=1e-6)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10, verbose=True)

    step_count = 0
    optim.zero_grad()
    best_loss = float("inf")
    train_losses = []
    val_losses = []
    
    early_stopping = False
    patience = 200
    num_epochs_no_improvement = 0
    best_val_loss = float("inf")

    for e in range(args.total_epoch):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for img in tqdm(iter(train_loader)):
            step_count += 1
            # img = img.to(device)
            img = img[0].to(device, non_blocking=True)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            train_loss += loss.item()
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()
        # calculate the average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        ''' '''
        model.eval()
        correct_ones = 0
        total_ones = 0
        with torch.no_grad():
            val_loss = 0.0
            for val_img in val_loader:
                # val_img = val_img.to(device)
                val_img = val_img[0].to(device, non_blocking=True)
                predicted_val_img, mask = model(val_img)
                loss = torch.mean((predicted_val_img - val_img) ** 2 * mask) / args.mask_ratio
                val_loss += loss.item()
##                For tensorboard      visualize the first 16 predicted images on val dataset          
#                 predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
#                 img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
#                 img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
#                 writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
            # calculate the average validation loss
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            epoch_time = time.time() - start_time

            print(  f'Epoch [{e+1}/{args.total_epoch}], '
                    f'Training Loss: {train_loss:.8f}, '
                    f'Validation Loss: {val_loss:.8f}, '
                    f'Current learning rate: {current_lr}, '
                    f'Time: {epoch_time:.2f} s')
            
            ''' save model and loss'''
            # if the model is a DataParallel object, then we need to get the model object out of the DataParallel object,
            # otherwise the model is the model itself
            if isinstance(model, torch.nn.DataParallel):
                model_instance = model.module
            else:
                model_instance = model

            # save loss
            train_loss_file = f'./results/Loss/MAE_img{args.img_size}_train_loss_mr{args.mask_ratio}_eb{args.embedding_dim}_data{data_num}.txt'
            val_loss_file = f'./results/Loss/MAE_img{args.img_size}_val_loss_mr{args.mask_ratio}_eb{args.embedding_dim}_data{data_num}_{best_val_loss}.txt'
            np.savetxt(train_loss_file, train_losses, fmt='%.8f')
            np.savetxt(val_loss_file, val_losses, fmt='%.8f')

            plt.clf()
            plt.plot(train_losses, label='train loss')
            plt.plot(val_losses, label='val loss')
            plt.legend()
            plt.savefig(f'./results/Loss/MAE_img{args.img_size}_loss_mr{args.mask_ratio}_eb{args.embedding_dim}_data{data_num}.png')
            
            # save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_epochs_no_improvement = 0
                # save the encoder
                file_name = f'./results/Model/V_pretrain/MAE_img{args.img_size}_encoder_mr{args.mask_ratio}_eb{args.embedding_dim}_data{data_num}.pth'
                torch.save(model_instance.encoder.state_dict(), file_name)
                # save the decoder
                file_name = f'./results/Model/V_pretrain/MAE_img{args.img_size}_decoder_mr{args.mask_ratio}_eb{args.embedding_dim}_data{data_num}.pth'
                torch.save(model_instance.decoder.state_dict(), file_name)
                # save the whole model
                file_name = f'./results/Model/V_pretrain/MAE_img{args.img_size}_mr{args.mask_ratio}_eb{args.embedding_dim}_data{data_num}.pth'
                torch.save(model_instance.state_dict(), file_name)
                print(f'-------> Model saved, with loss {best_val_loss} <------------------------------------------------------')
            else:
                num_epochs_no_improvement += 1
                print(f'No improvement for {num_epochs_no_improvement} epochs.')
                if num_epochs_no_improvement == patience:
                    early_stopping = True
                    print("Early stopping!")
                    break

    if not early_stopping:
        print("Training completed without early stopping.")