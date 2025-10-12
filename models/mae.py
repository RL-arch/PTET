'''
coding: utf-8 
        Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
        torch 2.2.2+cu121
Created on 2024-08-16
@title: mae.py
@description: define the MAE (Masked Autoencoder) model
@author: <Ronald B Liu>
@github: <https://github.com/RL-arch>
@version: 0.1.0
'''
import torch
import torch.nn as nn
import timm
import numpy as np
import scipy.io
import time

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

def mat2tensor64(V_mat):
    print("Converting mat to tensor...")
    start_time = time.time()
    new_matrices = []

    # Process each (16, 16) matrix
    for matrix_index in range(V_mat.shape[0]):
        # Initialize the new (64, 64) matrix
        V0_mat_64x64 = np.zeros((64, 64), dtype=np.float16)
        
        # Populate the new matrix
        for i in range(16):
            for j in range(16):
                value = np.float16(V_mat[matrix_index, i, j])  # Convert the value to float16
                V0_mat_64x64[i*4:(i+1)*4, j*4:(j+1)*4] = np.full((4, 4), value, dtype=np.float16)
        
        # Add the new matrix to the list
        new_matrices.append(V0_mat_64x64)

    # Convert the list to a numpy array
    V_new = np.array(new_matrices)

    # print(f"New matrices dtype: {V_new.dtype}")  # Should be float16
    # print(f"New matrices shape: {V_new.shape}")  # Should be (11520, 64, 64)

    # Save the new matrices to a .npy file
    # np.save('Data/V/V_EIM0_64x64.npy', V_new)
    V_tensor = torch.tensor(V_new).unsqueeze(1).float()
    # print(f"New tensor shape: {V_tensor.shape}")  # Should be (11520, 1, 64, 64)
    # Move the tensor to the GPU if available
    if torch.cuda.is_available():
        V_tensor = V_tensor.to('cuda')
        
    end_time = time.time() - start_time
    print(f"Tensor loaded. time is {end_time}")
    
    return V_tensor

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes
    
class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=64,
                 patch_size=4,
                 emb_dim=256,
                 num_layer=12,
                 num_head=4,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(1, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        # print("- [Enc] Size of patches:", patches.size())  # ([89, 256, 16, 16])
        # print("- [Enc] Size of pos_embedding (zeros):", self.pos_embedding.size())  # ([256, 1, 256]) 256 = （64 / 4）** 2, 1, 256
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        # print("- [Enc] Size of rearranged patches:", patches.size())  # ([256, 89, 256])
        patches = patches + self.pos_embedding
        # print("- [Enc] Size of patches after adding pos_embedding:", patches.size())  # ([256, 89, 256])
        patches, forward_indexes, backward_indexes = self.shuffle(patches) # masking
        # print("- [Enc] Size of patches after shuffle:", patches.size()) # ([64, 89, 256])

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        # print("- [Enc] Size of patches after adding cls token:", patches.size())  # ([65, 89, 256]), 65 = 256 * (1 - 0.75) + 1 (cls token)
        patches = rearrange(patches, 't b c -> b t c')
        # print("- [Enc] Size of patches after rearrange:", patches.size())  # ([89, 65, 256])
        features = self.layer_norm(self.transformer(patches))
        # print("- [Enc] Size of features after transformer:", features.size())  # ([89, 65, 256])
        features = rearrange(features, 'b t c -> t b c')
        # print("- [Enc] Size of features after rearrange:", features.size())  # ([65, 89, 256])

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=64,
                 patch_size=4,
                 emb_dim=256,
                 num_layer=2,
                 num_head=4,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 1 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        # print(" [Dec] T:", T)  # 65
        backward_indexes = torch.cat(
            [
                torch.zeros(
                    1,
                    backward_indexes.shape[1]).to(backward_indexes),
                    backward_indexes + 1
            ], dim=0)
        # print(" - [Dec] Size of backward_indexes:", backward_indexes.size())  # ([257, 89]) 257 = 256 + 1 (cls token)
        features = torch.cat([
                                features, 
                                self.mask_token.expand(
                                    backward_indexes.shape[0] - features.shape[0],
                                    features.shape[1],
                                    -1
                                    )
                            ], dim=0)
        # print(" - [Dec] Size of cat features:", features.size()) # ([257, 89, 256])
        features = take_indexes(features, backward_indexes)
        # print(" - [Dec] Features size, take_ids:", features.size()) # ([257, 89, 256])
        features = features + self.pos_embedding
        # print(" - [Dec] Features size, + pos_emb:", features.size()) # ([257, 89, 256])

        features = rearrange(features, 't b c -> b t c')
        # print(" - [Dec] Features size, rearrange, input transformer:", features.size()) #([89, 257, 256])
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature (class token)
        # print(" - [Dec] Features size, remove cls token:", features.size()) # ([256, 89, 256]) blocks, batch, emb_dim
        
        patches = self.head(features)
        # print(" - [Dec] patches size: ", patches.size()) # ([256, 89, 16])
        mask = torch.zeros_like(patches)
        # print(" - [Dec] mask size: ", mask.size()) # ([256, 89, 16])
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        # print(" - [Dec] mask size, take_indexes:", mask.size()) # ([256, 89, 16])
        img = self.patch2img(patches)
        # print(" - [Dec] img size, patch2img:", img.size()) # ([89, 1, 64, 64])
        mask = self.patch2img(mask)

        return img, mask

'''
for img_size 64:
all params set as
ViT_large / 4
'''
class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=64,
                 patch_size=4,
                 emb_dim=256,
                 encoder_layer=12,
                 encoder_head=4,
                 decoder_layer=2,
                 decoder_head=4,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


################################
class CAE_Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(CAE_Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 12*12*5)
        self.reshape = nn.Sequential(
            nn.ELU(),
            nn.Unflatten(1, (5, 12, 12))
        )
        self.conv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 尺寸翻倍
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 尺寸翻倍
            nn.Conv2d(5, 1, kernel_size=3, padding=1),
            nn.ELU()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        return self.conv_layers(x)

################################

if __name__ == '__main__':
    # shuffle = PatchShuffle(0.75)
    # a = torch.rand(16, 2, 10)
    # b, forward_indexes, backward_indexes = shuffle(a)
    # print(b.shape)
    # print(forward_indexes.shape)

    img = torch.rand(89, 1, 64, 64)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    ####
    print('feature main', features.shape) #torch.Size([65, 89, 256])
    # no_cls_features = features[1:] 
    new_features = rearrange(features, 't b c -> b t c')
    print('new feature main', new_features.shape) #torch.Size([89, 65, 256])
    # [89, 64, 256] or [89, 65, 256] aim input to finetune 
    # [89, 256, 256] when mask_ratio = 0
    ####
    predicted_img, mask = decoder(features, backward_indexes)
    print('predict', predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)
