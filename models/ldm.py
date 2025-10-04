'''
coding: utf-8 
        Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
        torch 2.2.2+cu121
Created on 2024-08-16
@title: ldm.py
@description: define the LDM (Latent Diffusion Model) model
@author: <|Ronald B Liu|liu.ronald@icloud.com|>
@github: <https://github.com/RL-arch>
@version: 0.1.0
'''
import torch
import torch.nn.functional as F


class Resnet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.time = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.torch.nn.Linear(512, dim_out),
            torch.nn.Unflatten(dim=1, unflattened_size=(dim_out, 1, 1)),
        )

        self.s0 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_in,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_in,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
        )

        self.s1 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_out,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv2d(dim_out,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.torch.nn.Conv2d(dim_in,
                                             dim_out,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)

    def forward(self, x, time):
        #x -> [1, 128, 6, 6] - 128 channels, 6x6 spatial dim
        # time -> [1, 512] the proportion of noise in the feature map x

        res = x

        #[1, 512] -> [1, 256, 1, 1]
        time = self.time(time)

        #[1, 128, 6, 6] -> [1, 256, 12, 12]
        x = self.s0(x) + time

        # Ensure time embedding matches spatial dimensions
        if time.shape[-2:] != x.shape[-2:]:
            time = time.expand(-1, -1, x.shape[-2], x.shape[-1])
        
        #[1, 256, 12, 12]
        x = self.s1(x)

        #[1, 256, 12, 12] -> [1, 128, 6, 6]
        if self.res:
            res = self.res(res)

        
        #[1, 256, 12, 12]
        x = res + x

        return x

# Resnet(128, 256)(torch.randn(1, 128, 6, 6), torch.randn(1, 512)).shape # torch.Size([1, 256, 6, 6])

# num_head = 4
class CrossAttention(torch.nn.Module):

    def __init__(self, dim_q, dim_kv):
        #dim_q -> 128 image data
        #dim_kv -> 256 text data

        super().__init__()

        self.dim_q = dim_q

        self.q = torch.nn.Linear(dim_q, dim_q, bias=False)
        self.k = torch.nn.Linear(dim_kv, dim_q, bias=False)
        self.v = torch.nn.Linear(dim_kv, dim_q, bias=False)

        self.out = torch.nn.Linear(dim_q, dim_q)

    def forward(self, q, kv):
        #x -> [1, 36, 128]
        #kv -> [1, 65, 256]

        #[1, 36, 128] -> [1, 36, 128]
        q = self.q(q)
        #[1, 65, 256] -> [1, 65, 128]
        k = self.k(kv)
        #[1, 65, 256] -> [1, 65, 128]
        v = self.v(kv)

        def reshape(x):
            #x -> [1, 36, 128] （36 = 6*6）
            b, lens, dim = x.shape

            #[1, 36, 128] -> [1, 36, 6, 64]， 64 = 128/4
            x = x.reshape(b, lens, 4, dim // 4)

            #[1, 36, 8, 16] -> [1, 8, 36, 16]
            x = x.transpose(1, 2)

            #[1, 8, 36, 16] -> [8, 36, 16]
            x = x.reshape(b * 4, lens, dim // 4)

            return x
        '''Reshape and split for multi-head attention'''
        #[1, 36, 128] -> [8, 36, 16]
        q = reshape(q)
        #[1, 65, 128] -> [8, 65, 16]
        k = reshape(k)
        #[1, 65, 128] -> [8, 65, 16]
        v = reshape(v)

        #[8, 36, 16] * [8, 16, 65] -> [8, 36, 65]
        #atten = q.bmm(k.transpose(1, 2)) * (self.dim_q // 8)**-0.5

        # Mathematically equivalent, but may produce small numerical errors in practice
        atten = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device),
            q,
            k.transpose(1, 2),
            beta=0,
            alpha=(self.dim_q // 4)**-0.5,
        )

        atten = atten.softmax(dim=-1)

        #[8, 36, 65] * [8, 65, 40] -> [8, 36, 40]
        atten = atten.bmm(v)

        def reshape(x):
            #x -> [8, 36, 40]
            b, lens, dim = x.shape

            #[8, 36, 40] -> [1, 8, 36, 40]
            x = x.reshape(b // 4, 4, lens, dim)

            #[1, 8, 36, 40] -> [1, 36, 8, 40]
            x = x.transpose(1, 2)

            #[1, 36, 128]
            x = x.reshape(b // 4, lens, dim * 4)

            return x

        
        #[8, 36, 40] -> [1, 36, 128]
        # combine multi-heads into one
        atten = reshape(atten)

        #[1, 36, 128] -> [1, 36, 128]
        # linear output
        atten = self.out(atten)

        return atten

# CrossAttention(128, 256)(
#                             torch.randn(1, 36, 128), 
#                             torch.randn(1, 65, 256)
#                         ).shape # torch.Size([1, 36, 128])

class Transformer(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        #in
        self.norm_in = torch.nn.GroupNorm(num_groups=32,
                                          num_channels=dim,
                                          eps=1e-6,
                                          affine=True)
        self.cnn_in = torch.nn.Conv2d(dim,
                                      dim,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

        #atten
        self.norm_atten0 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten1 = CrossAttention(dim, dim)
        self.norm_atten1 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten2 = CrossAttention(dim, 256)

        #act
        self.norm_act = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.fc0 = torch.nn.Linear(dim, dim * 8)
        self.act = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(dim * 4, dim)

        #out
        self.cnn_out = torch.nn.Conv2d(dim,
                                       dim,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, q, kv):
        # Input data and image data simultaneously into transformer layer
        #q -> [1, 128, 6, 6]
        #kv -> [1, 65, 256]
        b, _, h, w = q.shape
        res1 = q

        #----in----
        #keep the same dimension
        #[1, 128, 6, 6]
        q = self.cnn_in(self.norm_in(q))

        #[1, 128, 6, 6] -> [1, 6, 6, 128] -> [1, 36, 128]
        q = q.permute(0, 2, 3, 1).reshape(b, h * w, self.dim)

        #----atten----
        #keep the same dimension
        #[1, 36, 128]
        q = self.atten1(q=self.norm_atten0(q), kv=self.norm_atten0(q)) + q
        q = self.atten2(q=self.norm_atten1(q), kv=kv) + q

        #----act----
        #[1, 36, 128]
        res2 = q

        #[1, 36, 128] -> [1, 36, 1024] ([2]x8)
        q = self.fc0(self.norm_act(q))

        #512
        d = q.shape[2] // 2

        #[1, 36, 512] * [1, 36, 512] -> [1, 36, 512]
        q = q[:, :, :d] * self.act(q[:, :, d:])

        #[1, 36, 512] -> [1, 36, 128]
        q = self.fc1(q) + res2

        #----out----
        #[1, 36, 128] -> [1, 6, 6, 128] -> [1, 128, 6, 6]
        q = q.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()

        # keep the same dimension
        #[1, 128, 6, 6]
        q = self.cnn_out(q) + res1

        return q

# Transformer(128)(torch.randn(1, 128, 6, 6), torch.randn(1, 65, 256)).shape # torch.Size([1, 128, 6, 6])

# Unet first half
class DownBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.tf0 = Transformer(dim_out)
        self.res0 = Resnet(dim_in, dim_out)

        self.tf1 = Transformer(dim_out)
        self.res1 = Resnet(dim_out, dim_out)

        # downsample
        self.out = torch.nn.Conv2d(dim_out,
                                   dim_out,
                                   kernel_size=3,
                                   stride=1,# stride=2
                                   padding=1)

    def forward(self, out_vae, out_encoder, time):
        outs = [] # save for later use in Up

        out_vae = self.res0(out_vae, time) # add noise
        out_vae = self.tf0(out_vae, out_encoder) # add text information
        outs.append(out_vae)

        out_vae = self.res1(out_vae, time) # add noise
        out_vae = self.tf1(out_vae, out_encoder) # add text information
        outs.append(out_vae)
        # print('before downsapled:', out_vae.shape) # ([1, 256, 6, 6])
        out_vae = self.out(out_vae) # downsample
        # print('downsapled:', out_vae.shape) #([1, 256, 6, 6])
        outs.append(out_vae)

        return out_vae, outs

# # The encoder output in VAE is not downsampled, it remains (1, 128, 6, 6)
# DownBlock(128, 256)(torch.randn(1, 128, 6, 6), torch.randn(1, 65, 256),
#                     torch.randn(1, 512))[0].shape #torch.Size([1, 256, 3, 3])

class UpBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out, dim_prev, add_up):
        super().__init__()

        self.res0 = Resnet(dim_out + dim_prev, dim_out)
        self.res1 = Resnet(dim_out + dim_out, dim_out)
        self.res2 = Resnet(dim_in + dim_out, dim_out)

        self.tf0 = Transformer(dim_out)
        self.tf1 = Transformer(dim_out)
        self.tf2 = Transformer(dim_out)

        self.out = None
        if add_up:
            self.out = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=1, mode='nearest'), # scale_factor=2
                torch.nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
            )

    def forward(self, out_vae, out_encoder, time, out_down):
        out_vae = self.res0(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf0(out_vae, out_encoder)

        out_vae = self.res1(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf1(out_vae, out_encoder)

        out_vae = self.res2(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf2(out_vae, out_encoder)

        if self.out:
            out_vae = self.out(out_vae)

        return out_vae

# UpBlock(128, 256, 512, True)(torch.randn(1, 512, 6, 6),
#                               torch.randn(1, 65, 256), torch.randn(1, 512), [
#                                   torch.randn(1, 128, 6, 6),
#                                   torch.randn(1, 256, 6, 6),
#                                   torch.randn(1, 256, 6, 6)
#                               ]).shape # torch.Size([1, 256, 6, 6])


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #in
        self.in_vae = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1) # 320 -> 128, 4->128

        self.in_time = torch.nn.Sequential(
            torch.nn.Linear(128, 512), # *4
            torch.nn.SiLU(),
            torch.nn.Linear(512, 512),
        )

        #down
        self.down_block0 = DownBlock(128, 128)
        self.down_block1 = DownBlock(128, 256)
        self.down_block2 = DownBlock(256, 512)

        self.down_res0 = Resnet(512, 512)
        self.down_res1 = Resnet(512, 512)

        #mid
        self.mid_res0 = Resnet(512, 512)
        self.mid_tf = Transformer(512)
        self.mid_res1 = Resnet(512, 512)

        #up
        self.up_res0 = Resnet(1024, 512)
        self.up_res1 = Resnet(1024, 512)
        self.up_res2 = Resnet(1024, 512)

        self.up_in = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
        )

        self.up_block0 = UpBlock(256, 512, 512, True)
        self.up_block1 = UpBlock(128, 256, 512, True)
        self.up_block2 = UpBlock(128, 128, 256, False)

        #out
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-5),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        
    def get_time_embed(self, t):
        # Ensure t is a tensor of shape (batch_size,)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # Calculate embedding
        e = torch.arange(64, dtype=torch.float32, device=t.device) * -9.210340371976184 / 64
        e = e.exp().to(t.device) * t

        # Concatenate cosine and sine embeddings
        e = torch.cat([e.cos(), e.sin()], dim=1)

        # Return the embedding
        return e

    def forward(self, out_vae, out_encoder, time):
        #out_vae -> [1, 128, 6, 6]
        #out_encoder -> [1, 65, 256]
        #time -> [1]

        #----in----
        #[1, 128, 6, 6] -> [1, 256, 6, 6]
        out_vae = self.in_vae(out_vae)
        # print('1', out_vae.shape) # torch.Size([1, 128, 6, 6])

        #[1] -> [1, 128]
        time = self.get_time_embed(time)
        #[1, 128] -> [1, 512]
        time = self.in_time(time)

        #----down----
        out_down = [out_vae]

        #[1, 128, 6, 6],[1, 65, 256],[1, 512] -> [1, 128, 3, 3]
        #out -> [1, 128, 6, 6],[1, 128, 6, 6][1, 128, 3, 3]
        out_vae, out = self.down_block0(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        # print('2', out_vae.shape)
        out_down.extend(out)

        #[1, 128, 3, 3],[1, 65, 256],[1, 512] -> [1, 256, 2, 2]
        #out -> [1, 256, 3, 3],[1, 256, 3, 3],[1, 256, 2, 2]
        out_vae, out = self.down_block1(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        # print('3', out_vae.shape)
        out_down.extend(out)

        #[1, 256, 2, 2],[1, 65, 256],[1, 512] -> [1, 512, 1, 1]
        #out -> [1, 512, 2, 2],[1, 512, 2, 2],[1, 512, 1, 1]
        out_vae, out = self.down_block2(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        # print('4', out_vae.shape)
        out_down.extend(out)

        #[1, 512, 1, 1],[1, 512] -> [1, 512, 1, 1]
        out_vae = self.down_res0(out_vae, time)
        # print('5', out_vae.shape)
        out_down.append(out_vae)

        #[1, 512, 1, 1],[1, 512] -> [1, 512, 1, 1]
        out_vae = self.down_res1(out_vae, time)
        # print('6', out_vae.shape)
        out_down.append(out_vae)

        #----mid----
        #[1, 512, 1, 1],[1, 512] -> [1, 512, 1, 1]
        out_vae = self.mid_res0(out_vae, time)
        # print('7', out_vae.shape)

        #[1, 512, 1, 1],[1, 65, 256] -> [1, 512, 1, 1]
        out_vae = self.mid_tf(out_vae, out_encoder)
        # print('8', out_vae.shape)

        #[1, 512, 1, 1],[1, 512] -> [1, 512, 1, 1]
        out_vae = self.mid_res1(out_vae, time)
        # print('9', out_vae.shape)

        #----up----
        #[1, 512+512, 1, 1],[1, 512] -> [1, 512, 1, 1]
        out_vae = self.up_res0(torch.cat([out_vae, out_down.pop()], dim=1),
                               time)
        # print('10', out_vae.shape)

        #[1, 512+512, 1, 1],[1, 512] -> [1, 512, 1, 1]
        out_vae = self.up_res1(torch.cat([out_vae, out_down.pop()], dim=1),
                               time)
        # print('11', out_vae.shape)

        #[1, 512+512, 1, 1],[1, 512] -> [1, 512, 1, 1]
        out_vae = self.up_res2(torch.cat([out_vae, out_down.pop()], dim=1),
                               time)
        # print('12', out_vae.shape)

        #[1, 512, 1, 1] -> [1, 512, 2, 2]
        out_vae = self.up_in(out_vae)
        # print('13', out_vae.shape)

        #[1, 512, 2, 2],[1, 65, 256],[1, 512] -> [1, 512, 3, 3]
        #out_down -> [1, 256, 2, 2],[1, 512, 2, 2],[1, 512, 2, 2]
        out_vae = self.up_block0(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)
        # print('14', out_vae.shape)

        #[1, 512, 3, 3],[1, 65, 256],[1, 512] -> [1, 256, 6, 6]
        #out_down -> [1, 128, 3, 3],[1, 256, 3, 3],[1, 256, 3, 3]
        out_vae = self.up_block1(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)
        # print('15', out_vae.shape)

        #[1, 256, 6, 6],[1, 65, 256],[1, 512] -> [1, 128, 6, 6]
        #out_down -> [1, 128, 6, 6],[1, 128, 6, 6],[1, 128, 6, 6]
        out_vae = self.up_block2(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)
        # print('16', out_vae.shape)

        #----out----
        #[1, 128, 6, 6] -> [1, 128, 6, 6]
        out_vae = self.out(out_vae)
        # print('17', out_vae.shape) 

        return out_vae

if __name__ == '__main__':
    UNet()(torch.randn(1, 128, 6, 6), torch.randn(1, 65, 256),
           torch.LongTensor([26])).shape

### debug output ###
# 1 torch.Size([1, 128, 6, 6])
# 2 torch.Size([1, 128, 6, 6])
# 3 torch.Size([1, 256, 6, 6])
# 4 torch.Size([1, 512, 6, 6])
# 5 torch.Size([1, 512, 6, 6])
# 6 torch.Size([1, 512, 6, 6])
# 7 torch.Size([1, 512, 6, 6])
# 8 torch.Size([1, 512, 6, 6])
# 9 torch.Size([1, 512, 6, 6])
# 10 torch.Size([1, 512, 6, 6])
# 11 torch.Size([1, 512, 6, 6])
# 12 torch.Size([1, 512, 6, 6])
# 13 torch.Size([1, 512, 6, 6])
# 14 torch.Size([1, 512, 6, 6])
# 15 torch.Size([1, 256, 6, 6])
# 16 torch.Size([1, 128, 6, 6])
# 17 torch.Size([1, 128, 6, 6])
