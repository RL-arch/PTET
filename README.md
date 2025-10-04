# PTET: Pre-trained Transformer for EIT-based Tactile Reconstruction

## Description
Implementing for the paper "**Efficient Tactile Perception with Soft Electrical Impedance Tomography and Pre-trained Transformer**", ([Preprint](https://arxiv.org/pdf/2506.02824)), accepted by _IEEE Transactions on Industrial Electronics (TIE)_ in 2025. 



## Overview

![Architecture](./figs/Architecture-1.png)

<!-- You can also view the detailed architecture in [this PDF](./figs/Architecture.pdf). -->


<!-- ## Installation

To install the project, follow these steps:

1. Clone the repository.
2. Run `pip install -r requirements.txt` to install the dependencies.
3. ... -->


## 1. Data preparation

1. Download the raw data from Edinburgh Data Share [DOI:10.7488/ds/7985](https://doi.org/10.7488/ds/7985)
2. Put datasets in the folder [./data/](./data/) in order:

        ./data
            |
            |__GT
            |  |
            |  |__GT0.npy
            |  |__GT1.npy
            |  |__ ...
            |
            |__V
            |  |
            |  |__V_EIM0_64.npy
            |  |__V_EIM1_64.npy
            |  |__ ...

<!-- 2. For the raw data in .mat, please inquire @Huazhi -->
<!-- 3. For the pre-processed data in .npy, check [./data/](./data/) -->
3. ***Voltage (V)*** are saved in 64*64 [EIM (Electrical Impedance Map)](https://ieeexplore.ieee.org/abstract/document/10962299), to preprocess the _.mat_ data use [v_data_mat2npy.ipynb](./tools/0_data_preprocessing/v_data_mat2npy.ipynb)
4. ***Ground truth (GT)*** are saved in 48*48, to preprocess the _.mat_ data use [gt_data_mat2npy.ipynb](./tools/0_data_preprocessing/gt_data_mat2npy.ipynb)

## 2. Pretraining

### 2.1 Pretrain the V (volatge signal data)
- run [pretrain_v.py](./pretrain_v.py) (check **Note** for details)
- check saved [results](./results/)

#### Note
- model is defined in [mae.py](./models/mae.py)
- The **V** signal is 64*64 image format
- The MAE ViT model is defined in [mae.py](./models/mae.py)
- to adjust the model parameters use the hyp params defined in [pretrain_v.py](./pretrain_v.py) (e.g., add embbedding_dim to use large ViT model)
- you can also adjust more params in [mae.py](./models/mae.py) (e.g., head numbers etc). The current params are optimized for 64*64 input, based on ViT-Tiny settings
- you can adjust the patch_size and GPU numbers in [pretrain_v.py](./pretrain_v.py) to accelerate the training process
- no overfitting observed, as the model is large, so no need to use early stopping
- to debug the model, set the random input in comments and run `python mae.py`; you can see the output shape from the encoder, which will be the input for finetuning



### 2.2 Pretrain the GT 
- run [pretrain_gt.py](./pretrain_gt.py) (check **Note** for details)
- check saved [results](./results/)

#### Note
- model is defined in [vae.py](./models/vae.py)
- the params are designed for 48*48 specific input
- encoded shape: torch.Size([1, 128, 6, 6])
- VAE is saved every 10 epochs (see code in [pretrain_gt.py](./pretrain_gt.py)
- similar to above...

## 3. Finetune

We provide several finetuning methods:


### 3.1 Use MLP Linear projection (this paper, standard finetune approach in [MAE](https://github.com/facebookresearch/mae/blob/main/main_finetune.py))
- run [finetune_mlp.py](./finetune_mlp.py) 
- check saved [results](./results/)

Optional:
Use [finetune_mlp_S.py](./finetune_mlp_S.py) to finetune with Sensitivity Matrix (S) ([Sensitivity16_48.txt](Sensitivity16_48.txt)) as physical inductive bias.

#### Note
- recommend to define the model with the same **mask_ratio** as the pretraining model (e.g., 0.75)
- the reason above see [mae_test](./tools/1_result_check_pretrained/mae_test_img64.ipynb), if you change the mask ratio, the reconstruction will be bad
- MLP projection layer, input dimension is 16640 ((256+1) \* 256 \* (1-0.75)=16640), see [vit](./figs/vit.gif)
- if you set mask_ratio=0, the input is 65792 = 257 \* 256
- output dimension is for [1, 128, 6, 6]
- adjsut the hyperparam `num_ep` to select the best epoch of pretrained VAE model (default:250)
- the optimized code is direct projection, you may try to add MLP layers (see `class Linear()` with  `hidden_dim` in comments in [finetune_mlp.py](./finetune_mlp.py))

### 3.2 Use [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) 

- model is defined in [ldm.py](./models/ldm.py)
- run [finetune_ldm.py](./finetune_ldm.py) 
- check saved [results](./results/)

### 3.3 Knowledge Distillation
- run [finetune_dist.py](./finetune_dist.py)
- check saved [results](./results/)


## 4. Evaluation
Use notebooks in [tools](./tools/)

You may need to move the notebook to the root directory of this repo, and revise the paths in the scripts, then run it.

<!-- ## Contributing

Contributions are welcome! Please follow the guidelines outlined in CONTRIBUTING.md. -->

## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE
                       Version 3](LICENSE).
