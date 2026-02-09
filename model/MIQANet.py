from traceback import print_tb
import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange
import numpy as np

import torchvision.models

def continuous2discrete(depth, d_min, d_max, n_c):
    depth = torch.round(torch.log(depth / d_min) / np.log(d_max / d_min) * (n_c - 1))
    return depth

def discrete2continuous(depth, d_min, d_max, n_c):
    depth = torch.exp(depth / (n_c - 1) * np.log(d_max / d_min) + np.log(d_min))
    return depth

class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class MIQANet(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

        layers = 18
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=True)

        self.conv12 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        self.bilinear = nn.Upsample(size=(28,28), mode='bilinear', align_corners=True)

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv22 = nn.Conv2d(num_channels,384,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(384)

        # Learnable Adaptive Parameter (AP) to balance CNN and Transformer branches
        # We keep it as a scalar and squash to (0,1) via sigmoid during forward.
        # Initialise to 0.8 (same as the previous fixed weight) in logit space.
        self.ap = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(0.8)))))
    
    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    # def soft_ordinal_regression(self, pred_prob, d_min, d_max, n_c):
    #     pred_prob_sum = torch.sum(pred_prob, 1, keepdim=True)
    #     Intergral = torch.floor(pred_prob_sum)
    #     Fraction = pred_prob_sum - Intergral
    #     depth_low = (discrete2continuous(Intergral, d_min, d_max, n_c) +
    #                     discrete2continuous(Intergral + 1, d_min, d_max, n_c)) / 2
    #     depth_high = (discrete2continuous(Intergral + 1, d_min, d_max, n_c) +
    #                     discrete2continuous(Intergral + 2, d_min, d_max, n_c)) / 2
    #     pred_depth = depth_low * (1 - Fraction) + depth_high * Fraction
    #     return pred_depth

    # def inference(self, y):
    #     if isinstance(y, list):
    #         y = y[-1]
    #     if isinstance(y, dict):
    #         y = y['y']
    #     # mode
    #     # OR = Ordinal Regression
    #     # CE = Cross Entropy
        
    #     inferenceFunc = self.soft_ordinal_regression


    #     pred_depth = inferenceFunc(y, 0.71, 10, 18)
    #     return pred_depth

    def forward(self, x):
        # ****************************
        x1 = self.conv12(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        # x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)        
        x1 = self.layer4(x1)
        x1 = self.conv22(x1)
        x1 = self.bn2(x1)
        # strategy one:
        # x1 = self.bilinear(x1)
        # strategy two:
        x1 = torch.cat([x1,x1], dim = 2)
        x1 = torch.cat([x1,x1], dim = 3)

        # print(x1.shape)         

        # ****************************  

        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        # print(x.shape) ([2, 768, 28, 28])
        # x = torch.cat([x,x1],dim=2)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        # print(x.shape,'need fusion') (2,384,28,28)
        # print(x[0,0,4:10,4:10],'x')
        # print(x1[0,0,4:10,4:10],'x1')
        # x = x + x1 * 0.8  # (deprecated) fixed fusion weight
        # x = torch.cat(x,x1)
        # Adaptive fusion between Transformer feature map (x) and CNN feature map (x1)
        alpha = torch.sigmoid(self.ap)
        x = alpha * x + (1.0 - alpha) * x1

        x = self.swintransformer2(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        # print(x.shape,'need fusion')
        score_list = []
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w,dim = 0) / torch.sum(w, dim=0)
            score_list.append(_s)
            # print(score.shape,'hahahaha')
        return torch.stack(score_list, dim=0)
