import torch
import torch.nn as nn
import timm
import numpy as np
from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange


class FrameEncoder(nn.Module):
    def __init__(self, input_dim, block_num):
        super(FrameEncoder, self).__init__()
        self.block_num = block_num
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1)

        # encoder block
        self.enc_pi1 = nn.Conv1d(input_dim, input_dim // 2, kernel_size=1, stride=1)
        self.enc_pi2 = nn.Conv1d(input_dim // 2, input_dim, kernel_size=1, stride=1)
        self.enc_tao = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1)

        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1)
        self.conv_z_attention = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.conv1(x)

        for i in range(self.block_num):
            pi = self.relu(h)
            pi = self.enc_pi1(pi)
            pi = self.relu(pi)
            pi = self.enc_pi2(pi)

            tau = self.enc_tao(h)
            tau = self.sigmoid(tau)
            h = h * (1 - tau) + pi * tau

        z = self.conv2(h)

        z_attention = self.conv_z_attention(h)
        z_attention = self.sigmoid(z_attention)

        z = torch.multiply(z, z_attention)

        return z


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


# 创建保存hook内容的对象
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class MANAQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        # 创建ViT网络
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        # 为卷积层注册hook
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        #
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

        # frame encoder
        self.enc = FrameEncoder(input_dim=50, block_num=3).cuda()

        self.mlp1 = nn.Linear(66, 224)
        self.mlp2 = nn.Linear(50, 224)
        self.relu = nn.ReLU()

        # self.conv_0 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 39), stride=1, padding=1).cuda()
        # self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=4).cuda()

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        # print('manaqa:',np.shape(x))
        b, l, j, _ = x.shape
        x = x.view(b, l, -1)
        # x = self.enc(x)
        x = self.mlp1(x)
        x = self.relu(self.mlp2(x.permute(0, 2, 1)))
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)

        """        
        x1 = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        # x = self.enc(x1)

        x2 = x.unsqueeze(0)
        x3 = torch.cat((x2, x2, x2), dim=0)
        x4 = x3.transpose(1, 0)

        x = self.conv_0(x4)
        for i in range((224 - 52) // (4 * 2 + 1 - 5)):
            x = self.conv_1(x)
        # print('conv:',np.shape(x))
        """

        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score
