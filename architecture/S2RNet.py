import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import numpy as np
import numbers
import matplotlib.pyplot as plt
# from .modules import ConvBlock
# from .Unet import Unet
from timm.models.layers import DropPath, to_2tuple


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type(Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    

class ConvNeXt_Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm_ConvNext(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# Spectral SE block
class SSE_Block(nn.Module):
    """
    从输入特征 x 生成通道重要性权重 w
    x: [B, C, H, W]
    w: [B, C] in (0, 1)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)   # [B, C]
        w = self.mlp(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        return w


class SAM_Gated(nn.Module):
    def __init__(self, dim, heads, num_blocks, attention_type):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LayerNorm(dim),
                Spatial_Spectral_Atten_wt_Gate(dim=dim, heads=heads, attention_type=attention_type),
                LayerNorm(dim),
                PreNorm(dim, mult=4)
            ]))
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (norm1, atten, norm2, ffn) in self.blocks:
            x = atten(norm1(x)) + x
            x = ffn(norm2(x)) + x
        return x


class Spatial_Spectral_Atten_wt_Gate(nn.Module):
    def __init__(self, dim, heads, attention_type='base', reduction=16):
        super().__init__()
        self.num_heads = heads
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.spa_fre_embed = nn.Conv2d(dim*2, dim, kernel_size=1, bias=False)
        self.conv_block = ConvNeXt_Block(dim)
        self.fre_awre = Frequency_Aware_Block(dim)
        self.attention_type = attention_type

        self.channel_gate = SSE_Block(dim, reduction=reduction)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.ones(heads, 1, 1))

    
    def forward(self, x_in):
        """
        x_in: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x_in.shape

        if self.attention_type == 'base':
            x = x_in
        if self.attention_type == 'spatial_conv':
            x = self.conv_block(x_in) # ConvNeXt block

        q_in = self.q_dwconv(self.to_q(x))
        k_in = self.k_dwconv(self.to_k(x))
        v_in = self.v_dwconv(self.to_v(x))

        q = rearrange(q_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
  
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        atten = (q @ k.transpose(-2, -1)) * self.rescale
        atten = atten.softmax(dim=-1)

        gate = self.channel_gate(x)   # [B, dim, 1, 1]
        gate = rearrange(gate, 'b (head c) 1 1 -> b head c', head=self.num_heads)
        gate_matrix = gate.unsqueeze(-1) * gate.unsqueeze(-2)
        atten = atten * gate_matrix
        atten = atten / (atten.sum(dim=-1, keepdim=True) + 1e-6)


        out = (atten @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)

        return out



class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PreNorm(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.relu = nn.GELU()
        self.out_conv = nn.Conv2d(dim * 2, dim, 1, 1, bias=False)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out = torch.cat((out1, out2), dim=1)
        return self.out_conv(self.relu(out))
    


class LayerNorm_ConvNext(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class S2RNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=61, dim=32, deep_stage=3, num_blocks=[1, 1, 1], num_heads=[1, 2, 4], attention_type='base'):
        # input: mosaic (b, 1, h, w)
        # output: hsi (b, 61, h, w)
        super(S2RNet, self).__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.stage = deep_stage # num of stage

        # Image embedding: for mosaic image -> extend dimension
        self.embedding1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)

        # Mask embeding: for 61/121 channel mask after calibration
        self.embedding2 = nn.Conv2d(out_channels, dim, kernel_size=3, padding=1, bias=False)

        # Image+Mask embedding after concating Image and Mask
        self.embedding = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False)
        
        # self.down_sample = nn.Conv2d(dim, dim, 4, 2, 1, bias=False)
        # self.up_sample = nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0)

        self.mapping = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

        # Encoder -> expand channel dimension, feature (spatial) downsample
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(deep_stage):
            self.encoder_layers.append(nn.ModuleList([
                SAM_Gated(dim=dim_stage, heads=num_heads[i], num_blocks=num_blocks[i], attention_type=attention_type),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottle neck
        self.bottleneck = SAM_Gated(
            dim=dim_stage, heads=num_heads[-1], num_blocks=num_blocks[-1], attention_type=attention_type)

        # Decoder -> expand spatial dimension(upsample)
        # spectral(channel): 128 -> 64 -> 32
        self.decoder_layers = nn.ModuleList([])
        for i in range(deep_stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False), # add residual
                SAM_Gated(dim=dim_stage // 2, heads=num_heads[deep_stage - 1 - i], num_blocks=num_blocks[deep_stage - 1 - i], attention_type=attention_type),
            ]))
            dim_stage //= 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # x: [b, 1, h, w] -> [b, dim, h, w] dim=32
        # mask: [b, 61, h, w] -> [b, dim, h, w] dim=32
        # cat(x, mask): [b, dim*2, h, w]
        x = self.embedding1(x)
        mask = self.embedding2(mask)
        x = torch.cat((x, mask), dim=1)

        # fea: [b, dim*2, h, w] -> [b, dim, h, w]
        fea = self.embedding(x)
        residual = fea
        # fea = self.down_sample(fea)

        fea_encoder = []
        # Spectral Attention + Conv for Feature downsample
        # Attention: [b, dim, h, w]
        # FeaDownSample: [b, dim, h, w] -> [b, dim*2, h//2, w//2]
        for (Attention, FeaDownSample) in self.encoder_layers:
            fea = Attention(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

    
        fea = self.bottleneck(fea)

        # Spectral Attention + residual + TransposeConv for Feature downsample
        # FeaUpSample: [b, dim*2, h//2, w//2] -> [b, dim, h, w]
        # Fution: [b, dim*2(with residual), h, w] -> [b, dim, h, w]
        # Attention: [b, dim, h, w]
        for i, (FeaUpSample, Fution, Attention) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1)) # add residual
            fea = Attention(fea)
 
        # fea = self.up_sample(fea)
        out = fea + residual
        out = self.mapping(out)

        return out





if __name__ == '__main__':
    model = model = S2RNet(in_channels=1, out_channels=40, dim=32, deep_stage=3, num_blocks=[1, 1, 1, 1], num_heads=[1, 2, 4, 8], attention_type='spa_fre_conv')
    input = torch.rand(size=(1, 1, 256, 256))
    mask = torch.rand(size=(1, 46, 256, 256))
    output_tensor = model.forward(input, mask)
    print(output_tensor.shape)






