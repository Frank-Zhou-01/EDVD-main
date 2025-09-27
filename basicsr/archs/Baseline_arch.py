import numbers

import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn import functional as Fun

from basicsr.archs.kpn_pixel import IDynamicDWConv
from basicsr.utils.registry import ARCH_REGISTRY


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DownSampling(nn.Module):
    """
    PixelUnshuffle 下采样两倍
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DownSampling, self).__init__()

        self.conv = nn.Conv2d(in_channels * 2 * 2, out_channels, kernel_size=1)
        self.down = nn.PixelUnshuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.conv(x)
        return x


class UpSampling(nn.Module):
    """
    PixelUnshuffle 上采样两倍
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpSampling, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels * 2 * 2, kernel_size=1)
        self.down = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.down(x)
        return x


class ResBlock(nn.Module):
    """
    残差块 (3x3卷积, LeakReLU, 3x3卷积)
    """
    def __init__(self, channels: int) -> None:
        super(ResBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(x)


@ARCH_REGISTRY.register()
class Baseline_arch(nn.Module):
    def __init__(self, channels: int, attn_blocks: int) -> None:
        super().__init__()

        scale2_channels = channels + channels // 2
        scale3_channels = channels + channels

        self.project_in = nn.Conv3d(3, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.project_out = nn.Conv3d(channels, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.conv_down1 = nn.Conv2d(channels, scale2_channels, kernel_size=3, stride=2, padding=1)
        self.conv_down2 = nn.Conv2d(scale2_channels, scale3_channels, kernel_size=3, stride=2, padding=1)

        self.up1 = UpSampling(in_channels=scale3_channels, out_channels=scale2_channels)
        self.up2 = UpSampling(in_channels=scale2_channels, out_channels=channels)

        self.EventEnhancementModule = nn.Sequential(
            *[nn.Conv2d(scale3_channels, scale3_channels, kernel_size=3, padding=1) for _ in range(attn_blocks)]
        )

        self.Backward = nn.Conv3d(scale2_channels, scale2_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Forward = nn.Conv3d(scale2_channels, scale2_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.Reconstruct = nn.Sequential(
            *[ResBlock(channels) for _ in range(2)]
        )

    def forward(self, x: Tensor, e: Tensor, gt: Tensor, use_LE=False) -> Tensor:
        res0 = x
        B, T, C, H, W = x.shape

        x = self.project_in(rearrange(x, 'B T C H W -> B C T H W'))
        x = self.conv_down1(rearrange(x, 'B C T H W -> (B T) C H W'))
        res1 = x
        x = self.conv_down2(x)

        for block in self.EventEnhancementModule:
            x = block(x)

        x = self.up1(x) + res1
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)

        x = self.Backward(x)
        x = self.Forward(x)
        x = rearrange(x, 'B C T H W -> (B T) C H W')

        x = self.up2(x)
        x = self.Reconstruct(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        x = self.project_out(x)
        x = rearrange(x, 'B C T H W -> B T C H W')

        return x.contiguous() + res0
