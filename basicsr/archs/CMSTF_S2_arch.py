import torch
import time
from torch import nn as nn
from torch import Tensor
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from basicsr.archs.common import *
from basicsr.archs.ldm.ddpm import DDPM
from basicsr.archs.CMSTF_S1_arch import Deblur_S1_arch


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


class ResMLP(nn.Module):
    """
    残差 MLP (多层感知机)
    """

    def __init__(self, channels: int = 512) -> None:
        super(ResMLP, self).__init__()

        self.resmlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.resmlp(x)
        return res


class ConditionEncoder(nn.Module):
    """
    条件编码器
    """

    def __init__(self, channels: int = 64, n_encoder_res: int = 6, spatial_scale: tuple = (1, 1)) -> None:
        super(ConditionEncoder, self).__init__()

        E1 = [nn.Conv2d(48, channels, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]
        E2 = [
            ResBlock(channels) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(spatial_scale),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(channels * 4, channels * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(channels * 4, channels * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B C H W]
        """
        x = self.pixel_unshuffle(x)
        x = self.E(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        x = self.mlp(x)
        return x


class DenoiseNetwork(nn.Module):
    def __init__(self, channels: int, denoise_blocks: int, timesteps: int) -> None:
        super(DenoiseNetwork, self).__init__()

        self.max_period = timesteps * 10
        n_featsx4 = 4 * channels
        resmlp = [
            nn.Linear(n_featsx4 * 2 + 1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(denoise_blocks):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp = nn.Sequential(*resmlp)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        t = t.float()
        t = t / self.max_period
        t = t.view(-1, 1)
        x = torch.cat([c, t, x], dim=1)
        x = self.resmlp(x)

        return x


class DiffusionModel(nn.Module):
    def __init__(self, channels: int, denoise_blocks: int, linear_start: float, linear_end: float, timesteps: int) -> None:
        super(DiffusionModel, self).__init__()

        self.CE = ConditionEncoder(channels)
        self.Denoise = DenoiseNetwork(channels, denoise_blocks, timesteps)
        self.Diffusion = DDPM(denoise=self.Denoise,
                              condition=self.CE,
                              n_feats=channels,
                              linear_start=linear_start,
                              linear_end=linear_end,
                              timesteps=timesteps)

    def forward(self, x: Tensor, y=None):
        if self.training:
            IPR, IPR_list = self.Diffusion(x, y)
            return IPR, IPR_list
        else:
            IPR = self.Diffusion(x)
            return IPR


# @ARCH_REGISTRY.register()
class Deblur_S2_arch(nn.Module):
    def __init__(self, diff_channels: int,
                 deblur_channels: int,
                 event_dim: int,
                 attn_blocks: int,
                 head: int,
                 fusion_blocks: int,
                 denoise_blocks: int,
                 linear_start: float,
                 linear_end: float,
                 timesteps: int,
                 spatial_scale: tuple = (3, 3)) -> None:
        super().__init__()

        self.spatial_scale = spatial_scale
        self.DeblurNet = Deblur_S1_arch(deblur_channels, event_dim, attn_blocks, head, fusion_blocks)
        self.DM = DiffusionModel(channels=diff_channels, denoise_blocks=denoise_blocks, linear_start=linear_start,
                                 linear_end=linear_end, timesteps=timesteps)

        self.DownLayer = nn.Sequential(
            nn.Conv2d(deblur_channels * 4, deblur_channels * 4, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lin = nn.Linear(deblur_channels * 4, deblur_channels * 4)

        self.UpLayer = nn.Sequential(
            nn.Conv2d(deblur_channels * 4, deblur_channels * 4, kernel_size=1),
            nn.Conv2d(deblur_channels * 4, deblur_channels * 4, kernel_size=1),
            nn.Upsample(size=(2, 2), mode='bilinear'),
            nn.Conv2d(deblur_channels * 4, deblur_channels * 4, kernel_size=1),
            nn.Conv2d(deblur_channels * 4, deblur_channels * 4, kernel_size=1),
            nn.Upsample(size=(3, 3), mode='bilinear'),
            ResBlock(channels=diff_channels * 4),
            ResBlock(channels=diff_channels * 4)
        )

    def forward(self, x: Tensor, e: Tensor, y: Tensor):
        if self.training:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            y = rearrange(y, 'b t c h w -> (b t) c h w')
            y1 = self.DeblurNet.LE(x, y)
            y2 = self.DownLayer(y1)
            y2 = torch.squeeze(y2, dim=-1)
            y2 = torch.squeeze(y2, dim=-1)
            x, y = self.DM(x, y2)
            x = torch.unsqueeze(x, dim=-1)
            x = torch.unsqueeze(x, dim=-1)
            x = self.UpLayer(x)

            return y1, x

        else:
            s = self.DM(rearrange(x, 'b t c h w -> (b t) c h w'))
            s = torch.unsqueeze(s, dim=-1)
            s = torch.unsqueeze(s, dim=-1)
            s = self.UpLayer(s)
            out = self.DeblurNet(x, e, s, use_LE=False)

            return out.contiguous()
