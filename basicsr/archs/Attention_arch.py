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


class CAB(nn.Module):
    """
    通道注意力块
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(CAB, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.CA = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.conv(x)
        x = x * self.CA(x)
        return x + self.res_conv(res)


class HWT(nn.Module):
    def __init__(self, channel_in):
        super(HWT, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = Fun.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out[:, :self.channel_in, :, :], out[:, self.channel_in:self.channel_in * 4, :, :]
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return Fun.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)


class MLP(nn.Module):
    def __init__(self, channels: int) -> None:
        super(MLP, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(channels, channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = rearrange(x, 'N C H W -> N H W C')
        x = self.layer(x)
        x = rearrange(x, 'N H W C -> N C H W')
        return x + res


class GatedFFN3D(nn.Module):
    """
    3D门控前馈网络
    """
    def __init__(self, channels: int, expansion: float) -> None:
        super(GatedFFN3D, self).__init__()

        mid_channels = int(channels * expansion)
        self.conv_in = nn.Conv3d(channels, mid_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_out = nn.Conv3d(mid_channels // 2, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x: Tensor) -> Tensor:
        res = x

        x = rearrange(x, 'B T C H W -> B C T H W')
        x = self.conv_in(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = x1 * x2
        x = self.conv_out(x)
        x = rearrange(x, 'B C T H W -> B T C H W')
        return x + res


class HTA(nn.Module):
    """
    垂直转置注意力
    """
    def __init__(self, head: int) -> None:
        super(HTA, self).__init__()

        self.head = head
        self.beta = nn.Parameter(torch.ones(head, 1, 1))

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        N, C, H, W = q.shape

        q = rearrange(q, 'N (head C) H W -> N head H (W C)', head=self.head)
        k = rearrange(k, 'N (head C) H W -> N head H (W C)', head=self.head)
        v = rearrange(v, 'N (head C) H W -> N head H (W C)', head=self.head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.beta
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'N head H (W C) -> N (head C) H W', H=H, head=self.head, W=W)

        return out


class WTA(nn.Module):
    """
    横向转置注意力
    """
    def __init__(self, head: int) -> None:
        super(WTA, self).__init__()

        self.head = head
        self.beta = nn.Parameter(torch.ones(head, 1, 1))

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        N, C, H, W = q.shape

        q = rearrange(q, 'N (head C) H W -> N head W (H C)', head=self.head)
        k = rearrange(k, 'N (head C) H W -> N head W (H C)', head=self.head)
        v = rearrange(v, 'N (head C) H W -> N head W (H C)', head=self.head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.beta
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'N head W (H C) -> N (head C) H W', H=H, head=self.head)

        return out


class CTA(nn.Module):
    """
    通道转置注意力
    """

    def __init__(self, head: int) -> None:
        super(CTA, self).__init__()

        self.head = head
        self.beta = nn.Parameter(torch.ones(head, 1, 1))

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        N, C, H, W = q.shape

        q = rearrange(q, 'N (head C) H W -> N head C (H W)', head=self.head)
        k = rearrange(k, 'N (head C) H W -> N head C (H W)', head=self.head)
        v = rearrange(v, 'N (head C) H W -> N head C (H W)', head=self.head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.beta
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'N head C (H W) -> N (head C) H W', H=H, head=self.head)

        return out


class CMCA(nn.Module):
    """
    Cross-Modal Coordination Attention
    """
    def __init__(self, channels: int, head: int) -> None:
        super(CMCA, self).__init__()

        self.conv_in = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.dconv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv_out = nn.Conv3d(channels * 3, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_e = nn.Conv2d(channels, channels, kernel_size=1)

        self.HTA = HTA(head)
        self.WTA = WTA(head)
        self.CTA = CTA(head)

        self.mlp = MLP(channels * 3)

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> B C T H W')
        x = self.conv_in(x)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.dconv1(x)

        e = rearrange(e, 'B T C H W -> (B T) C H W')
        e = self.conv_e(e)

        # 跨模态协调注意力
        out_h = self.HTA(x, e, e)
        out_w = self.WTA(x, e, e)
        out_c = self.CTA(x, e, e)

        out = torch.cat((out_h, out_w, out_c), dim=1)
        out = self.mlp(out)
        out = rearrange(out, '(B T) C H W -> B C T H W', B=B, T=T)
        out = self.conv_out(out)
        out = rearrange(out, 'B C T H W -> B T C H W')

        return out


class CMBlock(nn.Module):
    def __init__(self, channels: int, head: int) -> None:
        super(CMBlock, self).__init__()

        self.norm = LayerNorm(channels, LayerNorm_type='WithBias')
        self.norm_e = LayerNorm(channels, LayerNorm_type='WithBias')
        self.attn = CMCA(channels, head)

    def forward(self, x: Tensor, e: Tensor, s: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        res = x
        x = rearrange(self.norm(rearrange(x, 'B T C H W -> (B T) C H W')), '(B T) C H W -> B T C H W', B=B)
        e = rearrange(self.norm_e(rearrange(e, 'B T C H W -> (B T) C H W')), '(B T) C H W -> B T C H W', B=B)
        x = self.attn(x, e)
        x = x + res

        return x


class LatentEncoder(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=5):
        super(LatentEncoder, self).__init__()
        E1 = [nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]

        E2 = [ResBlock(n_feats) for _ in range(n_encoder_res)]

        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((3, 3)),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(4)

    def forward(self, x, gt):
        gt0 = self.pixel_unshuffle(gt)
        x0 = self.pixel_unshuffle(x)
        x = torch.cat([x0, gt0], dim=1)
        feat = self.E(x)
        feat = rearrange(feat, 'B C H W -> B H W C')
        feat = self.mlp(feat)
        feat = rearrange(feat, 'B H W C -> B C H W')
        return feat


@ARCH_REGISTRY.register()
class Attention_arch(nn.Module):
    def __init__(self, channels: int, event_dim: int, attn_blocks: int, head: int) -> None:
        super().__init__()

        scale2_channels = channels + channels // 2
        scale3_channels = channels + channels

        self.project_in = nn.Conv3d(3, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.project_event = nn.Conv3d(event_dim, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.project_out = nn.Conv3d(channels, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.conv_down1 = nn.Conv2d(channels, scale2_channels, kernel_size=3, stride=2, padding=1)
        self.conv_down2 = nn.Conv2d(scale2_channels, scale3_channels, kernel_size=3, stride=2, padding=1)

        self.conv_down3 = nn.Conv2d(channels, scale2_channels, kernel_size=3, stride=2, padding=1)
        self.conv_down4 = nn.Conv2d(scale2_channels, scale3_channels, kernel_size=3, stride=2, padding=1)

        self.up1 = UpSampling(in_channels=scale3_channels, out_channels=scale2_channels)
        self.up2 = UpSampling(in_channels=scale2_channels, out_channels=channels)

        self.EventEnhancementModule = nn.Sequential(
            *[CMBlock(scale3_channels, head) for _ in range(attn_blocks)]
        )

        self.Backward = nn.Conv3d(scale2_channels, scale2_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.Forward = nn.Conv3d(scale2_channels, scale2_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.Reconstruct = nn.Sequential(
            *[ResBlock(channels) for _ in range(2)]
        )

        self.LE = LatentEncoder()

    def forward(self, x: Tensor, e: Tensor, gt: Tensor, use_LE=False) -> Tensor:
        res0 = x
        B, T, C, H, W = x.shape

        s = self.LE(rearrange(x, 'B T C H W -> (B T) C H W'), rearrange(gt, 'B T C H W -> (B T) C H W'))

        x = self.project_in(rearrange(x, 'B T C H W -> B C T H W'))
        x = self.conv_down1(rearrange(x, 'B C T H W -> (B T) C H W'))
        res1 = x
        x = self.conv_down2(x)
        x = rearrange(x, '(B T) C H W -> B T C H W', B=B)

        e = self.project_event(rearrange(e, 'B T C H W -> B C T H W'))
        e = self.conv_down3(rearrange(e, 'B C T H W -> (B T) C H W'))
        e1 = e
        e2 = self.conv_down4(e)
        e2 = rearrange(e2, '(B T) C H W -> B T C H W', B=B)

        for block in self.EventEnhancementModule:
            x = block(x, e2, s)

        x = self.up1(rearrange(x, 'B T C H W -> (B T) C H W')) + res1
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
