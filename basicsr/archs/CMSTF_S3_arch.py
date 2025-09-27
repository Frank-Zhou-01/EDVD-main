import torch
from torch import Tensor
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.CMSTF_S2_arch import Deblur_S2_arch


@ARCH_REGISTRY.register()
class Deblur_S3_arch(Deblur_S2_arch):
    def forward(self, x: Tensor, e: Tensor, y: Tensor):
        if self.training:
            KS = self.DeblurNet.LE(rearrange(x, 'b t c h w -> (b t) c h w'),
                                       rearrange(x, 'b t c h w -> (b t) c h w'))
            IPR_S2 = self.DownLayer(KS)
            IPR_S2 = torch.squeeze(IPR_S2, dim=-1)
            IPR_S2 = torch.squeeze(IPR_S2, dim=-1)
            IPR_DM, IPR_list = self.DM(rearrange(x, 'b t c h w -> (b t) c h w'), IPR_S2)

            IPR_DM = torch.unsqueeze(IPR_DM, dim=-1)
            IPR_DM = torch.unsqueeze(IPR_DM, dim=-1)
            IPR_DM = self.UpLayer(IPR_DM)
            output = self.DeblurNet(x, e, IPR_DM)

            return output.contiguous(), KS, IPR_DM
        else:
            s = self.DM(rearrange(x, 'b t c h w -> (b t) c h w'))
            s = torch.unsqueeze(s, dim=-1)
            s = torch.unsqueeze(s, dim=-1)
            s = self.UpLayer(s)
            out = self.DeblurNet(x, e, s, use_LE=False)

            return out.contiguous()
