import sys
# sys.path.append('/root/DiffIR-master/DiffIR-demotionblur/')

from archs.S2_arch import DiffIRS2

import torch
from thop import profile


def get_params(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("总参数数量和: " + f'{(k / (1024 * 1024)):.2f}M')


def count_flops(name, model):
    device = torch.device('cuda')
    model = model.to(device)
    x = torch.randn((1, 3, 256, 256)).to(device)

    flops, params = profile(model, inputs=(x,))

    print(f'{name}: FLOPS: {flops / (1024 * 1024 * 1024 * 3)}G, Params: {params / (1024 * 1024)}M')


net = DiffIRS2()


count_flops('MVSSM-S:', net)



