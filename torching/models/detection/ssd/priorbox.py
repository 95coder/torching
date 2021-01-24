import torch
from torch import nn
from itertools import product
import math


def make_scales(min_scale, max_scale, num_pyramids):
    for i in range(num_pyramids):
        yield (min_scale + (max_scale - min_scale) / (num_pyramids - 1) * i)


class PriorboxGenerator(nn.Module):
    def __init__(self, pyramid_sizes, min_scale, max_scale, aspect_ratios):
        """
        先验框生成器.

        Arguments:
            pyramid_sizes: list
                各尺度特征图的size: [38, 19, 10, 5, 3, 1]
            aspect_ratios: list[list]
                纵横比: [[2, 3], ...]
        """
        super().__init__()
        self.pyramid_sizes = pyramid_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.aspect_ratios = aspect_ratios

    def forward(self):
        priorboxs = []

        # 计算各层的尺度
        scales = list(make_scales(self.min_scale, self.max_scale, len(self.pyramid_sizes)))
        scales = [self.min_scale / 2] + scales[:-1] 
        scales_prime = []
        for i in range(len(scales) - 1):
            scales_prime.append(math.sqrt(scales[i] * scales[i + 1]))
        scales_prime += [self.max_scale]

        for k, pyramid_size in enumerate(self.pyramid_sizes):
            sk = scales[k]
            sk_prime = scales_prime[k]
            aspect_ratios_k  = self.aspect_ratios[k]
            
            boxs = []
            for i, j in product(range(pyramid_size), repeat=2):
                cx = (i + 0.5) / pyramid_size
                cy = (j + 0.5) / pyramid_size

                boxs.append(self._make_box(cx, cy, sk, 1))
                boxs.append(self._make_box(cx, cy, sk_prime, 1))

                for aspect_ratio in aspect_ratios_k:
                    boxs.append(self._make_box(cx, cy, sk, aspect_ratio))
                    boxs.append(self._make_box(cx, cy, sk, 1 / aspect_ratio))

            priorboxs.append(boxs)
        
        priorboxs = [torch.tensor(priorboxs_k) for priorboxs_k in priorboxs]
        priorboxs = torch.cat(priorboxs, dim=0).view(-1, 4)
        return priorboxs
                
    def _make_box(self, cx, cy, scale, aspect_ratio):
        w =  scale * math.sqrt(aspect_ratio)
        h =  scale * math.sqrt(aspect_ratio)
        return [cx, cy, w, h]


if __name__ == "__main__":
    image_size = 300
    pyramid_sizes = [38, 19, 10, 5, 3, 1]
    min_scale = 0.2
    max_scale = 1.05
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    priorbox_gen = PriorboxGenerator(pyramid_sizes, min_scale, max_scale, aspect_ratios)
    priorboxs = priorbox_gen.forward()
    print(priorboxs)