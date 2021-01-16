import torch
from torch import nn
from itertools import product
import math


class PriorboxGenerator(nn.Module):
    def __init__(self, image_size, pyramid_sizes, box_sizes, aspect_ratios):
        """
        先验框生成器.

        Arguments:
            image_size: int or list
                图像尺寸.
            pyramid_sizes: list
                各尺度特征图的size: [38, 19, 10, 5, 3, 1]
            box_sizes: list
                box的尺度（非比例）: [[sk, sk_prime], ...]  # 
            aspect_ratios: list[list]
                纵横比: [[2, 3], ...]
        """
        super().__init__()
        if isinstance(image_size, list):
            self.image_size = image_size[0]
        else:
            self.image_size = image_size

        self.pyramid_sizes = pyramid_sizes
        self.box_sizes = box_sizes
        self.aspect_ratios = aspect_ratios

    def forward(self):
        priorboxs = []

        s = [[box_size[0] / self.image_size, box_size[1] / self.image_size] for box_size in self.box_sizes]
        s = [[s[k][0], math.sqrt(s[k][0] * s[k + 1][0])] for k in range(len(s) - 1)]  # box的尺度比例: [[sk, sk_prime], ...]
        s.append(s[-1])

        for k, pyramid_size in enumerate(self.pyramid_sizes):
            cxcy = product(range(pyramid_size), repeat=2)  # box的中心
            sk, sk_prime = s[k]
            aspect_ratios_k  = self.aspect_ratios[k]
            
            boxs = []
            for cx, cy in cxcy:
                boxs.append(self._make_box(cx, cy, sk, 1))
                boxs.append(self._make_box(cx, cy, sk_prime, 1))

                for aspect_ratio in aspect_ratios_k:
                    boxs.append(self._make_box(cx, cy, sk, aspect_ratio))
                    boxs.append(self._make_box(cx, cy, sk, 1 / aspect_ratio))

            priorboxs.append(boxs)
        
        priorboxs = [torch.tensor(priorboxs_k) for priorboxs_k in priorboxs]
        priorboxs = torch.cat(priorboxs, dim=0).view(-1, 4)
        return priorboxs
                
    def _make_box(self, cx, cy, box_scale, aspect_ratio):
        w =  box_scale * math.sqrt(aspect_ratio)
        h =  box_scale * math.sqrt(aspect_ratio)
        xmin = (cx - 0.5) * w
        ymin = (cy - 0.5) * h
        xmax = (cy + 0.5) * w
        ymax = (cy + 0.5) * w
        return (xmin, ymin, xmax, ymax)


if __name__ == "__main__":
    image_size = 300
    pyramid_sizes = [38, 19, 10, 5, 3, 1]
    box_sizes = [[30, 60], [60, 111], [111, 162], [162, 213], [213, 264], [264, 315]]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    priorbox_gen = PriorboxGenerator(image_size, pyramid_sizes, box_sizes, aspect_ratios)
    priorboxs = priorbox_gen.forward()
    print(priorboxs.shape)