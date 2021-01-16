import torch
from torch import nn


_pyramid_cfg = [[256, 1, 1], # (out_channels, kernel_size, stride) 
                [512, 3, 2],
                [128, 1, 1],
                [256, 3, 2],
                [128, 1, 1],
                [256, 3, 1],
                [128, 1, 1],
                [256, 3, 1]]


class Pyramid(nn.Module):
    def __init__(self, in_channels=1024, cfg=_pyramid_cfg):
        super().__init__()
        self.feature_layers = self._make_layers(in_channels, cfg)

    def forward(self, x):
        outs = []
        for k, layer in enumerate(self.feature_layers):
            x = layer(x)
            if k % 2 == 1:
                """
                conv8_2: (None, 512, 10, 10)
                conv9_2: (None, 256, 5, 5)
                conv10_2: (None, 256, 3, 3)
                conv11_2: (None, 256, 1, 1)
                """
                outs.append(x)
        return outs

    def feature_out_layers(self):
        return self.feature_layers[1::2]

    def _make_layers(self, in_channels, layer_cfg):
        """
        创建除conv4_3的另外5个尺度的特征提取卷积层, 位于conv7的relu之后.
        """
        layers = []
        for out_channels, ksize, stride in layer_cfg:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=ksize, 
                                 stride=stride, padding=(0, 1)[stride == 2])] # 
            in_channels = out_channels
        return nn.ModuleList(layers)