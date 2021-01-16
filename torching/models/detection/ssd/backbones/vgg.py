import torch
from torch import nn


class VGGBackBone(nn.Module):
    def __init__(self, features, feature_out_idxs, init_weights=True):
        super().__init__()
        self.features = features
        self.feature_out_idxs = feature_out_idxs

        if init_weights:
            self._init_weights()

    def forward(self, x):
        outs = []

        for layer in self.features:
            x = layer(x)
            outs.append(x)

        features_outs = [outs[i] for i in self.feature_out_idxs]
        return features_outs

    def feature_out_layers(self):
        return [self.features[i] for i in self.feature_out_idxs]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(layer_cfg, in_channels, out_channels, batch_norm=False):
    """
    创建vgg stem. 在原来的基础上将修改pool5, 将fc6、fc7替换为conv6、conv7.
    """
    assert layer_cfg[-1] == 512

    layers = []
    for v in layer_cfg:
        if v == 'MP':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MP_CEIL':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)] # ceil_mode=True是为了让conv4中特征图的形状为38x38 -> ceil(75 / 2)
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] # 这里将BatchNorm放在了激活层之前. 看了torchvision的某些网络的实现，也是将其放在之前.
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    maxpool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 将pool5从stride=2, 2x2的maxpool变为stride=1, 3x3的maxpool，
                                                                # 注意要将padding设置为1, 使输出特征图的形状和原来的一致.

    conv6 = nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=6, 
                      bias=False, dilation=6)  # conv6为stride=1, 3x3的卷积，且rate为6的空洞卷积. 将padding置为6, 使输出特征图的形状和原来的一致.
    conv7 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False) # conv7为1x1的卷积
    layers += [maxpool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


_archs = {
    'vgg16': {
        'layer_cfg': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP_CEIL', 512, 512, 512, 'MP', 512, 512, 512],
        'feature_out_idxs': [21, -2]
    },
    'vgg16_bn': {
        'layer_cfg': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP_CEIL', 512, 512, 512, 'MP', 512, 512, 512],
        'feature_out_idxs': [-1, -1]
    },
    'vgg19': {
        'layer_cfg': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 256, 'MP_CEIL', 512, 512, 512, 512, 'MP', 512, 512, 512, 512],
        'feature_out_idxs': [25, -2]
    },
    'vgg19_bn': {
        'layer_cfg': [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 256, 'MP_CEIL', 512, 512, 512, 512, 'MP', 512, 512, 512, 512],
        'feature_out_idxs': [-1, -1]
    }
}


def make_backbone(arch, *, in_channels=3, out_channels=1024):
    assert arch in _archs
    layer_cfg = _archs[arch]['layer_cfg']
    feature_out_idxs = _archs[arch]['feature_out_idxs']

    return VGGBackBone(make_layers(layer_cfg, in_channels, out_channels), 
                       feature_out_idxs=feature_out_idxs)


if __name__ == "__main__":
    from torchsummary import summary

    vgg = make_backbone('vgg16')
    # print(vgg)
    summary(vgg, (3, 300, 300), device='cpu')