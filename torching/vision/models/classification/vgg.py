import torch
import warnings
from torch import nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifer(out)
        return out

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


def _make_feature_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _make_vgg(arch, batch_norm, pretrained, weights_file=None):
    assert(arch in cfgs)

    features = _make_feature_layers(cfgs[arch], batch_norm)

    if not pretrained:
        model = VGG(features, init_weights=True)
    else:
        model = VGG(features, init_weights=False)
        if not weights_file:
            warnings.warn('No weights file specified.')
        else:
            state_dict = torch.load(weights_file)
            model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, **kwargs):
    return _make_vgg('vgg11', False, pretrained, **kwargs)


def vgg13(pretrained=False, **kwargs):
    return _make_vgg('vgg13', False, pretrained, **kwargs)

    
def vgg16(pretrained=False, **kwargs):
    return _make_vgg('vgg16', False, pretrained, **kwargs)


def vgg19(pretrained=False, **kwargs):
    return _make_vgg('vgg19', False, pretrained, **kwargs)


if __name__ == "__main__":
    from torchsummary import summary

    vgg = vgg11()
    # print(vgg)
    input_shape = (3, 100, 100)
    summary(vgg, input_shape, device='cpu')