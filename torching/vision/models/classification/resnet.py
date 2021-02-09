import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False, dilation=dilation)


def conv1x1_downsample(in_channels, out_channels, stride=1):
    return nn.Sequential(
        conv1x1(in_channels, out_channels, stride=stride),
        nn.BatchNorm2d(out_channels)
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x) # (N, in_channels, H, W)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out) # (N, out_channels, H, W)
        out = self.relu(out)
        out = self.bn2(out)
        if self.downsample is not None:
            indentity = self.downsample(x) # (N, out_channels, H, W)
        else:
            indentity = x # (N, in_channels, H, W)
        out += indentity  # 形状会匹配吗?
        out = F.relu(out)
        return out


class BottleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x) # (N, out_channels, H, W)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out) # (N, out_channels, H, W)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv3(out) # (N, out_channels, H, W)
        out = self.relu(out)
        out = self.bn3(out)
        if self.downsample is not None:
            indentity = self.downsample(x) # (N, out_channels, H, W)
        else:
            indentity = x # (N, in_channels, H, W)
        out += indentity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block_cls, cfg, num_classes=10, zero_init_residual=False):
        super().__init__()
        in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_blocks(block_cls, cfg[0], 64, 64, 1),
            self._make_blocks(block_cls, cfg[1], 64, 128, 2),
            self._make_blocks(block_cls, cfg[2], 128, 256, 2),
            self._make_blocks(block_cls, cfg[3], 256, 512, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(512, num_classes)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, BottleBlock):
                    nn.init.constant_(m.bn3.weight, 0)
    
    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return out

    @staticmethod
    def _make_blocks(block_cls, num_blocks, in_channels, out_channels, stride=1):
        blocks = []
        
        dawnsample = None
        if (stride != 1) or (in_channels != out_channels):
            dawnsample = conv1x1_downsample(in_channels, out_channels, stride=stride)
        blocks.append(block_cls(in_channels, out_channels, stride=stride, downsample=dawnsample))

        for _ in range(1, num_blocks):
            blocks.append(block_cls(out_channels, out_channels))

        return nn.Sequential(*blocks)


cfgs = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3],
}


def _make_resnet(arch, block_cls, pretrained, weights_file=None):
    assert(arch in cfgs)
    if pretrained: assert(weights_file)

    model = ResNet(block_cls, cfgs[arch])

    if pretrained:
        state_dict = torch.load(weights_file)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, weights_file=None):
    return _make_resnet('resnet18', ResidualBlock, pretrained, weights_file)


def resnet34(pretrained=False, weights_file=None):
    return _make_resnet('resnet34', ResidualBlock, pretrained, weights_file)


def resnet50(pretrained=False, weights_file=None):
    return _make_resnet('resnet50', BottleBlock, pretrained, weights_file)


def resnet101(pretrained=False, weights_file=None):
    return _make_resnet('resnet101', BottleBlock, pretrained, weights_file)


def resnet152(pretrained=False, weights_file=None):
    return _make_resnet('resnet152', BottleBlock, pretrained, weights_file)



if __name__ == "__main__":
    from torchsummary import summary

    resnet = resnet34()
    summary(resnet, (3, 224, 224), device='cpu')