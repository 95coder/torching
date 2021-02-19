from torch import nn


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=1, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=0, bias=False, dilation=1)


def maxpool2x2():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def upconv2x2(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,
                              stride=1, padding=0, bias=False, dilation=1)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1_1 = conv3x3(in_channels, 1)
        self.conv1_2 = conv3x3(1, 64)
        self.maxpool1 = maxpool2x2()

        self.conv2_1 = conv3x3(64, 128)
        self.conv2_2 = conv3x3(128, 128)
        self.maxpool2 = maxpool2x2()

        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(256, 256)
        self.maxpool3 = maxpool2x2()

        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(512, 512)
        self.maxpool4 = maxpool2x2()

        self.conv5_1 = conv3x3(512, 1024)
        self.conv5_2 = conv3x3(1024, 1024)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.maxpool1(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.maxpool2(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.maxpool3(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.maxpool4(out)

        out = self.conv5_1(out)
        out = self.conv5_2(out)

        return out


class UpSampleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.upconv1 = 


if __name__ == "__main__":
    from torchsummary import summary

    block = DownSampleBlock(3)
    summary(block, (3, 572, 572), device='cpu')