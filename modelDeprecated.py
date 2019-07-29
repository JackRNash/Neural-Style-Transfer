import torch.nn as nn
import torch.nn.functional as F

'''
Use to build the image transformation network.
Deprecated because it uses zero padding. The starry night model was trained using the deprecated network, but the
scream network was not and all future networks should not use the deprecated modules. The reflection padding reduces
artifacts in the generated images. This class is purely for reproducibility in generating the starry night model
'''


class ImageTransNetDeprecated(nn.Module):

    def __init__(self, shape=(256, 256)):
        # Takes in an RGB image, in training this image will be 256 x 256
        super(ImageTransNetDeprecated, self).__init__()
        height, width = shape # shape is a two dimension tuple of the dimensions of the image
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.norm32 = nn.InstanceNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.norm64 = nn.InstanceNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.norm128 = nn.InstanceNorm2d(128)

        self.res1 = ResBlockDeprecated()
        self.res2 = ResBlockDeprecated()
        self.res3 = ResBlockDeprecated()
        self.res4 = ResBlockDeprecated()
        self.res5 = ResBlockDeprecated()

        self.upsample_conv1 = UpsampleConv((int(height / 2), int(width / 2)), 128, 64, mode='nearest')
        self.upsample_conv2 = UpsampleConv((height, width), 64, 32, mode='nearest')

        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        self.norm3 = nn.InstanceNorm2d(3)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = F.relu(self.norm32(x))

        x = self.pad(x)
        x = self.conv2(x)
        x = F.relu(self.norm64(x))

        x = self.pad(x)
        x = self.conv3(x)
        x = F.relu(self.norm128(x))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.upsample_conv1(x)
        x = F.relu(self.norm64(x))

        x = self.upsample_conv2(x)
        x = F.relu(self.norm32(x))

        x = self.conv4(x)
        x = F.relu(self.norm3(x))

        return x


# Residual block, as defined in the paper
class ResBlockDeprecated(nn.Module):

    def __init__(self, channels_in=128, kernel_size=3, filters=[128, 128]):
        super(ResBlockDeprecated, self).__init__()
        f1, f2 = filters
        self.conv1 = nn.Conv2d(channels_in, f1, kernel_size, padding=int((kernel_size - 1) / 2))
        self.norm1 = nn.InstanceNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size, padding=int((kernel_size - 1) / 2))
        self.norm2 = nn.InstanceNorm2d(f2)

    def forward(self, x):
        x_input = x
        x = self.conv1(x)
        x = F.relu(self.norm1(x))
        x = self.conv2(x)
        return self.norm2(x) + x_input


# Upsample --> Convolution blocks to replace transposed convolution
class UpsampleConvDeprecated(nn.Module):

    def __init__(self, size, in_channels, out_channels, mode='nearest'):
        super(UpsampleConvDeprecated, self).__init__()
        self.upsample = nn.Upsample(size=size, mode=mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
