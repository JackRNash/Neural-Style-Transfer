import torch.nn as nn
import torch.nn.functional as F

'''
Use to build the image transformation network. Explanation of existence of modelDeprecated explained in it
'''

class ImageTransNet(nn.Module):

    def __init__(self):
        # Takes in an RGB image, in training this image will be 256 x 256
        super(ImageTransNet, self).__init__()
        self.pad1 = nn.ReflectionPad2d(4)
        self.pad2 = nn.ReflectionPad2d((1, 0, 1, 0))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1)
        self.norm32 = nn.InstanceNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.norm64 = nn.InstanceNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.norm128 = nn.InstanceNorm2d(128)

        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.res5 = ResBlock()

        self.upsample_conv1 = UpsampleConv((128, 128), 128, 64, mode='nearest')
        self.upsample_conv2 = UpsampleConv((256, 256), 64, 32, mode='nearest')

        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1)
        self.norm3 = nn.InstanceNorm2d(3)

    def forward(self, x_in, dimensions=(256, 256)):
        x = self.conv1(self.pad1(x_in))
        x = F.relu(self.norm32(x))

        x = self.pad2(x)
        x = self.conv2(x)
        x = F.relu(self.norm64(x))

        x = self.pad2(x)
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

        x = self.conv4(self.pad1(x))
        x = F.relu(self.norm3(x))

        return x


# Residual block, as defined in the paper
class ResBlock(nn.Module):

    def __init__(self, channels_in=128, kernel_size=3, filters=[128, 128]):
        super(ResBlock, self).__init__()
        f1, f2 = filters
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))
        self.conv1 = nn.Conv2d(channels_in, f1, kernel_size)
        self.norm1 = nn.InstanceNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size)
        self.norm2 = nn.InstanceNorm2d(f2)

    def forward(self, x):
        x_input = x
        x = self.conv1(self.pad(x))
        x = F.relu(self.norm1(x))
        x = self.conv2(self.pad(x))
        return self.norm2(x) + x_input


# Replaces the transposed convolution module as this reduces artifacts in the generated images
class UpsampleConv(nn.Module):

    def __init__(self, size, in_channels, out_channels, mode='nearest'):
        super(UpsampleConv, self).__init__()
        self.upsample = nn.Upsample(size=size, mode=mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(self.pad(x))
        return x
