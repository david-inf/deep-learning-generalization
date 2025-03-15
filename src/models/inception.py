
from ipdb import launch_ipdb_on_exception

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import Inception3

from models_utils import visualize

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels3):
        super().__init__()
        self.branch1 = ConvModule(in_channels, out_channels1, kernel_size=(1, 1), stride=(1,1))
        self.branch2 = ConvModule(in_channels, out_channels3, kernel_size=(3, 3), stride=(1,1), padding="same")

    def forward(self, x):
        # just change the output channel size
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # exit with out_channels1 + out_channels3
        return torch.cat([x1, x2], dim=1)


class DownSampleModule(nn.Module):
    def __init__(self, in_channels, out_channels3):
        super().__init__()
        self.branch1 = ConvModule(in_channels, out_channels3, kernel_size=(3, 3), stride=(2, 2))
        self.branch2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat([x1, x2], dim=1)


class InceptionSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # first block
        self.Conv2d_3x3 = ConvModule(3, 96, kernel_size=(3, 3), stride=(1, 1))  # exit 96
        # second block
        self.inception1 = InceptionModule(96, 32, 32)  # exit 64
        self.inception2 = InceptionModule(64, 32, 48)  # exit 80
        self.downsample1 = DownSampleModule(80, 80)  # exit 160
        # third block
        self.inception3 = InceptionModule(160, 112, 48)  # exit 160
        self.inception4 = InceptionModule(160, 96, 64)  # exit 160
        self.inception5 = InceptionModule(160, 80, 80)  # exit 160
        self.inception6 = InceptionModule(160, 48, 96)  # exit 144
        self.downsample2 = DownSampleModule(144, 96)  # exit 240
        # fourth block
        self.inception7 = InceptionModule(240, 176, 160)  # exit 336
        self.inception8 = InceptionModule(336, 176, 160)  # exit 336
        self.mean_pooling = nn.AvgPool2d(kernel_size=(7, 7), padding=(1, 1))  # flattened to 1x1x336
        self.head = nn.Linear(336, num_classes)

    def forward(self, x):
        # N x 3 x 28 x 28
        x = self.Conv2d_3x3(x)# ; print(x.shape)
        # N x 96 x 26 x 26
        x = self.inception1(x)# ; print(x.shape)
        # N x 64 x 26 x 26
        x = self.inception2(x)# ; print(x.shape)
        # N x 80 x 26 x 26
        x = self.downsample1(x)# ; print(x.shape)
        # N x 160 x 12 x 12
        x = self.inception3(x)# ; print(x.shape)
        # N x 160 x 12 x 12
        x = self.inception4(x)# ; print(x.shape)
        # N x 160 x 12 x 12
        x = self.inception5(x)# ; print(x.shape)
        # N x 160 x 12 x 12
        x = self.inception6(x)# ; print(x.shape)
        # N x 144 x 12 x 12
        x = self.downsample2(x)# ; print(x.shape)
        # N x 240 x 12 x 12
        x = self.inception7(x)# ; print(x.shape)
        # N x 336 x 5 x 5
        x = self.inception8(x)# ; print(x.shape)
        # N x 336 x 5 x 5
        x = self.mean_pooling(x)# ; print(x.shape)
        # N x 336 x 1 x 1
        x = x.flatten(1)# ; print(x.shape)
        # N x 336
        x = self.head(x)# ; print(x.shape)
        # N x 10 (num_classes)
        return x


def main():
    model = InceptionSmall()
    input_data = torch.randn(64, 3, 28, 28)
    visualize(model, "Inception", input_data)


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
