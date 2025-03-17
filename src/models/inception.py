
from ipdb import launch_ipdb_on_exception

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision.models import Inception3

class ConvModule(nn.Module):
    """ Conv -> BatchNom -> ReLU """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=False,
        )
        if bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        else:
            self.bn = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionModule(nn.Module):
    """ Combination of 1x1 and 3x3 convolutions """
    def __init__(self, in_channels, out_channels1, out_channels3, bn=True):
        super().__init__()
        self.branch1 = ConvModule(in_channels, out_channels1, kernel_size=1, bn=bn)
        self.branch3 = ConvModule(in_channels, out_channels3, kernel_size=3, bn=bn)

    def forward(self, x):
        # just change the output channel size
        x1 = self.branch1(x)
        x2 = self.branch3(x)
        # exit with out_channels1 + out_channels3
        return torch.cat([x1, x2], dim=1)


class DownSampleModule(nn.Module):
    """ Downsample with 3x3 conv and 3x3 max pooling """
    def __init__(self, in_channels, out_channels3, bn=True):
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels3, kernel_size=3, stride=2, bn=bn)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.max_pool(x)
        return torch.cat([x1, x2], dim=1)


class InceptionSmall(nn.Module):
    """
    Small version of Inception for 28x28 images
    with flag for removing batch norm
    """
    def __init__(self, num_classes=10, bn=True):
        super().__init__()
        # first block
        self.Conv2d_3x3 = ConvModule(3, 96, kernel_size=3, bn=bn)  # exit 96
        # second block
        self.inception1 = InceptionModule(96, 32, 32, bn=bn)  # exit 64
        self.inception2 = InceptionModule(64, 32, 48, bn=bn)  # exit 80
        self.downsample1 = DownSampleModule(80, 80, bn=bn)  # exit 160
        # third block
        self.inception3 = InceptionModule(160, 112, 48, bn=bn)  # exit 160
        self.inception4 = InceptionModule(160, 96, 64, bn=bn)  # exit 160
        self.inception5 = InceptionModule(160, 80, 80, bn=bn)  # exit 160
        self.inception6 = InceptionModule(160, 48, 96, bn=bn)  # exit 144
        self.downsample2 = DownSampleModule(144, 96, bn=bn)  # exit 240
        # fourth block
        self.inception7 = InceptionModule(240, 176, 160, bn=bn)  # exit 336
        self.inception8 = InceptionModule(336, 176, 160, bn=bn)  # exit 336
        self.mean_pooling = nn.AvgPool2d(kernel_size=7, padding=1)  # flattened to 1x1x336
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


# def main():
#     from models_utils import visualize

#     ## Inception small
#     model = InceptionSmall()
#     input_data = torch.randn(64, 3, 28, 28)
#     visualize(model, "InceptionSmall", input_data)

#     ## Inception small without batch norm
#     model = InceptionSmall(bn=False)
#     input_data = torch.randn(64, 3, 28, 28)
#     visualize(model, "InceptionSmall", input_data)


# if __name__ == "__main__":
#     with launch_ipdb_on_exception():
#         main()
