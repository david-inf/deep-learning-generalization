
from ipdb import launch_ipdb_on_exception

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import AlexNet


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
        )
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3))
        self.lrn = nn.LocalResponseNorm(size=5)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.lrn(x)
        return x


class AlexNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 192)
        self.mlp = nn.Sequential(
            nn.Linear(192, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(192, num_classes)

    def forward(self, x):
        # N x 3 x 28 x 28
        x = self.conv1(x)# ; print(x.shape)
        # N 64 x 8 x 8
        x = self.conv2(x)# ; print(x.shape)
        # N x 192 x 1 x 1
        x = torch.flatten(x, 1)# ; print(x.shape)
        # N x 192
        x = self.mlp(x)# ; print(x.shape)
        # N x 192
        x = self.head(x)# ; print(x.shape)
        # N x 10
        return x


# def main():
#     from models_utils import visualize
#     model = AlexNetSmall()
#     input_data = torch.randn(64, 3, 28, 28)
#     visualize(model, "AlexNetSmall", input_data)


# if __name__ == "__main__":
#     with launch_ipdb_on_exception():
#         main()
