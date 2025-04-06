
import torch
import torch.nn as nn
from torchvision.models import AlexNet


class AlexNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, k=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, k=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(256*3*3, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(192, num_classes)

    def forward(self, x):
        # N x 3 x 28 x 28
        x = self.conv1(x)
        x = self.conv2(x); print(x.shape)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.mlp(x)
        # N x 192
        x = self.head(x)
        # N x 10
        return x
