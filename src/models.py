
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import visualize


class SimpleBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        return self.layers(x)


class Net(nn.Module):
    def __init__(self, num_filters, mlp_size, num_classes):
        super().__init__()
        self.blocks = nn.Sequential(
            SimpleBlock(3, num_filters, (5, 5)),
            SimpleBlock(num_filters, num_filters*2, (3, 3)),
            SimpleBlock(num_filters*2, num_filters*4, (3, 3))
        )
        self.bottleneck = nn.Conv2d(
            in_channels=num_filters*4,
            out_channels=num_filters,
            kernel_size=(1, 1)
        )
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(num_filters*3*3, mlp_size),
            nn.ReLU()
        )
        self.head = nn.Linear(mlp_size, num_classes)

    def forward(self, x):
        h = x
        h = self.blocks(h)
        h = self.bottleneck(h)
        h = self.flatten(h)
        h = self.mlp(h)
        h = self.head(h)
        return h


def main():

    ## Visualize simple network
    model = Net(8, 128, 10)
    input_data = torch.randn(64, 3, 28, 28)
    visualize(model, "Net", input_data)


if __name__ == "__main__":
    main()