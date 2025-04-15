
import torch
import torch.nn as nn


class MLP(nn.Module):
    """ MLP with n_units hidden units with size hidden_size """
    def __init__(self, n_units=1, hidden_size=512, num_classes=10):
        super().__init__()

        self.flatten = nn.Flatten()
        self.input_adapter = nn.Linear(28*28*3, hidden_size)

        layer_sizes = [hidden_size] * n_units
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

        self.head = nn.Linear(layer_sizes[-1], num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_adapter(x)  # hidden_size
        x = self.mlp(x)  # blocks
        x = self.head(x)  # logits
        return x
