
import torch
import torch.nn as nn


class MLP1(nn.Module):
    """A 2-layers MLP with given hidden size"""
    def __init__(self, hidden_size=512, num_classes=10):
        super().__init__()

        self.flatten = nn.Flatten()

        self.layer1 = nn.Linear(28*28*3, hidden_size)
        self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)  # 28*28*3

        x = self.relu(self.layer1(x))  # hidden_size

        x = self.classifier(x)  # logits
        return x


class MLP3(nn.Module):
    """A 4-layers MLP with given hidden size"""
    def __init__(self, hidden_size=512, num_classes=10):
        super().__init__()

        self.flatten = nn.Flatten()

        self.layer1 = nn.Linear(28*28*3, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)

        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)  # 28*28*3

        x = self.relu(self.layer1(x))  # hidden_size
        x = self.relu(self.layer2(x))  # hidden_size
        x = self.relu(self.layer3(x))  # hidden_size

        x = self.head(x)  # logits
        return x
