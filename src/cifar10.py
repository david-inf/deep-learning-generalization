
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split


class CIFAR10(Dataset):
    def __init__(self, opts=None, train=True, crop=True):
        self.opts = opts

        # Get CIFAR10 dataset
        dataset = datasets.CIFAR10(  # (X, y)
            root="./data",
            train=train,
            download=True
        )

        X = dataset.data  # PIL Image
        y = dataset.targets  # int

        # Data pre-processing
        self.X = torch.tensor(X).permute(0, 3, 1, 2) / 255.0
        if crop:  # take central part of each image
            margin = (32 - 28) // 2  # 28x28 image
            self.X = self.X[:, :, margin:-margin, margin:-margin]

        # Target pre-processing
        self.num_classes = len(np.unique(y))
        EYE = np.eye(self.num_classes)
        y_oh = EYE[y]  # one-hot encoding
        self.y = torch.tensor(y_oh)

        # Data normalization
        mean = torch.mean(self.X, dim=(0, 2, 3))
        std = torch.std(self.X, dim=(0, 2, 3))
        # Apply normalization to the data
        for c in range(3):  # For each channel
            self.X[:, c, :, :] = (self.X[:, c, :, :] - mean[c]) / std[c]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AugmentedCIFAR10(CIFAR10):
    def __init__(self, opts=None, train=True):
        super().__init__(opts, train, crop=False)
        self.augmentation_pipeline = v2.Compose([
            v2.RandomRotation(degrees=self.opts.rotation_degrees, expand=False),
            v2.RandomCrop(size=28),  # 28x28 image
            v2.RandomHorizontalFlip(p=self.opts.horizontal_flip_prob),
            # v2.RandomVerticalFlip(p=self.opts.vertical_flip_prob)
        ])

    def __getitem__(self, idx):
        return self.augmentation_pipeline(self.X[idx]), self.y[idx]


def make_loader(data, opts):
    # data: Dataset object
    # opts: object whose attributes are the configs
    loader = DataLoader(
        data,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True
    )

    return loader


def mean_std_channel(loader):
    # following BFR algorithm for clustering
    N = 0  # pixel count for each channel
    SUM = torch.zeros(3)  # sum pixels channel-wise
    SUMSQ = torch.zeros(3)  # sum of squared pixels channels-wise
    with tqdm(loader, unit="batch") as tepoch:
        for (X, _) in tepoch:
            # print("Data:", X.shape)  # [B, C, W, H]
            # print("Targets:", y.shape)
            b, _, w, h = X.shape
            N += b * w * h
            SUM += X.sum(dim=[0, 2, 3])
            SUMSQ += X.pow_(2).sum(dim=[0, 2, 3])
            # print(f"Batch - mean: {X.mean():.3f}, std: {X.std():.3f}")
            # print(f"Batch - min: {X.min():.3f}, max: {X.max():.3f}")
            # break
    mean = SUM / N
    std = torch.sqrt((SUMSQ / N) - mean**2)

    return mean, std


if __name__ == "__main__":
    # Hyperparams
    config = dict(batch_size=128, num_workers=0)
    opts = SimpleNamespace(**config)

    # Get Dataset and DataLoader
    trainset = CIFAR10(opts)
    train_loader = make_loader(trainset, opts)

    # Verify data
    mean, std = mean_std_channel(train_loader)
    print("Mean:", mean)
    print("Std:", std)

    # Verify for augmented dataset
    config = dict(batch_size=64, num_workers=2,
                  rotation_degrees=15, horizontal_flip_prob=0.2)
    opts = SimpleNamespace(**config)
    trainset = AugmentedCIFAR10(opts)
    train_loader = make_loader(trainset, opts)
    mean, std = mean_std_channel(train_loader)
    print("Mean:", mean)
    print("Std:", std)
