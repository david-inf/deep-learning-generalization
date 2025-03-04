
import random
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split

from utils import plot_data

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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
        self.X = torch.tensor(X).permute(0, 3, 1, 2) / 255.0  # scale to [0,1]
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
        """
        returns:
            X: [C, W, H] (tensor)
            y: [num_classes] (one-hot encoded)
        """
        return self.X[idx], self.y[idx]


class LabelCorruptor:
    """
    Corrupts labels with a given probability

    Args:
        prob (float): probability of corrupting the label
            prob=0. means true labels
            prob>0 and prob<1. means partially corrupted labels
            prob=1. means random labels
    """
    def __init__(self, prob=0.):
        self.prob = prob  # fraction of labels to corrupt

    def __call__(self, y):
        # y is one-hot encoded
        label = torch.argmax(y).item()  # one-hot to int
        num_classes = len(y)

        if random.random() < self.prob:
            new_label = random.randint(0, num_classes - 2)
            if new_label == label:
                new_label += 1
        else:
            new_label = label

        new_y = torch.zeros_like(y)
        new_y[new_label] = 1.0

        return new_y


class DataCorruptor:
    """
    Corrupts data with a given probability in different manners

    Args:
        corruption_type (str): type of corruption
            "none": no corruption
            "shuffled pixels": random permutation is chosed and applied to all images (train and test)
            "random pixels": different random permutation is applied to each image independently
            "gaussian": gaussian distribution with matching mean and variance to the original dataset is used to generate random pixels for each image
        prob (float): probability of corrupting the data
    """
    def __init__(self, corruption_type, prob=0.):
        self.prob = prob
        self.corruption_type = corruption_type

    def __call__(self, X):
        # TODO: corruption types
        return X


class CorruptedCIFAR10(CIFAR10):
    """ Extends CIFAR10 Dataset corrupting labels and data """
    def __init__(self, opts, train=True, crop=True):
        super().__init__(opts, train, crop)
        # what about using v2.Compose?
        self.label_corruptor = LabelCorruptor(
            opts.label_corruption_prob
        )
        self.data_corruptor = DataCorruptor(
            opts.data_corruption_type,
            opts.data_corruption_prob
        )

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)  # tensor and one-hot
        y = self.label_corruptor(y)  # corrupt label with given probability
        X = self.data_corruptor(X)  # corrupt data with given probability and type

        return X, y


class AugmentedCIFAR10(CIFAR10):
    def __init__(self, opts=None, train=True):
        super().__init__(opts, train, crop=False)
        self.augmentation_pipeline = v2.Compose([
            v2.RandomRotation(degrees=self.opts.rotation_degrees, expand=False),
            v2.RandomCrop(size=28),  # 28x28 image
            v2.RandomHorizontalFlip(p=self.opts.horizontal_flip_prob)
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


def main_normalization():
    # Hyperparams
    config = dict(batch_size=128, num_workers=0)
    opts = SimpleNamespace(**config)

    # Get Dataset and DataLoader
    trainset = CIFAR10(opts)
    train_loader = make_loader(trainset, opts)

    # Verify data
    mean, std = mean_std_channel(train_loader)
    print("Base dataset")
    print("Mean:", mean)
    print("Std:", std)

    # Check labels and images
    X, y = next(iter(train_loader))
    print("Data:", X.shape)
    print("Targets:", y.shape)


def main_corruption():
    # Hyperparams
    config = dict(
        batch_size=128, num_workers=0, label_corruption_prob=0.2,
        data_corruption_prob=0.2, data_corruption_type="noise"
    )
    opts = SimpleNamespace(**config)

    # Get Dataset and DataLoader
    original_trainset = CIFAR10(opts)
    original_train_loader = make_loader(original_trainset, opts)
    corrupted_trainset = CorruptedCIFAR10(opts)
    corrupted_train_loader = make_loader(corrupted_trainset, opts)

    # Verify data
    # mean, std = mean_std_channel(train_loader)
    # print("Corrupted dataset")
    # print("Mean:", mean)
    # print("Std:", std)

    # Check labels and images
    # X, y = next(iter(corrupted_train_loader))
    # print("Data:", X.shape)
    # print("Targets:", y.shape)

    # done = 0
    # for X, y in corrupted_train_loader:
    #     print(X.shape)
    #     plot_data(
    #         [[xx for xx in X[:opts.batch_size//2]],[xx for xx in X[opts.batch_size//2:]]],
    #         [[yy for yy in y[:opts.batch_size//2]],[yy for yy in y[opts.batch_size//2:]]]
    #     )
    #     done += 1
    #     if done == 1:
    #         break

    # Check original labels againts corrupted labels
    confusion = np.zeros((10, 10), dtype=int)
    for i in range(1000):
        _, y = original_trainset[i]
        _, y_corrupted = corrupted_trainset[i]
        confusion[torch.argmax(y).item(), torch.argmax(y_corrupted).item()] += 1
    print(confusion)
    tot = confusion.sum()
    acc = confusion.trace() / tot
    print(f"Corruption rate: {1-acc:.2f}")


if __name__ == "__main__":
    # main_normalization()
    main_corruption()
