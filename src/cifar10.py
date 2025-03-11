
import random
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
from ipdb import launch_ipdb_on_exception

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split

from utils import imshow

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ModifiedCIFAR10(Dataset):
    def __init__(self, opts=None, crop=True):
        self.opts = opts

        # Get full CIFAR10 dataset, first separated
        trainset = datasets.CIFAR10(  # (X, y) N=50000
            root="./data",
            train=True,
            download=True
        )
        testset = datasets.CIFAR10(  # (X, y) N=10000
            root="./data",
            train=False,
            download=True
        )

        # Combine train and test sets
        # so that corruption can be applied to both at the same time
        X_train = trainset.data  # PIL Image, numpy array
        y_train = trainset.targets  # int, list
        X_test = testset.data  # PIL Image, numpy array
        y_test = testset.targets  # int, list

        X = np.vstack((X_train, X_test))  # [60000, 32, 32, 3]
        y = np.hstack((y_train, y_test))  # [60000]

        # Data pre-processing
        self.X = torch.tensor(X).permute(0, 3, 1, 2) / 255.0  # scale to [0,1]
        if crop:  # take central part of each image
            margin = (32 - 28) // 2  # 28x28 image
            self.X = self.X[:, :, margin:-margin, margin:-margin]

        # Data normalization
        mean = torch.mean(self.X, dim=(0, 2, 3))
        std = torch.std(self.X, dim=(0, 2, 3))
        # Apply normalization to the data
        for c in range(3):  # For each channel
            self.X[:, c, :, :] = (self.X[:, c, :, :] - mean[c]) / std[c]

        # Target pre-processing -> one-hot encoding
        self.num_classes = len(np.unique(y))  # 10
        self.y = self._one_hot(y)  # [60000, 10]

        # Corruption
        if opts.label_corruption_prob > 0.:
            self._corrupt_labels(opts.label_corruption_prob)
        if opts.data_corruption_type in ("shuffled pixels", "random pixels", "gaussian"):
            self._corrupt_data(opts.data_corruption_type)

    def _one_hot(self, y):
        """ Givent int y, return one-hot encoded y """
        EYE = np.eye(self.num_classes)  # [10, 10]
        y_oh = EYE[y]  # extract rows
        return y_oh

    def _corrupt_labels(self, prob):
        """
        Corrupts labels with a given probability, i.e. dataset fraction

        Args:
            prob (float): probability of corrupting the label
                prob=0. means true labels
                prob>0 and prob<1. means partially corrupted labels
                prob=1. means random labels
        """
        # from one-hot go back to int
        labels = np.argmax(self.y, axis=1)  # [60000]
        # mask for labels to be changed
        mask = np.random.rand(len(labels)) <= prob  # [60000]
        # draw random labels
        rnd_labels = np.random.choice(self.num_classes, mask.sum())  # [prob*60000]
        # change labels
        labels[mask] = rnd_labels
        # convert to one-hot
        self.y = self._one_hot(labels)

    def _corrupt_data(self, corruption):
        """
        Corrupts all data with a given corruption type

        Args:
            corruption_type (str): type of corruption
                "none": no corruption
                "shuffled pixels": random permutation is chosed and applied
                    to all images (train and test)
                "random pixels": different random permutation is applied
                    to each image independently
                "gaussian": gaussian distribution with matching mean
                    and variance to the original dataset is used to
                    generate random pixels for each image
        """
        if corruption == "none":
            return  # No corruption needed

        # Get shape information
        n_samples, n_channels, height, width = self.X.shape
        n_pixels = height * width

        if corruption == "shuffled pixels":
            # Generate a single permutation for all images
            perm = torch.randperm(n_pixels)
            # Apply the same permutation to all images across all channels
            for c in range(n_channels):  # 3 channels
                # Reshape to [n_samples, n_pixels]
                flat_imgs = self.X[:, c, :, :].reshape(n_samples, n_pixels)
                # Apply permutation to each image over all pixels
                flat_imgs = flat_imgs[:, perm]
                # Reshape back to original shape
                self.X[:, c, :, :] = flat_imgs.reshape(n_samples, height, width)

    def __len__(self):
        return self.X.shape[0]  # 60000

    def __getitem__(self, idx):
        """
        returns:
            X: [C, W, H] (tensor)
            y: [num_classes] (one-hot encoded)
        """
        return self.X[idx], self.y[idx]


# class AugmentedCIFAR10(CIFAR10):
#     def __init__(self, opts=None, train=True):
#         super().__init__(opts, train, crop=False)
#         self.augmentation_pipeline = v2.Compose([
#             v2.RandomRotation(degrees=self.opts.rotation_degrees, expand=False),
#             v2.RandomCrop(size=28),  # 28x28 image
#             v2.RandomHorizontalFlip(p=self.opts.horizontal_flip_prob)
#         ])

#     def __getitem__(self, idx):
#         return self.augmentation_pipeline(self.X[idx]), self.y[idx]


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


class MakeDataLoaders():
    def __init__(self, opts, data):
        # generator = torch.Generator().manual_seed(opts.seed)
        train, test = random_split(
            data, lengths=[1-opts.test_size, opts.test_size],
            # generator=generator
        )

        self.train_loader = DataLoader(
            train, batch_size=opts.batch_size, shuffle=True,
            num_workers=opts.num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            test, batch_size=opts.batch_size, shuffle=True,
            num_workers=opts.num_workers, pin_memory=True
        )


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


def main():
    # Hyperparams
    config = dict(
        batch_size=128, num_workers=0, label_corruption_prob=0.,
        data_corruption_type="none", test_size=0.2
    )
    opts = SimpleNamespace(**config)

    # Get Dataset and DataLoader
    data = ModifiedCIFAR10(opts)
    cifar10 = MakeDataLoaders(opts, data)
    train_loader = cifar10.train_loader

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
    config_orig = dict(
        batch_size=128, num_workers=0, label_corruption_prob=0.,
        data_corruption_type="none", test_size=0.2
    )
    opts_orig = SimpleNamespace(**config_orig)
    config_corrup = dict(
        batch_size=128, num_workers=0, label_corruption_prob=0.2,
        data_corruption_type="shuffled pixels", test_size=0.2
    )
    opts_corrup = SimpleNamespace(**config_corrup)

    # Get Dataset and DataLoader
    original_data = ModifiedCIFAR10(opts_orig)
    # original_cifar10 = MakeDataLoaders(opts_orig, original_data)
    # original_train_loader = original_cifar10.train_loader
    corrupted_data = ModifiedCIFAR10(opts_corrup)
    corrupted_cifar10 = MakeDataLoaders(opts_corrup, corrupted_data)
    corrupted_train_loader = corrupted_cifar10.train_loader

    print("Verify corrupted data")
    mean, std = mean_std_channel(corrupted_train_loader)
    print("Corrupted dataset")
    print("Mean:", mean)
    print("Std:", std)

    print("Check labels and images")
    X, y = next(iter(corrupted_train_loader))
    print("Data:", X.shape)
    print("Targets:", y.shape)

    print("Check original labels againts corrupted labels")
    confusion = np.zeros((10, 10), dtype=int)
    for i in range(1000):
        _, y = original_data[i]
        _, y_corrupted = corrupted_data[i]
        confusion[np.argmax(y).item(), np.argmax(y_corrupted).item()] += 1
    print(confusion)
    tot = confusion.sum()
    acc = confusion.trace() / tot
    print(f"Corruption rate: {1-acc:.2f}")

    print("Plot original and corrupted images together along with labels")
    imgs = 8
    X_orig, y_orig = original_data[:imgs]
    X_corrupted, y_corrupted = corrupted_data[:imgs]
    print("Original labels:", [classes[np.argmax(y).item()] for y in y_orig])
    print("Corrupted labels:", [classes[np.argmax(y).item()] for y in y_corrupted])
    grid = make_grid(torch.cat((X_orig, X_corrupted), dim=0), nrow=8)
    imshow(grid)


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        # main()
        main_corruption()
