
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
        # (X, y) train: N=50000 ; test: N=10000
        trainset = datasets.CIFAR10(root="./data", train=True, download=True)
        testset = datasets.CIFAR10(root="./data", train=False, download=True)

        # Combine train and test sets
        # so that corruption can be applied to both at the same time
        # PIL Image <> numpy array ; int <> list
        X_train, y_train = trainset.data, trainset.targets
        X_test, y_test = testset.data, testset.targets

        X = np.vstack((X_train, X_test))  # [60000, 32, 32, 3]
        y = np.hstack((y_train, y_test))  # [60000]

        # Data pre-processing
        self.X = torch.tensor(X).permute(0, 3, 1, 2) / 255.0  # scale to [0,1]
        if crop:  # take central part of each image
            margin = (32 - 28) // 2  # 28x28 image
            self.X = self.X[:, :, margin:-margin, margin:-margin]

        # Data normalization
        self.mean = torch.mean(self.X, dim=(0, 2, 3))  # original data mean
        self.std = torch.std(self.X, dim=(0, 2, 3))  # original data std
        # Apply normalization to the data
        for c in range(3):  # For each channel
            self.X[:, c, :, :] = (self.X[:, c, :, :] - self.mean[c]) / self.std[c]

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

        corruption (str): type of corruption
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
            perm = torch.randperm(n_pixels)  # [H*W]
            flat_imgs = self.X.flatten(2)  # flattened imgs [B, C, H*W]
            # Apply the same permutation to all images:
            # apply permutation to the last dimension (pixels)
            # go back to the original batch shape
            self.X = flat_imgs[:, :, perm].view(n_samples, n_channels, height, width)

        elif corruption == "random pixels":
            # Generate a different permutation for each image
            perms = torch.stack([torch.randperm(n_pixels) for _ in range(n_samples)])  # [B, H*W]
            idx_perm = perms.unsqueeze(1).expand(-1, n_channels, -1)  # expand (copy) perm along channels and batch [B, C, H*W]
            flat_imgs = self.X.flatten(2)  # flattened imgs [B, C, H*W]
            # Apply the permutation to each image independently:
            # rearranges pixels gather(2, ...) -> [B, C, H*W] following the permutation
            # go back to the original batch shape
            self.X = flat_imgs.gather(2, idx_perm).view(n_samples, n_channels, height, width)
            # gather rearranges in the (flattened) pixels dim according to perm_idx
            # self.X.flatten(2) and idx_perm must have the same shape

        elif corruption == "gaussian":
            # Generate random noise with the same mean and std as the original data
            noise = torch.randn_like(self.X)  # pure gaussian noise [B, C, H, W]
            # For broadcasting operations, we need to add dimensions
            mean = self.mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [C] -> [1, C, 1, 1]
            std = self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [C] -> [1, C, 1, 1]
            # Apply mean and std from original data (we have new data)
            self.X = noise * std + mean  # [B, C, H, W]
            # Scale to [0, 1]
            self.X = torch.clamp(self.X, 0., 1.)

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
    # Original dataset
    config_orig = dict(
        batch_size=128, num_workers=0, label_corruption_prob=0.,
        data_corruption_type="none", test_size=0.2
    )
    original_data = ModifiedCIFAR10(SimpleNamespace(**config_orig))
    # Random labels
    config_labels = dict(
        batch_size=128, num_workers=0, label_corruption_prob=0.5,
        data_corruption_type="none", test_size=0.2
    )
    # opts_labels = SimpleNamespace(**config_labels)
    labels_data = ModifiedCIFAR10(SimpleNamespace(**config_labels))
    # labels_cifar10 = MakeDataLoaders(opts_labels, labels_data)
    # labels_loader = labels_cifar10.train_loader
    # Shuffled pixels
    config_shuff = dict(
        batch_size=128, num_workers=0, label_corruption_prob=0.,
        data_corruption_type="shuffled pixels", test_size=0.2
    )
    # opts_shuff = SimpleNamespace(**config_shuff)
    shuff_data = ModifiedCIFAR10(SimpleNamespace(**config_shuff))
    # shuff_cifar10 = MakeDataLoaders(opts_shuff, ModifiedCIFAR10(opts_shuff))
    # shuff_loader = shuff_cifar10.train_loader
    # Random pixels
    config_rand = dict(
        batch_size=128, num_workers=0, label_corruption_prob=0.,
        data_corruption_type="random pixels", test_size=0.2
    )
    # opts_rand = SimpleNamespace(**config_rand)
    rand_data = ModifiedCIFAR10(SimpleNamespace(**config_rand))
    # rand_cifar10 = MakeDataLoaders(opts_rand, ModifiedCIFAR10(opts_rand))
    # rand_loader = rand_cifar10.train_loader
    # Gaussian pixels
    config_gauss = dict(
        batch_size=128, num_workers=0, label_corruption_prob=0.,
        data_corruption_type="gaussian", test_size=0.2
    )
    # opts_gauss = SimpleNamespace(**config_gauss)
    gauss_data = ModifiedCIFAR10(SimpleNamespace(**config_gauss))
    # gauss_cifar10 = MakeDataLoaders(opts_gauss, ModifiedCIFAR10(opts_gauss))
    # gauss_loader = gauss_cifar10.train_loader

    ## ***** Labels corruption *****
    # print("Check labels and images")
    # X, y = next(iter(labels_loader))
    # print("Data:", X.shape)
    # print("Targets:", y.shape)

    print("Check original labels againts corrupted labels")
    confusion = np.zeros((10, 10), dtype=int)
    for i in range(1000):
        _, y = original_data[i]
        _, y_corrupted = labels_data[i]
        confusion[np.argmax(y).item(), np.argmax(y_corrupted).item()] += 1
    print(confusion)
    tot = confusion.sum()
    acc = confusion.trace() / tot
    print(f"Corruption rate: {1-acc:.2f}")

    import matplotlib.pyplot as plt
    import os
    dir = "plots/figures"
    os.makedirs(dir, exist_ok=True)
    ## ***** Shuffled pixels corruption *****
    print("Shuffled pixels")
    imgs = 8
    X_orig, _ = original_data[:imgs]
    X_corrupted, _ = shuff_data[:imgs]
    grid = make_grid(torch.cat((X_orig, X_corrupted), dim=0), nrow=8)
    imshow(grid, os.path.join(dir, "shuffled_pixels.png"))

    ## ***** Random pixels corruption *****
    print("Random pixels")
    X_corrupted, _ = rand_data[:imgs]
    grid = make_grid(torch.cat((X_orig, X_corrupted), dim=0), nrow=8)
    imshow(grid, os.path.join(dir, "random_pixels.png"))

    ## ***** Gaussian pixels corruption *****
    print("Gaussian pixels")
    X_corrupted, _ = gauss_data[:imgs]
    grid = make_grid(torch.cat((X_orig, X_corrupted), dim=0), nrow=8)
    imshow(grid, os.path.join(dir, "gaussian_pixels.png"))


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        # main()
        main_corruption()
