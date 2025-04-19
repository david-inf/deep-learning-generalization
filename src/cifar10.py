
import random
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
from ipdb import launch_ipdb_on_exception

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from utils import imshow, set_seeds

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CorruptedCIFAR10(Dataset):
    """ CIFAR10 dataset with corrupted labels or data """

    def __init__(self, opts, crop=True):
        self.opts = opts
        self.num_classes = 10

        # Get full CIFAR10 dataset, first separated
        # (X, y) train: N=50000 ; test: N=10000
        trainset = datasets.CIFAR10(root="../../data", train=True, download=True)
        testset = datasets.CIFAR10(root="../../data", train=False, download=True)

        # Combine train and test sets
        # so that corruption can be applied to both at the same time
        # PIL Image <> numpy array ; int <> list
        X_train, y_train = trainset.data, trainset.targets
        X_test, y_test = testset.data, testset.targets

        X = np.vstack((X_train, X_test))  # [60000, 32, 32, 3]
        y = np.hstack((y_train, y_test))  # [60000]

        # Prepare data
        self.X = torch.tensor(X).permute(0, 3, 1, 2) / 255.0  # scale to [0,1]
        if crop:  # take central part of each image
            margin = (32 - 28) // 2  # 28x28 image
            self.X = self.X[:, :, margin:-margin, margin:-margin]
        self.y = torch.tensor(y)

        # Normalize data with given mean and std
        self.mean = torch.tensor([0.489255, 0.475775, 0.439889])
        self.std = torch.tensor([0.243047, 0.239315, 0.255997])
        self.X = transforms.Normalize(self.mean, self.std)(self.X)

        set_seeds(opts.seed)  # set seed for reproducibility
        # Corruption
        assert opts.label_corruption_prob >= 0. and opts.label_corruption_prob <= 1.
        if opts.label_corruption_prob > 0.:
            self._corrupt_labels(opts.label_corruption_prob)

        assert opts.data_corruption_type in (
            "none", "shuff_pix", "rand_pix", "gauss_pix")
        if opts.data_corruption_type in ("shuff_pix", "rand_pix", "gauss_pix"):
            self._corrupt_data(opts.data_corruption_type)

    def _corrupt_labels(self, prob):
        """
        Corrupts labels with a given probability, i.e. dataset fraction

        Args:
            prob (float): probability of corrupting the label
                prob=0. means true labels
                prob>0 and prob<1. means partially corrupted labels
                prob=1. means random labels
        """
        # Set seed
        random.seed(self.opts.seed)
        np.random.seed(self.opts.seed)
        # labels to numpy
        labels = self.y.numpy()
        # mask for labels to be changed
        mask = np.random.rand(len(labels)) <= prob  # [60000]
        # draw random labels
        rnd_labels = np.random.choice(
            self.num_classes, mask.sum())  # [prob*60000]
        # change labels
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.y = torch.tensor(labels)

    def _corrupt_data(self, corruption):
        """
        Corrupts all data with a given corruption type

        corruption (str): type of corruption
            "none": no corruption
            "shuff_pix": (shuffled pixels) random permutation is chosed and applied
                to all images (train and test)
            "rand_pix": (random pixels) different random permutation is applied
                to each image independently
            "gauss_pix": (gaussian pixels) gaussian distribution with matching mean
                and variance to the original dataset is used to
                generate random pixels for each image
        """
        # Set seed
        random.seed(self.opts.seed)
        np.random.seed(self.opts.seed)
        torch.manual_seed(self.opts.seed)
        # Get shape information
        n_samples, n_channels, height, width = self.X.shape
        n_pixels = height * width

        if corruption == "shuff_pix":
            # Generate a single permutation for all images
            perm = torch.randperm(n_pixels)  # [H*W]
            flat_imgs = self.X.flatten(2)  # flattened imgs [B, C, H*W]
            # Apply the same permutation to all images:
            # apply permutation to the last dimension (pixels)
            # go back to the original batch shape
            self.X = flat_imgs[:, :, perm].view(
                n_samples, n_channels, height, width)

        elif corruption == "rand_pix":
            # Generate a different permutation for each image
            perms = torch.stack([torch.randperm(n_pixels)
                                for _ in range(n_samples)])  # [B, H*W]
            # expand (copy) perm along channels and batch [B, C, H*W]
            idx_perm = perms.unsqueeze(1).expand(-1, n_channels, -1)
            flat_imgs = self.X.flatten(2)  # flattened imgs [B, C, H*W]
            # Apply the permutation to each image independently:
            # rearranges pixels gather(2, ...) -> [B, C, H*W] following the permutation
            # go back to the original batch shape
            self.X = flat_imgs.gather(2, idx_perm).view(
                n_samples, n_channels, height, width)
            # gather rearranges in the (flattened) pixels dim according to perm_idx
            # self.X.flatten(2) and idx_perm must have the same shape

        elif corruption == "gauss_pix":
            # Generate random noise with the same mean and std as the original data
            # pure gaussian noise [B, C, H, W]
            noise = torch.randn_like(self.X)
            # For broadcasting operations, we need to add dimensions
            mean = self.mean.unsqueeze(
                0).unsqueeze(-1).unsqueeze(-1)  # [C] -> [1, C, 1, 1]
            std = self.std.unsqueeze(
                0).unsqueeze(-1).unsqueeze(-1)  # [C] -> [1, C, 1, 1]
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
            y: [] (tensor)
        """
        return self.X[idx], self.y[idx]


class MyCIFAR10(Dataset):
    """ Wrapper for CIFAR10 Dataset class """

    def __init__(self, opts, crop=True, train=True):
        self.opts = opts
        self.num_classes = 10

        dataset = datasets.CIFAR10(
            root="../../data", train=train, download=True)
        X, y = dataset.data, dataset.targets  # ndarray, list

        # Prepare data
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2) / 255.  # [0,1]
        if crop:
            margin = (32 - 28) // 2
            self.X = self.X[:, :, margin:-margin, margin:-margin]
        self.y = torch.tensor(y)

        # Normalize data with given mean and std
        self.mean = torch.tensor([0.489255, 0.475775, 0.439889])
        self.std = torch.tensor([0.243047, 0.239315, 0.255997])
        self.X = transforms.Normalize(self.mean, self.std)(self.X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        returns:
            X: [C, W, H] (tensor)
            y: [] (tensor)
        """
        image, target = self.X[idx], self.y[idx]
        return image, target


# class MyAugmentedCIFAR10(MyCIFAR10):
#     def __init__(self, opts):
#         super().__init__(opts, crop=False, train=True)
#         self.augmentation_pipeline = v2.Compose([
#             v2.RandomHorizontalFlip(p=0.5),
#             v2.RandomCrop(size=32, padding=4),
#         ])

#     def __getitem__(self, idx):
#         return self.augmentation_pipeline(self.X[idx]), self.y[idx]


class MakeDataLoaders:
    def __init__(self, opts, trainset, testset):
        set_seeds(opts.seed)
        generator = torch.Generator().manual_seed(opts.seed)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        batch_size = opts.batch_size
        num_workers = opts.num_workers
        self.train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            generator=generator, worker_init_fn=seed_worker
        )
        self.test_loader = DataLoader(
            testset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=True,
            generator=generator, worker_init_fn=seed_worker
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
    config = dict(batch_size=128, num_workers=0,
                  label_corruption_prob=0., seed=42,
                  data_corruption_type="none")
    opts = SimpleNamespace(**config)

    # Get Dataset and DataLoader
    trainset = MyCIFAR10(opts)
    testset = MyCIFAR10(opts, train=False)
    print(f"Train size: X={trainset.X.shape}, y={trainset.y.shape}")
    print(f"test size: X={testset.X.shape}, y={testset.y.shape}")
    cifar10 = MakeDataLoaders(opts, trainset, testset)
    train_loader = cifar10.train_loader

    # Verify data
    mean, std = mean_std_channel(train_loader)
    print("Check current statistics")
    print("Mean:", mean, "Std:", std)

    # Check labels and images
    X, y = next(iter(train_loader))
    print("Data shape:", f"X={X.shape}", f"y={y.shape}")

    # Print first batch
    import os
    os.makedirs("plots", exist_ok=True)
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    grid = make_grid(X, nrow=8, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("plots/figures/cifar10.png")


def main_corruption():
    # Original dataset
    imgs = 8
    config = dict(batch_size=imgs, num_workers=0, label_corruption_prob=0.,
                  data_corruption_type="none", seed=42)
    opts = SimpleNamespace(**config)
    original_data = CorruptedCIFAR10(opts)
    original = MakeDataLoaders(opts, original_data, original_data)
    original_loader = original.train_loader

    # Random labels
    config = dict(batch_size=imgs, num_workers=0, label_corruption_prob=0.5,
                  data_corruption_type="none", seed=42)
    opts = SimpleNamespace(**config)
    labels_data = CorruptedCIFAR10(opts)

    # Shuffled pixels
    config = dict(batch_size=imgs, num_workers=0, label_corruption_prob=0.,
                  data_corruption_type="shuff_pix", seed=42)
    opts = SimpleNamespace(**config)
    shuff_data = CorruptedCIFAR10(opts)
    shuff_cifar10 = MakeDataLoaders(opts, shuff_data, shuff_data)
    shuff_loader = shuff_cifar10.train_loader

    # Random pixels
    config = dict(batch_size=imgs, num_workers=0, label_corruption_prob=0.,
                       data_corruption_type="rand_pix", seed=42)
    opts = SimpleNamespace(**config)
    rand_data = CorruptedCIFAR10(opts)
    rand_cifar10 = MakeDataLoaders(opts, rand_data, rand_data)
    rand_loader = rand_cifar10.train_loader

    # Gaussian pixels
    config = dict(batch_size=imgs, num_workers=0, label_corruption_prob=0.,
                        data_corruption_type="gauss_pix", seed=42)
    opts = SimpleNamespace(**config)
    gauss_data = CorruptedCIFAR10(opts)
    gauss_cifar10 = MakeDataLoaders(opts, gauss_data, gauss_data)
    gauss_loader = gauss_cifar10.train_loader

    print("Check original labels againts corrupted labels")
    confusion = np.zeros((10, 10), dtype=int)
    for i in range(5000):
        _, y = original_data[i]
        _, y_corrupted = labels_data[i]
        confusion[y.item(), y_corrupted.item()] += 1
    tot = confusion.sum()
    print(confusion / tot)
    acc = confusion.trace() / tot
    print(
        f"Given prob: 0.5 <> Observed prob: {1-acc:.2f}")

    import os
    dir = "plots/figures"
    os.makedirs(dir, exist_ok=True)
    # ***** Shuffled pixels corruption *****
    print("Shuffled pixels")
    X_orig, _ = next(iter(original_loader))
    X_corrupted, _ = next(iter(shuff_loader))
    grid = make_grid(torch.cat((X_orig, X_corrupted), dim=0), nrow=8)
    imshow(grid, os.path.join(dir, "shuffled_pixels.png"))

    # ***** Random pixels corruption *****
    print("Random pixels")
    X_corrupted, _ = next(iter(rand_loader))
    grid = make_grid(torch.cat((X_orig, X_corrupted), dim=0), nrow=8)
    imshow(grid, os.path.join(dir, "random_pixels.png"))

    # ***** Gaussian pixels corruption *****
    print("Gaussian pixels")
    X_corrupted, _ = next(iter(gauss_loader))
    grid = make_grid(torch.cat((X_orig, X_corrupted), dim=0), nrow=8)
    imshow(grid, os.path.join(dir, "gaussian_pixels.png"))


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        # main()
        main_corruption()
