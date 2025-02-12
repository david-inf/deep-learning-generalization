import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split


class CIFAR10(Dataset):
    def __init__(self, opts=None, train=True):
        self.opts = opts
        self.dataset = datasets.CIFAR10(  # (X, y)
            root="./data", train=train, download=True
        )
        # self.preprocess = v2.Compose([
        #     # normalize
        #     # transform to tensor
        # ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # return (image, label)
        image, label = self.dataset[idx]

        return image, label

# TODO: augmentation dataset


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


data = CIFAR10()
print(type(data[0][0]))
