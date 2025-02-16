from types import SimpleNamespace
import argparse
import yaml
from ipdb import launch_ipdb_on_exception

import torch
import torch.optim as optim

import wandb

from cifar10 import CIFAR10, make_loader
from models import Net
from train import train_loop, test


def main(opts):

    ## Load Dataset and create DataLoader
    # TODO: validation set
    trainset = CIFAR10(opts)
    testset = CIFAR10(opts, train=False)
    train_loader = make_loader(trainset, opts)
    test_loader = make_loader(testset, opts)

    ## Define model
    model = Net(opts.num_filters, opts.mlp_size, opts.num_classes)
    model = model.to(opts.device)

    ## Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay
    )

    ## Training
    train_loop(opts, model, optimizer, train_loader)

    ## Testing
    test_acc = test(opts, model, test_loader)
    print(f"Accuracy: {100.*test_acc:.1f}%")
    wandb.log({"test acc": test_acc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML Configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)

    ## Create object
    opts = SimpleNamespace(**configs)

    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", opts.device)

    with launch_ipdb_on_exception():
        wandb_config = dict(
            project="deep-learning-project",
            config=configs,
            entity="david-inf-team",
            name="simple-net"
        )
        with wandb.init(**wandb_config) as run:
            main(opts)
