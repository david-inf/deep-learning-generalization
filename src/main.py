
import os
from types import SimpleNamespace
import argparse
import yaml
from ipdb import launch_ipdb_on_exception

import torch
import torch.optim as optim

import wandb

from cifar10 import CIFAR10, CorruptedCIFAR10, make_loader
from models import Net, MLP
from train import train_loop, test, save_checkpoint


def get_loaders(opts):

    trainset = CorruptedCIFAR10(opts)
    testset = CorruptedCIFAR10(opts, train=False)
    train_loader = make_loader(trainset, opts)
    test_loader = make_loader(testset, opts)

    return train_loader, test_loader


def get_model(opts):

    if opts.model_name == "Net":
        model = Net(16, 128, 10)
    elif opts.model_name == "MLP1":
        model = MLP(1)
    elif opts.model_name == "MLP3":
        model = MLP(3)  # 3 hidden units
    # elif opts.model_name == "AlexNet":
    #     model = AlexNet()
    # elif opts.model_name == "Inception":
    #     model = Inception()
    # a recent implementation uses ResNet

    model = model.to(opts.device)

    return model


def main(opts):

    ## Load Dataset and create DataLoader
    train_loader, test_loader = get_loaders(opts)

    ## Define model
    model = get_model(opts)

    ## Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay
    )

    ## Training
    os.makedirs(opts.checkpoint_dir, exist_ok=True)  # output dir not tracked by git
    train_loop(opts, model, optimizer, train_loader, test_loader)

    ## Testing
    test_acc = test(opts, model, test_loader)
    print(f"Final test accuracy: {100.*test_acc:.1f}%")
    # wandb.log({
    #     "test acc": test_acc,
    #     "test error": 1. - test_acc,
    #     # "time to overfit":  # time to reach zero-loss
    #     # "label corruption": opts.label_corruption_prob
    # })


if __name__ == "__main__":
    # This code runs a single experiment
    parser = argparse.ArgumentParser(description="Run experiment with given configuration")
    # A default configuration is set, but one may provide a different one
    # A different configuration is provided each time when running multiple experiments
    parser.add_argument("--config", default="config.yaml", help="YAML Configuration file")

    args = parser.parse_args()  # arguments are attributes of args
    with open(args.config, "r") as f:  # args.config is the configuration file
        # more arguments but in a different object
        configs = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    # Create object for given configuration
    opts = SimpleNamespace(**configs)
    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", opts.device)

    with launch_ipdb_on_exception():
        wandb_config = dict(
            entity="david-inf-team",
            project="deep-learning-project",
            name=opts.run_name,  # important for good tracking
            # tags=[opts.model_name],
            config=configs,
            # group=opts.experiment_name,  # TODO: group naming for organizing experiments
        )
        with wandb.init(**wandb_config) as run:
            main(opts)

    # TODO: resume training from a checkpoint and continue logging to the same wandb run
