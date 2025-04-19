""" Main script to run Figure 1 experiments """

# logging to Comet stuffs
from comet_ml import start
# from comet_ml.integration.pytorch import log_model

# pytorch stuffs
import random
import numpy as np
import torch

# my stuffs
from cifar10 import CorruptedCIFAR10, MakeDataLoaders
from models.mlp import MLP1, MLP3
from models.inception import InceptionSmall
from models.alexnet import AlexNetSmall
from models.wideresnet import WideResNet
from train import train_loop

from utils import LOG, update_yaml, set_seeds, visualize


def get_loaders(opts):
    from torch.utils.data import Subset
    # 1) Load full CIFAR10 dataset with needed corruptions
    dataset = CorruptedCIFAR10(opts)
    # 2) Train-Test split
    N = len(dataset)
    indices = list(range(N))
    trainset = Subset(dataset, indices[:50000])
    testset = Subset(dataset, indices[50000:])
    # 3) Loaders
    loaders = MakeDataLoaders(opts, trainset, testset)
    return loaders.train_loader, loaders.test_loader


def get_model(opts):
    if opts.model_name == "MLP1":
        model = MLP1()  # 1 hidden unit
    elif opts.model_name == "MLP3":
        model = MLP3()  # 3 hidden units
    elif opts.model_name in ("AlexNet", "AlexNetSmall"):
        model = AlexNetSmall()
    elif opts.model_name in ("Inception", "InceptionSmall"):
        model = InceptionSmall()
    elif opts.model_name in ("WideResNet", "WRN"):
        # (3*2)*num_blocks inner layers
        model = WideResNet(num_blocks=2, widen_factor=3)
    else:
        raise ValueError(f"Unknown model {opts.model_name}")
    model = model.to(opts.device)
    return model


def main(opts, experiment):
    # opts : SimpleNamespace
    # experiment : comet_ml.Experiment
    set_seeds(opts.seed)
    # Loaders
    train_loader, test_loader = get_loaders(opts)
    # Get model
    model = get_model(opts)
    # Training
    update_yaml(opts, "figure1", True)
    with experiment.train():
        LOG.info(f"Running experiment_name={opts.experiment_name}")
        train_loop(opts, model, train_loader, test_loader,
                   experiment, opts.resume_checkpoint)


def view_model(opts):
    # Get model
    model = get_model(opts)
    # Random data for images
    input_data = torch.randn(128, 3, 28, 28).to(opts.device)
    # Visualize model
    visualize(model, f"{opts.model_name}", input_data)


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception
    opts = parse_args()

    with launch_ipdb_on_exception():
        if opts.visualize:
            view_model(opts)  # no training

        else:
            # try resuming an experiment if experiment_key is provided
            # otherwise start a new experiment
            if not opts.experiment_key:
                experiment = start(project_name=opts.comet_project)
                experiment.set_name(opts.experiment_name)
                # Update with experiment key for resuming
                update_yaml(opts, "experiment_key", experiment.get_key())
                LOG.info("Added experiment key for resuming")
            else:
                # Resume using provided experiment key and checkpoint
                # the key is set above
                # the checkpoint is set with save_checkpoint in train_loop()
                experiment = start(project_name=opts.comet_project,
                                   mode="get", experiment_key=opts.experiment_key,)
            main(opts, experiment)
            experiment.log_parameters(vars(opts))
            experiment.end()
