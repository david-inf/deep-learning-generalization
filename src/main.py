""" Main script to run a single experiment"""

import os
from types import SimpleNamespace
import argparse
import yaml
from ipdb import launch_ipdb_on_exception

# logging to Comet stuffs
from comet_ml import start, ExperimentConfig
# from comet_ml.integration.pytorch import log_model

# pytorch stuffs
import random
import numpy as np
import torch

# my stuffs
from cifar10 import ModifiedCIFAR10, MakeDataLoaders
from models.simple_mlp import Net, MLP
from models.inception import InceptionSmall
from models.alexnet import AlexNetSmall
from train import train_loop, test

from utils import LOG, update_yaml


def set_seeds(seed):
    """Set seeds for all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loaders(opts):
    data = ModifiedCIFAR10(opts)  # full cifar10 dataset
    cifar10 = MakeDataLoaders(opts, data)  # class train and test loader
    train_loader = cifar10.train_loader  # split according to opts.test_size
    test_loader = cifar10.test_loader
    return train_loader, test_loader


def get_model(opts):
    if opts.model_name == "Net":
        model = Net(16, 128, 10)
    elif opts.model_name == "MLP1":
        model = MLP(1)
    elif opts.model_name == "MLP3":
        model = MLP(3)  # 3 hidden units
    elif opts.model_name in ("AlexNet", "AlexNetSmall"):
        model = AlexNetSmall()
    elif opts.model_name in ("Inception", "InceptionSmall"):
        model = InceptionSmall()
    # a recent implementation uses ResNet
    model = model.to(opts.device)
    return model


def log_samples(experiment, data_loader, num_samples=4):
    """ Log samples to comet_ml """
    images, targets = next(iter(data_loader))

    # log samples
    # TODO: fix
    for i in range(num_samples):
        experiment.log_image(
            images[i],
            name=targets[i],
            # image_channels="first"
        )
        # experiment.log_text(
        #     f"true: {data_loader.dataset.classes[targets[j]]}, pred: {data_loader.dataset.classes[preds[j]]}",
        #     name=f"sample_{i}_{j}_label"
        # )


def update_opts(opts, args):
    # update yaml file with updated and new attributes from opts
    opts.config = args.config  # keep the yaml file name

    # Device
    opts.device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info(f"Device: {opts.device}")

    # Update epochs
    if args.epochs > opts.num_epochs:
        prev = opts.num_epochs
        opts.num_epochs = args.epochs
        LOG.info(f"Updated number of epochs to {opts.num_epochs} from {prev}")

    # Model checkpointing
    if args.ckping:
        opts.checkpoint_every = args.ckping
        LOG.info(f"Checkpointing every {opts.checkpoint_every} epochs")
    else:
        opts.checkpoint_every = opts.num_epochs
        LOG.info(f"Checkpointing at the end of training")
    ckp_dir = os.path.join("checkpoints", opts.model_name)  # checkpoints directory
    os.makedirs(ckp_dir, exist_ok=True)  # output dir not tracked by git
    opts.checkpoint_dir = ckp_dir  # for saving and loading ckps

    # Resume from checkpoint
    # TODO: forse dovrei fare il dumping dello yaml quando aggiorno le epoche
    # TODO: così da poter recuperare il numero precedente senza ricorrere al loading
    # if args.resume_from == "last":
    #     # resume from the last checkpoint
    #     path = f"checkpoints/{opts.model_name}/"
    #     fname = f"e_"
    #     opts.resume_checkpoint = f"_prob_{opts.label_corruption_prob};type_{opts.data_corruption_type}.pt"
    #     print(f"Resuming from last checkpoint: {opts.resume_checkpoint}")

    # Update yaml file
    with open(opts.config, "w") as f:
        # dump the updated opts to the yaml file
        yaml.dump(opts.__dict__, f)


def main(opts, experiment):
    # opts : SimpleNamespace
    # experiment : comet_ml.Experiment
    set_seeds(opts.seed)

    # Load Dataset and create DataLoader
    train_loader, test_loader = get_loaders(opts)
    # log few samples to comet_ml
    # log_samples(experiment, train_loader)

    # Get model
    model = get_model(opts)

    # Training
    with experiment.train():
        LOG.info(f"Running {opts.experiment_name}")
        train_loop(opts, model, train_loader, test_loader,
                   experiment, opts.resume_checkpoint)

    # Testing
    with experiment.test():
        test_acc = test(opts, model, test_loader)
        LOG.info(f"Final test accuracy: {100.*test_acc:.1f}%")
        experiment.log_metrics({"acc": test_acc, "error": 1. - test_acc})

    # TODO: log to comet_ml
    # log_model(experiment, model, opts.model_name)


if __name__ == "__main__":
    # This code runs a single experiment
    parser = argparse.ArgumentParser(
        description="Main script for running a single experiment and logging to comet_ml")
    # A default configuration is set, but one may provide a different one
    # A different configuration is provided each time when running multiple experiments
    parser.add_argument("--config", default="config.yaml",
                        help="YAML Configuration file")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs, increase when resuming")
    parser.add_argument("--ckping", type=int, default=None,
                        help="Specify checkpointing frequency with epochs")
    # parser.add_argument("--resume_from", default="last", help="Resume from checkpoint (last or path)")

    args = parser.parse_args()  # arguments are attributes of args
    with open(args.config, "r") as f:  # args.config is the configuration file
        # more arguments but in a different object
        configs = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    # Create object for given configuration
    opts = SimpleNamespace(**configs)
    # TODO: maybe one can do better by joining somehow opts and args
    update_opts(opts, args)  # update opts with other given arguments

    with launch_ipdb_on_exception():
        # try resuming an experiment if experiment_key is provided
        # otherwise start a new experiment
        if not opts.experiment_key:
            # exp_name = f"{opts.model_name}_{opts.label_corruption_prob}_{opts.data_corruption_type}",
            # Create Experiment object
            experiment = start(
                project_name=opts.comet_project,
                experiment_config=ExperimentConfig(
                    name=opts.experiment_name
                )
            )
            # Update with experiment key for resuming
            update_yaml(opts, "experiment_key", experiment.get_key())
        else:
            # Resume using provided experiment key and checkpoint
            # the key is set above
            # the checkpoint is set with save_checkpoint in train_loop()
            experiment = start(
                project_name=opts.comet_project,
                mode="get",
                experiment_key=opts.experiment_key,
            )
        main(opts, experiment)
        experiment.log_parameters(vars(opts))
        experiment.end()
