""" Main script to run Figure 2 experiments """

# logging to Comet stuffs
from comet_ml import start
# from comet_ml.integration.pytorch import log_model

# pytorch stuffs
import random
import numpy as np
import torch

from cifar10 import MyCIFAR10, MakeDataLoaders
from models.inception import InceptionSmall
from train import train_loop
from utils import LOG, update_yaml, set_seeds


def get_loaders(opts):
    # 1) Load train and test sets
    trainset = MyCIFAR10(opts, train=True)
    testset = MyCIFAR10(opts, train=False)
    # 2) Loaders
    loaders = MakeDataLoaders(opts, trainset, testset)
    return loaders.train_loader, loaders.test_loader


def main(opts, experiment):
    set_seeds(opts.seed)
    # Loaders
    train_loader, test_loader = get_loaders(opts)
    # Model
    model = InceptionSmall(bn=opts.bn)
    model = model.to(opts.device)
    # Training
    update_yaml(opts, "figure1", False)
    with experiment.train():
        LOG.info(f"Running {opts.experiment_name}")
        train_loop(opts, model, train_loader, test_loader,
                      experiment, opts.resume_checkpoint)


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception
    opts = parse_args()

    with launch_ipdb_on_exception():
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
