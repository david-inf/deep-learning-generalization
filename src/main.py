""" Main script to run a single experiment"""

# logging to Comet stuffs
from comet_ml import start
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

from utils import LOG, update_yaml, set_seeds


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


if __name__ == "__main__":
    import cmd_args
    from ipdb import launch_ipdb_on_exception
    opts = cmd_args.get_args()

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
