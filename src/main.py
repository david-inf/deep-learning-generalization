""" Main script to run a single experiment"""

import os
from types import SimpleNamespace
import argparse
import yaml
from ipdb import launch_ipdb_on_exception

# logging to Comet stuffs
from comet_ml import start, ExperimentConfig
from comet_ml.integration.pytorch import log_model

# pytorch stuffs
import torch
import torch.optim as optim

# my stuffs
from cifar10 import ModifiedCIFAR10, MakeDataLoaders
from models import Net, MLP
from train import train_loop, test

from utils import LOG


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
    # elif opts.model_name == "AlexNet":
    #     model = AlexNet()
    # elif opts.model_name == "Inception":
    #     model = Inception()
    # a recent implementation uses ResNet

    model = model.to(opts.device)
    return model


def log_samples(experiment, data_loader, num_samples=4):
    """ Log samples to comet_ml """
    images, targets = next(iter(data_loader))

    # log samples
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

    ## Load Dataset and create DataLoader
    train_loader, test_loader = get_loaders(opts)
    # log few samples to comet_ml
    log_samples(experiment, train_loader)

    ## Define model and optimizer
    model = get_model(opts)
    optimizer = optim.SGD(
        model.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay
    )

    ## Training & checkpointing
    ckp_dir = os.path.join("checkpoints", opts.model_name)
    os.makedirs(ckp_dir, exist_ok=True)  # output dir not tracked by git
    opts.checkpoint_dir = ckp_dir  # for saving and loading ckps

    with experiment.train():
        LOG.info(f"Running {opts.experiment_name}")
        train_loop(
            opts, model, optimizer, train_loader, test_loader,
            experiment, opts.resume_checkpoint
        )

    ## Testing
    with experiment.test():
        test_acc = test(
            opts, model, test_loader
        )
        LOG.info(f"Final test accuracy: {100.*test_acc:.1f}%")
        experiment.log_metrics({
            "acc": test_acc,
            "error": 1. - test_acc,
            # "time to overfit": _zero_loss_time,
            "label corruption": opts.label_corruption_prob
        })

    # TODO: log to comet_ml
    # log_model(experiment, model, opts.model_name)


if __name__ == "__main__":
    # This code runs a single experiment
    parser = argparse.ArgumentParser(description="Main script for running a single experiment and logging to comet_ml")
    # A default configuration is set, but one may provide a different one
    # A different configuration is provided each time when running multiple experiments
    parser.add_argument("--config", default="config.yaml", help="YAML Configuration file")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs, increase when resuming")
    parser.add_argument("--ckping", type=int, default=None, help="Specify checkpointing frequency with epochs")
    # parser.add_argument("--experiment_key", default=None, help="Resume an experiment")
    # parser.add_argument("--resume_from", default="last", help="Resume from checkpoint (last or path)")

    args = parser.parse_args()  # arguments are attributes of args
    with open(args.config, "r") as f:  # args.config is the configuration file
        # more arguments but in a different object
        configs = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    # Create object for given configuration
    opts = SimpleNamespace(**configs)

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

    # Resume from checkpoint
    # TODO: forse dovrei fare il dumping dello yaml quando aggiorno le epoche
    # TODO: così da poter recuperare il numero precedente senza ricorrere al loading
    # if args.resume_from == "last":
    #     # resume from the last checkpoint
    #     path = f"checkpoints/{opts.model_name}/"
    #     fname = f"e_"
    #     opts.resume_checkpoint = f"_prob_{opts.label_corruption_prob};type_{opts.data_corruption_type}.pt"
    #     print(f"Resuming from last checkpoint: {opts.resume_checkpoint}")

    # Device
    opts.device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info(f"Device: {opts.device}")

    with launch_ipdb_on_exception():
        if not opts.experiment_key:
            if opts.experiment_name:
                exp_name = opts.experiment_name
            else:
                exp_name = f"{opts.model_name}_{opts.label_corruption_prob}_{opts.data_corruption_type}",
            experiment = start(
                project_name=opts.comet_project,
                experiment_config=ExperimentConfig(
                    name=exp_name
                )
            )
        else:
            experiment = start(
                project_name=opts.comet_project,
                mode="get",
                experiment_key=opts.experiment_key,
            )
        experiment.log_parameters(
            configs
        )
        main(opts, experiment)
        experiment.end()
        # try resuming an experiment if experiment_key is provided
        # otherwise start a new experiment
