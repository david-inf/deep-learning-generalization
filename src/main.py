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


def main(opts, experiment):
    # opts : SimpleNamespace
    # experiment : comet_ml.Experiment

    ## Load Dataset and create DataLoader
    train_loader, test_loader = get_loaders(opts)

    ## Define model and optimizer
    model = get_model(opts)
    optimizer = optim.SGD(
        model.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay
    )

    ## TODO: Resume training from checkpoint

    ## Training & checkpointing
    os.makedirs(opts.checkpoint_dir, exist_ok=True)  # output dir not tracked by git
    with experiment.train():
        train_loop(
            opts, model, optimizer, train_loader, test_loader,
            experiment, opts.resume_checkpoint
        )

    ## Testing
    with experiment.test():
        test_acc = test(
            opts, model, test_loader
        )
        print(f"Final test accuracy: {100.*test_acc:.1f}%")
        # wandb.log({
        #     "test acc": test_acc,
        #     "test error": 1. - test_acc,
        #     # "time to overfit":  # time to reach zero-loss
        #     # "label corruption": opts.label_corruption_prob
        # })
        # TODO: log to comet_ml
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
    parser = argparse.ArgumentParser(description="Runresume=opts.resume_checkpoint experiment with given configuration")
    # A default configuration is set, but one may provide a different one
    # A different configuration is provided each time when running multiple experiments
    parser.add_argument("--config", default="config.yaml", help="YAML Configuration file")

    args = parser.parse_args()  # arguments are attributes of args
    with open(args.config, "r") as f:  # args.config is the configuration file
        # more arguments but in a different object
        configs = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    # Device
    configs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", configs["device"])

    # Create object for given configuration
    opts = SimpleNamespace(**configs)

    with launch_ipdb_on_exception():
        if not opts.experiment_key:
            if opts.experiment_name:
                exp_name = opts.experiment_name
            else:
                exp_name = f"{opts.model_name}_prob_{opts.label_corruption_prob};type_{opts.data_corruption_type}",
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
                # experiment_config=ExperimentConfig(
                #     name=opts.experiment_name
                # )
            )
        experiment.log_parameters(
            configs
        )
        main(opts, experiment)
        experiment.end()
        # try resuming an experiment if experiment_key is provided
        # otherwise start a new experiment
