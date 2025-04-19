""" Arguments for the main programs """

import os
from types import SimpleNamespace
import argparse
import yaml
import torch
from utils import LOG, update_yaml


parser = argparse.ArgumentParser(
    description="Run an experiment and log to comet_ml")
parser.add_argument("--config", help="YAML configuration file")
# parser.add_argument("--config", default="src/experiments/MLP1/MLP1_0.0_none.yaml")  # debug
parser.add_argument("--epochs", help="Update epochs number")
parser.add_argument("--view", action="store_true",
                    help="Visualize architecture, no training")


def parse_args():
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)  # return dict
    opts = SimpleNamespace(**configs)

    opts.config = args.config  # add config file path for updates
    opts.visualize = args.view  # inspect model

    # check number of epochs
    if args.epochs is not None:
        new_epochs = int(args.epochs)
        if new_epochs > opts.num_epochs:
            old_epochs = opts.num_epochs
            update_yaml(opts, "num_epochs", new_epochs)
            LOG.info(f"Updated new_epochs={opts.num_epochs} from old_epochs={old_epochs}")
    else:
        LOG.info(f"Training until num_epochs={opts.num_epochs}")

    if opts.checkpoint_every is None:
        opts.checkpoint_every = opts.num_epochs
        LOG.info("Checkpoint at the end of training")
    else:
        LOG.info(f"Checkpoint checkpoint_every={opts.checkpoint_every}")

    return opts
