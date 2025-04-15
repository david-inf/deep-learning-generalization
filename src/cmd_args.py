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
parser.add_argument("--epochs", help="Update epochs number")


def parse_args():
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)  # return dict
    opts = SimpleNamespace(**configs)
    opts.config = args.config  # add config file path for updates

    # check number of epochs
    if args.epochs is not None:
        new_epochs = int(args.epochs)
        if new_epochs > opts.num_epochs:
            old_epochs = opts.num_epochs
            update_yaml(opts, "num_epochs", new_epochs)
            LOG.info(f"Updated epochs from {old_epochs} to {opts.num_epochs}")
    else:
        LOG.info(f"Training until {opts.num_epochs} epochs")

    if opts.checkpoint_every is None:
        opts.checkpoint_every = opts.num_epochs
        LOG.info("Checkpoint at the end of training")
    else:
        LOG.info(f"Checkpoint every {opts.checkpoint_every}")

    return opts
