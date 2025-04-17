
import os
import logging
from rich.logging import RichHandler

import matplotlib.pyplot as plt
import numpy as np


def N(x):
    # detach from computational graph
    # send back to cpu
    # numpy ndarray
    return x.detach().cpu().numpy()


def get_logger(log_file="out.log"):
    os.makedirs("logs", exist_ok=True)  # logs directory
    log_file = os.path.join("logs", log_file)  # log path

    # Create handlers
    rich_handler = RichHandler(rich_tracebacks=True)
    file_handler = logging.FileHandler(log_file)
    
    # Set the same format for both handlers
    FORMAT = "%(message)s"

    # Configure root logger
    logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", 
                        handlers=[rich_handler, file_handler])
    
    log = logging.getLogger("rich")
    return log

LOG = get_logger()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        # store metric statistics
        self.val = 0  # value
        self.sum = 0  # running sum
        self.avg = 0  # running average
        self.count = 0  # steps counter

    def update(self, val, n=1):
        # update statistic with given new value
        self.val = val  # like loss
        self.sum += val * n  # loss * batch_size
        self.count += n  # count batch samples
        self.avg = self.sum / self.count  # accounts for different sizes


def update_yaml(opts, key, value):
    """
    Update a key in the yaml configuration file

    Args:
        opts (SimpleNamespace): the configuration object
        key (str): the key to update
        value (any): the new value
    """
    import yaml
    # update the opts object
    opts.__dict__[key] = value
    # update the yaml file
    with open(opts.config, "w") as f:
        # dump the updated opts to the yaml file
        yaml.dump(opts.__dict__, f)


def set_seeds(seed):
    """Set seeds for all random number generators"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def imshow(img, save=None):
    # Denormalize the image
    # img = img * torch.tensor([0.2023, 0.1994, 0.2010])[:, None, None] + torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None]
    img = img.clamp(0, 1)  # Clamp values to be in the [0, 1] range

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")

    if save:
        plt.savefig(save)

    # plt.show()


def visualize(model, model_name, input_data):
    from torchinfo import summary
    from rich.console import Console
    out = model(input_data)

    console = Console()
    console.print(f"Model {model_name}, computed output shape = {out.shape}")

    model_stats = summary(
        model,
        input_data=input_data,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            # "params_percent",
            # "kernel_size",
            # "mult_adds",
        ],
        row_settings=("var_names",),
        col_width=18,
        depth=8,
        verbose=0,
    )
    console.print(model_stats)
