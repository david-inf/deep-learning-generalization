
import logging
from rich.logging import RichHandler

import matplotlib.pyplot as plt
import numpy as np


def N(x):
    # detach from computational graph
    # send back to cpu
    # numpy ndarray
    return x.detach().cpu().numpy()


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log
LOG = get_logger()


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


## ***************************************** ##


def plot_csv_data(csv_path, title=None, xlabel=None, ylabel=None,
                  figsize=(10, 6), style=None, save_path=None):
    """
    Plot data from a CSV file using matplotlib.
    
    Parameters:
    ----------
    csv_path : str
        Path to the CSV file.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    figsize : tuple, optional
        Figure size as (width, height) in inches.
    style : str or list, optional
        Line style for the plot (e.g., 'o-', '--', etc.) or list of styles for each y_column.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    
    Returns:
    -------
    fig, ax : tuple
        The figure and axis objects.
    """
    import pandas as pd
    import numpy as np

    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    x_data = df.index
    y_columns = df.columns[1:]

    # Plot each y column
    for i, col in enumerate(y_columns):
        # current_style = style[i] if isinstance(style, list) and i < len(style) else style
        ax.plot(x_data, df[col], label=col)  # add current_style if needed
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(df.columns[0])

    if ylabel:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    # Add legend if multiple y columns
    if len(y_columns) > 1:
        ax.legend()

    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    return fig, ax


if __name__ == "__main__":
    plot_csv_data(
        "data.csv",
        xlabel="step", ylabel="Training loss")
