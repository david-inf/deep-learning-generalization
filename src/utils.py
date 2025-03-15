
import logging
from rich.logging import RichHandler

import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

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


# def plot_data(imgs, labels, row_title=None, **imshow_kwargs):
#     # specifically for CIFAR10
#     NAMES = ['plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     num_rows = len(imgs)
#     num_cols = len(imgs[0])
#     fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     fig.set_figheight(12)
#     fig.set_figwidth(24)

#     for row_idx in range(len(imgs)):
#         row = imgs[row_idx]
#         for col_idx in range(len(row)):
#             img = row[col_idx]
#             # The label is one-hot...
#             label = labels[row_idx][col_idx].argmax().item()
#             boxes = None
#             masks = None
#             if isinstance(img, tuple):
#                 img, target = img
#                 if isinstance(target, dict):
#                     boxes = target.get("boxes")
#                     masks = target.get("masks")
#                 elif isinstance(target, tv_tensors.BoundingBoxes):
#                     boxes = target
#                 else:
#                     raise ValueError(f"Unexpected target type: {type(target)}")
#             img = F.to_image(img)
#             if img.dtype.is_floating_point and img.min() < 0:
#                 # Poor man's re-normalization for the colors to be OK-ish. This
#                 # is useful for images coming out of Normalize()
#                 img -= img.min()
#                 img /= img.max()

#             img = F.to_dtype(img, torch.uint8, scale=True)
#             if boxes is not None:
#                 img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
#             if masks is not None:
#                 img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

#             ax = axs[row_idx, col_idx]
#             ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#             ax.set(xlabel=f"{label}:{NAMES[label]}")

#     if row_title is not None:
#         for row_idx in range(num_rows):
#             axs[row_idx, 0].set(ylabel=row_title[row_idx])

#     plt.tight_layout()
#     plt.show()


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

# TODO: plotting routines

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
