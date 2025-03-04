
import matplotlib.pyplot as plt

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

def N(x):
    # detach from computational graph
    # send back to cpu
    # numpy ndarray
    return x.detach().cpu().numpy()


def visualize(model, model_name, input_data):
    from torchinfo import summary
    from rich.console import Console
    out = model(input_data)

    console = Console()
    console.print(f"Computed output shape = {out.shape}")

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


def plot_data(imgs, labels, row_title=None, **imshow_kwargs):
    # specifically for CIFAR10
    NAMES = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    fig.set_figheight(12)
    fig.set_figwidth(24)

    for row_idx in range(len(imgs)):
        row = imgs[row_idx]
        for col_idx in range(len(row)):
            img = row[col_idx]
            # The label is one-hot...
            label = labels[row_idx][col_idx].argmax().item()
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set(xlabel=f"{label}:{NAMES[label]}")

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


# TODO: plotting routines
