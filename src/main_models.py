"""Quick script for inspecting models"""

import torch
from utils import visualize
from main_fig1 import get_model


def main(opts):
    model = get_model(opts)
    input_data = torch.randn(128, 3, 28, 28).to(opts.device)
    visualize(model, f"{opts.model_name}", input_data)


if __name__ == "__main__":
    from cmd_args import parse_args
    from ipdb import launch_ipdb_on_exception
    opts = parse_args()
    with launch_ipdb_on_exception():
        main(opts)
