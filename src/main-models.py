""" Script for inspecting models """

import torch
from models.inception import InceptionSmall
from models.alexnet import AlexNetSmall
from models.simple_mlp import MLP
from utils import visualize


def main(args):
    if args.name.lower() == "inception":
        model = InceptionSmall()
    elif args.name.lower() == "alexnet":
        model = AlexNetSmall()
    elif args.name.lower() == "mlp":
        model = MLP(1)
    else:
        raise ValueError(f"Unknown model name {args.name}")
    input_data = torch.randn(128, 3, 28, 28)
    visualize(model, f"{args.name}", input_data)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Model name")
    args = parser.parse_args()

    with launch_ipdb_on_exception():
        main(args)
