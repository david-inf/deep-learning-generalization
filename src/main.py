from types import SimpleNamespace
import argparse
import yaml

import torch
import torch.optim as optim


def main(opts):
    print(opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML Configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)

    ## Create object
    opts = SimpleNamespace(**configs)

    opts.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", opts.device)

    main(opts)
