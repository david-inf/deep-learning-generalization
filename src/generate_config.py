""" Easy script for generating yaml configs """

import os
import yaml
import copy
from ipdb import launch_ipdb_on_exception


PROBS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CORRUPS = ["none", "shuff_pix", "rand_pix", "gauss_pix"]
MODELS = ["Net", "MLP1", "MLP3", "AlexNet", "Inception"]


def generate_config(param_seq, base_config_path="config-f1.yaml"):
    # param_seq : list of dict
    # output_dir : str

    # Load the base configuration to which apply the variations
    with open(base_config_path, "r") as f:
        base_config = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    # Generate all parameter combinations
    exp_count = 0
    for param_dict in param_seq:
        # Create a new configuration by updating the base config
        config = copy.deepcopy(base_config)

        # Update with the current parameter combination
        for key, value in param_dict.items():
            config[key] = value

        # Save the configuration to a YAML file
        if config["figure1"]:
            exp_name = f"{config['model_name']}_{config['label_corruption_prob']}_{config['data_corruption_type']}"
        else:
            exp_name = f"{config["model_name"]}_bn{config["bn"]}"
        config["experiment_name"] = exp_name
        fname = exp_name + ".yaml"
        output_dir = os.path.join("experiments", config["model_name"])
        output_path = os.path.join(output_dir, fname)
        with open(output_path, "w") as f:
            yaml.dump(config, f)
        exp_count += 1

        print(f"Generated config: {output_path}")
        print(f"Parameters: {param_dict}")
        print(f"Experiment name: {config["experiment_name"]}")
        print()

    print(f"Generated {exp_count} configurations")


# TODO: can be improved with **kwargs or something like that
def generate_dicts(model_name="Net", probs=PROBS, corrups=CORRUPS, lr=0.01):
    # Create output directory for YAML configuration files if it doesn't exist already
    os.makedirs(os.path.join("experiments", model_name), exist_ok=True)
    from itertools import product
    param_seq = []
    for prob, corrup in product(probs, corrups):
        param_seq.append({
            "model_name": model_name,
            "label_corruption_prob": prob,
            "data_corruption_type": corrup,
            "learning_rate": lr,
            "figure1": True,
        })
    return param_seq


if __name__ == "__main__":

    # param_seq = generate_dicts(corrups=["none"])
    param_seq = generate_dicts(model_name="MLP1", probs=[0.0, 0.1, 0.2, 1.0], corrups=["none"])
    # param_seq = generate_dicts(model_name="MLP1", probs=[0.3], corrups=["none"])
    # param_seq = generate_dicts(model_name="Inception", probs=[0.1], corrups=["none"])

    # Figure 2 dicts
    # param_seq = [
    #     {"model_name": "Inception", "figure1": False, "bn": True},
    #     {"model_name": "Inception", "figure1": False, "bn": False}
    # ]

    with launch_ipdb_on_exception():
        generate_config(param_seq, base_config_path="config-f1.yaml")
