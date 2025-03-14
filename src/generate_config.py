""" Easy script for generating yaml configs """

import os
import yaml
import copy
from ipdb import launch_ipdb_on_exception


def generate_config(param_seq):
    # param_seq : list of dict
    # output_dir : str

    # Load the base configuration to which apply the variations
    base_config_path = "config.yaml"
    with open(base_config_path, "r") as f:
        base_config = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    # Create output directory for YAML if it doesn't exist
    output_dir = "experiments"
    os.makedirs(output_dir, exist_ok=True)  # output dir not tracked by git

    # Generate all parameter combinations
    exp_count = 0
    for param_dict in param_seq:
        # Create a new configuration by updating the base config
        config = copy.deepcopy(base_config)
        # Update with the current parameter combination
        for key, value in param_dict.items():
            config[key] = value
        # Save the configuration to a YAML file
        exp_name = f"{config['model_name']}_prob_{config['label_corruption_prob']};type_{config['data_corruption_type']}"
        config["experiment_name"] = exp_name
        fname = exp_name + ".yaml"
        output_path = os.path.join(output_dir, fname)
        with open(output_path, "w") as f:
            yaml.dump(config, f)
        exp_count += 1
        print(f"Generated config: {output_path}")
        print(f"Parameters: {param_dict}")
        print(f"Experiment name: {config["experiment_name"]}")
        print()

    print(f"Generated {exp_count} configurations")


if __name__ == "__main__":

    param_seq = [
        # {"model_name": "Net", "label_corruption_prob": 0.0, "data_corruption_type": "none"},
        # {"model_name": "Net", "label_corruption_prob": 0.1, "data_corruption_type": "none"},
        # {"model_name": "Net", "label_corruption_prob": 0.2, "data_corruption_type": "none"},
        # {"model_name": "Net", "label_corruption_prob": 0.3, "data_corruption_type": "none"},
        # {"model_name": "Net", "label_corruption_prob": 0.4, "data_corruption_type": "none"},
        # {"model_name": "Net", "label_corruption_prob": 1.0, "data_corruption_type": "none"},
        # {"model_name": "Net", "label_corruption_prob": 0.0, "data_corruption_type": "shuffled pixels"},
    ]

    with launch_ipdb_on_exception():
        generate_config(param_seq)
