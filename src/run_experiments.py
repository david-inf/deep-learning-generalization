
import argparse
import subprocess
import os
import yaml
import itertools
import copy
from datetime import datetime
import numpy as np


def run_experiments(exp_name, param_variations, output_dir):
    """
    Run multiple experiments by varying parameters from a base configuration.
    Each experiment will be logged to comet_ml with its own unique name and config.
    
    Parameters:
    -----------
    exp_name : str
        General experiment name (learning curves, generalization error, etc.)
    param_variations : dict
        Dictionary where keys are parameter names and values are lists of values to try
    output_dir : str
        Directory to store generated configuration files
    """
    # Load the base configuration to which apply the variations
    base_config_path = "config.yaml"
    with open(base_config_path, "r") as f:
        base_config = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    # Create output directory for YAML if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)  # output dir not tracked by git

    # Generate all parameter combinations, these are based on the experiment being run
    param_names, param_values = list(param_variations.keys()), list(param_variations.values())

    # Generate timestamp for unique experiment naming
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # rivedere

    # Calculate total number of experiments
    total_experiments = 1
    for values in param_values:
        total_experiments *= len(values)
    print(f"Preparing to run {total_experiments} experiments...")

    # Generate and run experiments for each parameter combination
    experiment_count = 0
    for values in itertools.product(*param_values):  # like a nested loop
        # Create a new configuration by updating the base config
        config = copy.deepcopy(base_config)
        # Update with the current parameter combination
        param_desc = []
        for name, value in zip(param_names, values):
            config[name] = value
            param_desc.append(f"{name}={value}")

        # Create a descriptive run name for comet_ml; this would be a good naming
        # config["experiment_name"] = f"{exp_name}_{experiment_count}_{'-'.join(param_desc)}"
        config["experiment_name"] = exp_name

        # Save the configuration to a new file
        # config_filename = f"{output_dir}/config_{timestamp}_{experiment_count}.yaml"
        config_filename = f"{output_dir}/{exp_name}_{experiment_count}.yaml"
        with open(config_filename, "w") as f:
            # Write the config to a YAML file
            yaml.dump(config, f, default_flow_style=False)

        experiment_count += 1

        print(f"\nRunning experiment {exp_name} [{experiment_count}/{total_experiments}]")
        print(f"Experiment name: {config["experiment_name"]}")
        print(f"Parameters: {', '.join(param_desc)}")
        print(f"Config file: {config_filename}")

        # Run the experiment using the main.py script
        # This passes the generated config file to main.py
        # By doing this a new comet_ml experiment is created each time
        result = subprocess.run(["python", "main.py", "--config", config_filename])

        if result.returncode == 0:
            print(f"Experiment {experiment_count}/{total_experiments} completed successfully.")
        else:
            print(f"Experiment {experiment_count}/{total_experiments} failed with return code {result.returncode}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose which experiment to run")
    # in order to run experiments, we need to create different yaml files
    parser.add_argument("--output-dir", default="experiments", help="Directory for output config files")

    # Specify which experiment to run
    parser.add_argument("--name", type=str, help="Name of the experiment to run", default="learn-curves",
                        choices=["learn-curves", "conv+err", "inception-reg"])

    args = parser.parse_args()  # get the arguments

    # TODO: si può fare meglio perché tanto la roba tra learning curve e err è a comune
    # Define parameter variations for each experiment
    if args.name == "learn-curves":
        # learning curve varying randomization test
        # fixed model
        param_variations = {
            "label_corruption_prob": [0.0, 0.5, 1.0],
            "data_corruption_type": ["none"],
            "model_name": ["Net"]
        }

    if args.name == "conv+err":
        # convergence slowdown and generalization error growth
        # varying the label corruption and the model
        param_variations = {
            "label_corruption_prob": np.linspace(0., 1., 11, endpoint=True),
            "model_name": ["Net"]
        }

    # Run the experiments
    run_experiments(args.name, param_variations, args.output_dir)
    # At the end, all experiments will be mixed in comet_ml, but we can filter them by name

# Questo si può fare molto più semplice per partire
# definisco io a mano i parametri degli esperimenti e li lancio
# uno alla volta con un for
