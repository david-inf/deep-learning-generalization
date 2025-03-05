
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
    Each experiment will be logged to wandb with its own unique name and config.
    
    Parameters:
    -----------
    exp_name : str
        Experiment name (learning curves, generalization error, etc.)
    param_variations : dict
        Dictionary where keys are parameter names and values are lists of values to try
    output_dir : str
        Directory to store generated configuration files
    """
    # Load the base configuration
    base_config_path = "config.yaml"
    with open(base_config_path, "r") as f:
        base_config = yaml.load(f, Loader=yaml.SafeLoader)  # dict

    # Create output directory if it doesn't exist
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
        config["experiment_name"] = exp_name  # can be useful
        # Update with the current parameter combination
        param_desc = []
        for name, value in zip(param_names, values):
            config[name] = value
            param_desc.append(f"{name}={value}")

        # Create a descriptive run name for wandb
        # config["run_name"] = f"exp_{timestamp}_{experiment_count}_{'-'.join(param_desc)}"
        config["run_name"] = config["model_name"]

        # Save the configuration to a new file
        # config_filename = f"{output_dir}/config_{timestamp}_{experiment_count}.yaml"
        config_filename = f"{output_dir}/{exp_name}_{experiment_count}.yaml"
        with open(config_filename, "w") as f:
            # Write the config to a YAML file
            yaml.dump(config, f, default_flow_style=False)

        experiment_count += 1

        print(f"\nRunning experiment {exp_name} [{experiment_count}/{total_experiments}]")
        print(f"Parameters: {', '.join(param_desc)}")
        print(f"Run name: {config['run_name']}")
        print(f"Config file: {config_filename}")

        # Run the experiment using the main.py script
        # This passes the generated config file to main.py
        # By doing this a new wandb is created each time
        result = subprocess.run(["python", "main.py", "--config", config_filename])

        if result.returncode == 0:
            print(f"Experiment {experiment_count}/{total_experiments} completed successfully.")
        else:
            print(f"Experiment {experiment_count}/{total_experiments} failed with return code {result.returncode}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose which experiment to run")
    # in order to run experiments, we need to create different yaml files
    parser.add_argument("--output-dir", default="experiments", help="Directory for output config files")

    # Allow specifying parameter variations directly from command line
    # parser.add_argument('--label-corruption', type=str, help='Comma-separated label corruption probabilities')
    # parser.add_argument('--model-name', type=str, help='Comma-separated model names')
    # parser.add_argument('--num-epochs', type=str, help='Comma-separated number of epochs')

    # Specify which experiment to run
    parser.add_argument("--name", type=str, help="Name of the experiment to run", default="learn-curves",
                        choices=["learn-curves", "conv+err", "inception-reg"])

    args = parser.parse_args()

    # Build parameter variations from command line arguments
    # param_variations = {}
    # if args.label_corruption:
    #     param_variations["label_corruption_prob"] = [float(x) for x in args.label_corruption.split(",")]
    # if args.model_name:
    #     param_variations["model_name"] = args.model_name.split(",")
    # if args.num_epochs:
    #     param_variations["num_epochs"] = [int(x) for x in args.num_epochs.split(",")]
    
    # If no variations specified, use default ones
    # if not param_variations:
    #     param_variations = {
    #         "label_corruption_prob": [0.0, 0.1, 0.2, 0.3],
    #         "learning_rate": [0.001, 0.01],
    #     }
    #     print("Using default parameter variations:")
    #     for k, v in param_variations.items():
    #         print(f"  {k}: {v}")
    # else:
    #     print("Using parameter variations:")
    #     for k, v in param_variations.items():
    #         print(f"  {k}: {v}")

    # TODO: si può fare meglio perché tanto la roba tra learning curve e err è a comune
    # Define parameter variations for each experiment
    if args.name == "learn-curves":
        # learning curve varying randomization test
        # fixed model
        param_variations = {
            "label_corruption_prob": [0.0, 0.5, 1.0],
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
    # At the end, all experiments will be mixed in wandb, but we can filter them by name
