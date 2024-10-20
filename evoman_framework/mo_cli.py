import os
import argparse
import yaml
from evolution_multi_objective import run_experiment

if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run NSGA-II experiment with the given configuration.")
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file.')

    args = parser.parse_args()

    # Load the configuration from a YAML file
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Run experiment for each repetition with provided seeds
    for i_run, seed in zip(list(range(config['n_repetitions'])), config['seeds']):
        config["name"] = os.path.basename(args.config_path).split(".")[0]
        config["seed"] = seed
        run_experiment(config)
