import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import yaml

def load_config(config_file='config_nsga2.yaml'):
    """
    Load the configuration from a YAML file located in the same directory as this script.

    Parameters:
    - config_file: str, name of the YAML configuration file.

    Returns:
    - config: dict, configuration parameters.
    """
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found in {script_dir}.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def aggregate_metrics(experiment_directory, enemy_groups, multiple_runs, num_generations):
    """
    Aggregate mean and max scalar fitness metrics across multiple runs for each enemy group.

    Parameters:
    - experiment_directory: str, path to the experiment directory.
    - enemy_groups: list of lists, each sublist contains enemy IDs for a group.
    - multiple_runs: int, number of independent evolutionary runs.
    - num_generations: int, total number of generations per run.

    Returns:
    - aggregated_data: dict, structured as:
        {
            enemy_group_tuple: {
                'mean_fitness_mean': np.array([...]),
                'mean_fitness_std': np.array([...]),
                'max_fitness_mean': np.array([...]),
                'max_fitness_std': np.array([...])
            },
            ...
        }
    """
    aggregated_data = {}

    for group in enemy_groups:
        group_tuple = tuple(group)
        aggregated_data[group_tuple] = {
            'mean_fitness_runs': [],
            'max_fitness_runs': []
        }

        for run_id in range(1, multiple_runs + 1):
            group_str = '_'.join(map(str, group))
            metrics_filename = os.path.join(experiment_directory, f'metrics_run_{run_id}_group_{group_str}.csv')

            if not os.path.isfile(metrics_filename):
                print(f"Metrics file {metrics_filename} not found. Skipping run {run_id} for group {group_str}.")
                continue

            # Read the CSV file
            with open(metrics_filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                run_mean_fitness = []
                run_max_fitness = []
                for row in reader:
                    generation = int(row['Generation'])
                    if generation > num_generations:
                        continue
                    run_mean_fitness.append(float(row['Avg_Scalar_Fitness']))
                    run_max_fitness.append(float(row['Max_Scalar_Fitness']))

            # Ensure the run has data for all generations
            expected_generations = num_generations + 1  # Including generation 0
            if len(run_mean_fitness) != expected_generations or len(run_max_fitness) != expected_generations:
                print(f"Incomplete data in {metrics_filename}. Expected {expected_generations} generations, got {len(run_mean_fitness)}.")
                continue

            aggregated_data[group_tuple]['mean_fitness_runs'].append(run_mean_fitness)
            aggregated_data[group_tuple]['max_fitness_runs'].append(run_max_fitness)

        # Convert lists to NumPy arrays for statistical computations
        mean_fitness_runs = np.array(aggregated_data[group_tuple]['mean_fitness_runs'])
        max_fitness_runs = np.array(aggregated_data[group_tuple]['max_fitness_runs'])

        if mean_fitness_runs.size == 0 or max_fitness_runs.size == 0:
            print(f"No valid data collected for enemy group {group_tuple}.")
            continue

        # Compute mean and std deviation across runs for each generation
        aggregated_data[group_tuple]['mean_fitness_mean'] = np.mean(mean_fitness_runs, axis=0)
        aggregated_data[group_tuple]['mean_fitness_std'] = np.std(mean_fitness_runs, axis=0)
        aggregated_data[group_tuple]['max_fitness_mean'] = np.mean(max_fitness_runs, axis=0)
        aggregated_data[group_tuple]['max_fitness_std'] = np.std(max_fitness_runs, axis=0)

    return aggregated_data

def plot_fitness_over_generations(aggregated_data, num_generations, experiment_directory):
    """
    Generate and save plots for fitness over generations for each enemy group.

    Parameters:
    - aggregated_data: dict, aggregated fitness metrics.
    - num_generations: int, total number of generations.
    - experiment_directory: str, path to the experiment directory.
    """
    for enemy_group, metrics in aggregated_data.items():
        generations = np.arange(0, num_generations + 1)  # Including generation 0

        plt.figure(figsize=(10, 6))

        # Plot Mean of Mean Fitness
        plt.plot(generations, metrics['mean_fitness_mean'], label='Mean of Mean Fitness', color='blue')
        plt.fill_between(
            generations,
            metrics['mean_fitness_mean'] - metrics['mean_fitness_std'],
            metrics['mean_fitness_mean'] + metrics['mean_fitness_std'],
            color='blue',
            alpha=0.2
        )

        # Plot Mean of Max Fitness
        plt.plot(generations, metrics['max_fitness_mean'], label='Mean of Max Fitness', color='red', linestyle='--')
        plt.fill_between(
            generations,
            metrics['max_fitness_mean'] - metrics['max_fitness_std'],
            metrics['max_fitness_mean'] + metrics['max_fitness_std'],
            color='red',
            alpha=0.2
        )

        # Customize the plot
        enemy_group_str = ', '.join(map(str, enemy_group))
        plt.title(f'Fitness over Generations for Enemy Group [{enemy_group_str}]')
        plt.xlabel('Generation')
        plt.ylabel('Scalar Fitness')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(experiment_directory, f'fitness_plot_group_{"_".join(map(str, enemy_group))}.png')
        plt.savefig(plot_filename)
        plt.close()

        print(f"Plot saved to {plot_filename}")

def main():
    # Load configuration
    try:
        config = load_config('config_nsga2.yaml')
    except FileNotFoundError as e:
        print(e)
        return

    experiment_directory = config['experiment']['name']
    enemy_groups = config['experiment']['enemy_groups']
    multiple_runs = config['parallel']['multiple_runs']
    num_generations = config['experiment']['n_gen']

    # Check if experiment directory exists
    if not os.path.exists(experiment_directory):
        print(f"Experiment directory '{experiment_directory}' does not exist. Please run the evolutionary algorithm first.")
        return

    # Aggregate metrics
    print("Aggregating metrics...")
    aggregated_data = aggregate_metrics(experiment_directory, enemy_groups, multiple_runs, num_generations)

    if not aggregated_data:
        print("No data available to plot. Exiting.")
        return

    # Generate plots
    print("Generating plots...")
    plot_fitness_over_generations(aggregated_data, num_generations, experiment_directory)

    print("All plots have been generated and saved.")

if __name__ == "__main__":
    main()