import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd  # Added for easier data handling


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

        print(f"\nProcessing Enemy Group: {group_tuple}")

        for run_id in range(1, multiple_runs + 1):
            group_str = '_'.join(map(str, group))
            metrics_filename = os.path.join(experiment_directory, f'metrics_run_{run_id}_group_{group_str}.csv')

            if not os.path.isfile(metrics_filename):
                print(f"  [Run {run_id}] Metrics file {metrics_filename} not found. Skipping run.")
                continue

            try:
                # Read the CSV file using pandas for reliability
                df = pd.read_csv(metrics_filename)
            except Exception as e:
                print(f"  [Run {run_id}] Error reading {metrics_filename}: {e}. Skipping run.")
                continue

            # Validate required columns
            required_columns = {'Generation', 'Avg_Scalar_Fitness', 'Max_Scalar_Fitness'}
            if not required_columns.issubset(df.columns):
                print(f"  [Run {run_id}] Missing columns in {metrics_filename}. Required columns: {required_columns}. Skipping run.")
                continue

            # Filter generations up to num_generations
            df_filtered = df[df['Generation'] <= num_generations]

            # Check if all generations are present
            expected_generations = num_generations + 1  # Including generation 0
            if len(df_filtered) != expected_generations:
                print(f"  [Run {run_id}] Incomplete data in {metrics_filename}. Expected {expected_generations} generations, got {len(df_filtered)}. Skipping run.")
                continue

            # Sort by Generation to ensure correct order
            df_filtered = df_filtered.sort_values('Generation')

            # Append fitness data
            run_mean_fitness = df_filtered['Avg_Scalar_Fitness'].values
            run_max_fitness = df_filtered['Max_Scalar_Fitness'].values

            aggregated_data[group_tuple]['mean_fitness_runs'].append(run_mean_fitness)
            aggregated_data[group_tuple]['max_fitness_runs'].append(run_max_fitness)

            print(f"  [Run {run_id}] Successfully aggregated data.")

        # Convert lists to NumPy arrays for statistical computations
        mean_fitness_runs = np.array(aggregated_data[group_tuple]['mean_fitness_runs'])
        max_fitness_runs = np.array(aggregated_data[group_tuple]['max_fitness_runs'])

        if mean_fitness_runs.size == 0 or max_fitness_runs.size == 0:
            print(f"No valid data collected for enemy group {group_tuple}.")
            del aggregated_data[group_tuple]  # Remove the group since no data is present
            continue

        # Compute mean and std deviation across runs for each generation
        aggregated_data[group_tuple]['mean_fitness_mean'] = np.mean(mean_fitness_runs, axis=0)
        aggregated_data[group_tuple]['mean_fitness_std'] = np.std(mean_fitness_runs, axis=0)
        aggregated_data[group_tuple]['max_fitness_mean'] = np.mean(max_fitness_runs, axis=0)
        aggregated_data[group_tuple]['max_fitness_std'] = np.std(max_fitness_runs, axis=0)

        # Debugging: Print the first few std values to verify
        print(f"  Enemy Group {group_tuple}:")
        print(f"    Mean Fitness Mean (first 5 generations): {aggregated_data[group_tuple]['mean_fitness_mean'][:5]}")
        print(f"    Mean Fitness Std  (first 5 generations): {aggregated_data[group_tuple]['mean_fitness_std'][:5]}")
        print(f"    Max Fitness Mean  (first 5 generations): {aggregated_data[group_tuple]['max_fitness_mean'][:5]}")
        print(f"    Max Fitness Std   (first 5 generations): {aggregated_data[group_tuple]['max_fitness_std'][:5]}")
        print("-" * 60)

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

        plt.figure(figsize=(12, 8))

        # Plot Mean of Mean Fitness
        plt.plot(generations, metrics['mean_fitness_mean'], label='Mean of Mean Fitness', color='blue', linewidth=2)
        plt.fill_between(
            generations,
            metrics['mean_fitness_mean'] - metrics['mean_fitness_std'],
            metrics['mean_fitness_mean'] + metrics['mean_fitness_std'],
            color='blue',
            alpha=0.3,
            label='Std Dev of Mean Fitness'
        )

        # Plot Mean of Max Fitness
        plt.plot(generations, metrics['max_fitness_mean'], label='Mean of Max Fitness', color='red', linestyle='--', linewidth=2)
        plt.fill_between(
            generations,
            metrics['max_fitness_mean'] - metrics['max_fitness_std'],
            metrics['max_fitness_mean'] + metrics['max_fitness_std'],
            color='red',
            alpha=0.3,
            label='Std Dev of Max Fitness'
        )

        # Customize the plot
        enemy_group_str = ', '.join(map(str, enemy_group))
        plt.title(f'Fitness over Generations for Enemy Group [{enemy_group_str}]', fontsize=16)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Scalar Fitness', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(experiment_directory, f'fitness_plot_group_{"_".join(map(str, enemy_group))}.png')
        plt.savefig(plot_filename, dpi=300)
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
