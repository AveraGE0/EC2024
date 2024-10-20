import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd


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
    Aggregate all metrics across multiple runs for each enemy group.

    Parameters:
    - experiment_directory: str, path to the experiment directory.
    - enemy_groups: list of lists, each sublist contains enemy IDs for a group.
    - multiple_runs: int, number of independent evolutionary runs.
    - num_generations: int, total number of generations per run.

    Returns:
    - aggregated_data: dict, structured as:
        {
            enemy_group_tuple: {
                'metric_name': {
                    'mean': np.array([...]),
                    'std': np.array([...])
                },
                ...
            },
            ...
        }
    """
    aggregated_data = {}

    for group in enemy_groups:
        group_tuple = tuple(group)
        aggregated_data[group_tuple] = {}  # Initialize dictionary for this enemy group

        print(f"\nProcessing Enemy Group: {group_tuple}")

        # Initialize a dictionary to store metrics for all runs
        metrics_runs = {}

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

            # Convert all relevant columns to numeric types
            numeric_columns = [
                'Avg_Scalar_Fitness', 'Max_Scalar_Fitness',
                'Avg_Defeated_Enemies', 'Max_Defeated_Enemies',
                'Avg_Gain', 'Max_Gain',
                'Avg_Life', 'Max_Life',
                'Avg_Time', 'Min_Time',
                'Euclidean_Diversity', 'Std_Dev'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Ensure 'Generation' column is present and numeric
            if 'Generation' in df.columns:
                df['Generation'] = pd.to_numeric(df['Generation'], errors='coerce')
            else:
                print(f"  [Run {run_id}] 'Generation' column not found in {metrics_filename}. Available columns: {df.columns.tolist()}")
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

            # Collect metrics
            for col in numeric_columns:
                if col in df_filtered.columns:
                    if col not in metrics_runs:
                        metrics_runs[col] = []
                    metrics_runs[col].append(df_filtered[col].values)

            print(f"  [Run {run_id}] Successfully aggregated data.")

        # Compute mean and std deviation across runs for each generation
        for metric_name, runs_data in metrics_runs.items():
            runs_array = np.array(runs_data)

            if runs_array.size == 0:
                continue  # Skip if no data

            # Compute mean and std across runs for each generation
            metric_mean = np.nanmean(runs_array, axis=0)
            metric_std = np.nanstd(runs_array, axis=0)

            # Store in aggregated_data
            aggregated_data[group_tuple][metric_name] = {
                'mean': metric_mean,
                'std': metric_std
            }

        print(f"  [Enemy Group {group_tuple}] Metrics aggregated.")
        print("-" * 60)

    return aggregated_data


def plot_metrics_over_generations(aggregated_data, num_generations, experiment_directory):
    metrics_to_plot = [
        ('Avg_Scalar_Fitness', 'Max_Scalar_Fitness'),
        ('Avg_Gain', 'Max_Gain'),
        ('Avg_Life', 'Max_Life'),
        ('Avg_Defeated_Enemies', 'Max_Defeated_Enemies'),
        ('Avg_Time', 'Min_Time'),
        ('Euclidean_Diversity', 'Std_Dev')
    ]

    for enemy_group, metrics in aggregated_data.items():
        generations = np.arange(0, num_generations + 1)  # Including generation 0

        enemy_group_str = ', '.join(map(str, enemy_group))
        group_filename_part = '_'.join(map(str, enemy_group))

        # Create a figure for each pair of metrics
        for metric_pair in metrics_to_plot:
            plt.figure(figsize=(12, 8))

            for metric_name in metric_pair:
                if metric_name in metrics:
                    mean_values = metrics[metric_name]['mean']
                    std_values = metrics[metric_name]['std']

                    # Add debug statements to check the data
                    print(f"\nPlotting {metric_name} for Enemy Group {enemy_group}:")
                    print(f"Mean values: {mean_values}")
                    print(f"Std values: {std_values}")

                    # Check if mean_values contains valid data
                    if np.isnan(mean_values).all():
                        print(f"Warning: All mean values for {metric_name} are NaN. Skipping plot.")
                        continue

                    # Determine plot style based on metric name
                    if 'Avg' in metric_name:
                        label = f"Mean of {metric_name.replace('_', ' ')}"
                        color = 'blue'
                        linestyle = '-'
                    elif 'Max' in metric_name:
                        label = f"Mean of {metric_name.replace('_', ' ')}"
                        color = 'red'
                        linestyle = '--'
                    elif 'Min' in metric_name:
                        label = f"Mean of {metric_name.replace('_', ' ')}"
                        color = 'green'
                        linestyle = '-.'
                    else:
                        label = f"Mean of {metric_name.replace('_', ' ')}"
                        color = 'purple'
                        linestyle = ':'

                    plt.plot(generations, mean_values, label=label, color=color, linestyle=linestyle, linewidth=2)
                    plt.fill_between(
                        generations,
                        mean_values - std_values,
                        mean_values + std_values,
                        color=color,
                        alpha=0.3,
                        label=f"Std Dev of {metric_name.replace('_', ' ')}"
                    )

            # Customize the plot
            plt.title(f'{metric_pair[0].replace("_", " ")} and {metric_pair[1].replace("_", " ")} over Generations\nEnemy Group [{enemy_group_str}]', fontsize=16)
            plt.xlabel('Generation', fontsize=14)
            plt.ylabel('Value', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()

            # Save the plot
            plot_filename = os.path.join(experiment_directory, f'plot_{"_".join(metric_pair)}_group_{group_filename_part}.png')
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
    plot_metrics_over_generations(aggregated_data, num_generations, experiment_directory)

    print("All plots have been generated and saved.")


if __name__ == "__main__":
    main()
