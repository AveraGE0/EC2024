import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt

# Read the YAML configuration file to get the experiment name and enemy groups
try:
    with open('config_nsga2.yaml', 'r') as f:
        config = yaml.safe_load(f)
    experiment_name = config['experiment']['name']
    enemy_groups = config['experiment']['enemy_groups']
    multiple_runs = config['parallel']['multiple_runs']
except Exception as e:
    raise Exception("Error reading 'config_nsga2.yaml': " + str(e))

# Define the experiment directory as the current directory plus the experiment name
experiment_directory = os.path.join(os.getcwd(), experiment_name)

def get_mean_gains(filename):
    """Read a CSV file and return the mean gains per run."""
    df = pd.read_csv(filename)
    mean_gains = df.mean(axis=1)
    return mean_gains

# Initialize data structures
data = []
labels = []
positions = []
pos = 1  # Starting position for boxplots

# Read data from each file based on enemy groups
for idx, enemy_group in enumerate(enemy_groups):
    # Construct the filename based on the enemy group
    enemy_group_str = '_'.join(str(e) for e in enemy_group)
    filename = os.path.join(
        experiment_directory,
        f'gains_group_{enemy_group_str}.csv'
    )
    if not os.path.exists(filename):
        print(f"File {filename} does not exist. Skipping.")
        continue
    # Get mean gains
    mean_gains = get_mean_gains(filename)
    data.append(mean_gains)  # Only add to data if file exists
    # Create a label based on the enemy group
    group_label = f'Group {enemy_group_str}'
    labels.append(group_label)  # Only add to labels if file exists
    # Append position
    positions.append(pos)
    pos += 1  # Increment position
    # Add space after each pair of boxplots for future models
    if (idx + 1) % 2 == 0:
        pos += 1  # Leave a space

# Check if there's any data to plot
if len(data) == 0:
    print("No data available to plot. Exiting.")
else:
    # Create the boxplots
    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

    # Customize the plot
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Gains')
    ax.set_title('Comparison of Mean Gains between Enemy Groups')

    # Optional: Customize boxplot colors
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.tight_layout()

    # Save the plot as a PNG file in the experiment directory
    output_filename = os.path.join(experiment_directory, 'mean_gains_boxplot.png')
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")

    plt.show()