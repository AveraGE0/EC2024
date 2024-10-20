import numpy as np
import csv
import yaml
import os
from evoman.environment import Environment

# Read the YAML configuration file to get the experiment name and enemy groups
try:
    with open('config_nsga2.yaml', 'r') as f:
        config = yaml.safe_load(f)
    experiment_name = config['experiment']['name']
    enemy_groups = config['experiment']['enemy_groups']
except Exception as e:
    raise Exception("Error reading 'config_nsga2.yaml': " + str(e))

# Define the experiment directory as the current directory plus the experiment name
experiment_directory = os.path.join(os.getcwd(), experiment_name)

def evaluate_individual_on_single_enemies(individual, experiment_name):
    """Evaluate an individual against each enemy separately and return gains."""
    gains = []

    # Loop over each enemy (1 to 8)
    for enemy_id in range(1, 9):
        print(f"Evaluating Enemy {enemy_id}...")
        # Initialize the environment for the specific enemy
        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy_id],
            playermode='ai',
            enemymode='static',
            level=2,
            speed='fastest',
            multiplemode='no'
        )

        # Play the game with the current individual against the current enemy
        try:
            fitness, player_life, enemy_life, time = env.play(pcont=individual)
        except Exception as e:
            raise Exception(f"Error during simulation with Enemy {enemy_id}: " + str(e))

        # Compute the gain
        gain = player_life - enemy_life

        # Store the gain for the current enemy
        gains.append(gain)

    return gains

# Loop over each enemy group
for enemy_group in enemy_groups:
    # Create a string to identify the enemy group
    enemy_group_str = '_'.join(map(str, enemy_group))
    group_name = f"group_{enemy_group_str}"
    print(f"Processing enemy group: {group_name}")

    # Initialize data structure to collect gains per run
    gains_per_run = []

    # Loop over runs (files from 1 to 10)
    for run in range(1, 11):
        print(f"Processing run {run} for enemy group {group_name}...")
        # Construct file name
        filename = os.path.join(
            experiment_directory,
            f'best_solution_run_{run}_{enemy_group_str}.txt'
        )

        # Check if the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Best solution file '{filename}' not found.")

        # Read the weights from the file
        try:
            individual = np.loadtxt(filename)
        except Exception as e:
            raise Exception(f"Error reading '{filename}': " + str(e))

        # Evaluate the individual on single enemies
        gains = evaluate_individual_on_single_enemies(individual, experiment_name)

        # Collect the gains per run
        gains_per_run.append(gains)

    # Now, write the gains into a CSV file specific to the enemy group
    csv_filename = f'gains_{group_name}.csv'
    with open(csv_filename, mode='w', newline='') as csv_file:
        # Define the fieldnames as 'Enemy 1' to 'Enemy 8'
        fieldnames = [f'Enemy {i}' for i in range(1, 9)]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Write each run's gains as a row in the CSV
        for run_gains in gains_per_run:
            # Create a dictionary for the row
            row = {f'Enemy {i+1}': run_gains[i] for i in range(8)}
            writer.writerow(row)

    print(f"Gains have been successfully written to '{csv_filename}'.")