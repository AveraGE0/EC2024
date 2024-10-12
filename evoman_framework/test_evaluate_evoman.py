import numpy as np
import math
import os  # For creating the directory
from evoman.environment import Environment
from demo_controller import player_controller

# Neural network parameters
num_inputs = 20
num_outputs = 5
n_hidden_neurons = 10

# Calculate the number of weights in the neural network
num_weights = (
    (num_inputs * n_hidden_neurons) +
    (n_hidden_neurons * num_outputs) +
    n_hidden_neurons +  # Bias for hidden layer
    num_outputs         # Bias for output layer
)

# Initialize the real Evoman environment for testing
experiment_name = 'test_evaluate_evoman'

# Ensure the directory exists
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(
    experiment_name=experiment_name,
    enemies=[2, 4, 7],  # Example enemies
    multiplemode="yes",
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),  # Dynamically set hidden neurons
    enemymode="static",
    level=2,
    contacthurt='player',
    speed="fastest"
)

def evaluate_real(individual):
    """
    Evaluation function using the real Evoman environment.
    """
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))

    # Calculate gain (player life - enemy life)
    total_enemy_life = sum(enemy_life) if isinstance(enemy_life, list) else enemy_life
    gain = player_life - total_enemy_life

    # Logarithm of time (handling any zero-time edge cases)
    log_time = math.log(time) if time > 0 else 0

    # Calculate overall fitness as: player_life - enemy_life - log(time)
    overall_fitness = player_life - total_enemy_life - log_time

    # Print for analysis
    print(f"Player Life: {player_life}, Total Enemy Life: {total_enemy_life}, Time: {time}, Log Time: {log_time}")
    print(f"Gain: {gain}, Overall Fitness: {overall_fitness}")

    # Return the three objectives for the multi-objective optimization
    return total_enemy_life, -player_life, log_time

# Test with a random individual
test_individual = np.random.uniform(-1, 1, num_weights)  # Random individual with calculated number of weights
evaluate_real(test_individual)