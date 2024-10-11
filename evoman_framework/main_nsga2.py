import numpy as np
import pandas as pd
import random
import os
import csv
from deap import base, creator, tools
from evoman.environment import Environment
from demo_controller import player_controller

# Define the experiment name and create the directory if it doesn't exist
experiment_name = 'experiment_nsga2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Neural Network parameters
num_inputs = 20                # Number of inputs (depends on the game)
num_outputs = 5                # Number of outputs (actions)
n_hidden_neurons = 10          # Number of hidden neurons in the neural network

# Calculate the number of weights in the neural network
num_weights = (
    (num_inputs * n_hidden_neurons) +  # Input to hidden layer weights
    (n_hidden_neurons * num_outputs) +  # Hidden to output layer weights
    n_hidden_neurons +  # Hidden layer biases
    num_outputs  # Output layer biases
)

# Initialize the environment for multiple enemies
env = Environment(
    experiment_name=experiment_name,
    enemies=[2, 4, 7],  # Multiple enemies
    multiplemode="yes",  # Indicates a generalist agent
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    contacthurt='player',
    speed="fastest"
)

# Genetic Algorithm parameters
POP_SIZE = 50     # Population size
N_GEN = 10        # Number of generations
CXPB = 0.9        # Crossover probability
MUTPB = 0.1       # Mutation probability

# Define a multi-objective fitness function (minimize all objectives)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))

# Create the Individual class based on list with the defined fitness function
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Attribute generator: Each gene is a float between -1 and 1
toolbox.register("attr_float", random.uniform, -1, 1)

# Structure initializers: Define 'individual' and 'population'
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    # Play the game with the individual's weights against all enemies
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))

    # Calculate fitness objectives
    total_enemy_life = sum(enemy_life) if isinstance(enemy_life, list) else enemy_life
    return total_enemy_life, -player_life, time


toolbox.register("evaluate", evaluate)

# Crossover operator: Simulated Binary Crossover (SBX)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=15)

# Mutation operator: Polynomial mutation
toolbox.register("mutate", tools.mutPolynomialBounded, low=-1.0, up=1.0, eta=15, indpb=1.0/num_weights)

# Selection operator: NSGA-II selection
toolbox.register("select", tools.selNSGA2)


def log_generation_statistics(pop, gen, logbook, pareto_front, env, experiment_name):
    """
    Logs statistics of the current generation to a CSV file in a clear, interpretable format.

    Args:
        pop (list): The current population (list of individuals).
        gen (int): The current generation number.
        logbook: The logbook object (not used in this function but kept for compatibility).
        pareto_front: The current Pareto front (not used in this function but kept for compatibility).
        env: The environment object containing enemy information.
        experiment_name (str): The name of the experiment (used for file path).
    """
    filepath = os.path.join(experiment_name, "generation_statistics.csv")

    # Define the correct field names for the CSV file
    fieldnames = [
        'Generation', 'Evaluations',
        'Avg_Enemy_Health', 'Std_Enemy_Health', 'Min_Enemy_Health', 'Max_Enemy_Health',
        'Avg_Player_Health', 'Std_Player_Health', 'Min_Player_Health', 'Max_Player_Health',
        'Avg_Time', 'Std_Time', 'Min_Time', 'Max_Time'
    ]

    # Check if the file exists and if it is empty
    file_exists = os.path.isfile(filepath)
    file_is_empty = os.stat(filepath).st_size == 0 if file_exists else True

    # Calculate statistics
    fitness_values = [ind.fitness.values for ind in pop]
    avg_fitness = np.mean(fitness_values, axis=0)
    std_fitness = np.std(fitness_values, axis=0)
    min_fitness = np.min(fitness_values, axis=0)
    max_fitness = np.max(fitness_values, axis=0)

    # Open the file in append mode
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file does not exist or is empty
        if not file_exists or file_is_empty:
            writer.writeheader()

        # Write data for the current generation
        row = {
            'Generation': gen,
            'Evaluations': len(pop),
            'Avg_Enemy_Health': avg_fitness[0],
            'Std_Enemy_Health': std_fitness[0],
            'Min_Enemy_Health': min_fitness[0],
            'Max_Enemy_Health': max_fitness[0],
            'Avg_Player_Health': avg_fitness[1],
            'Std_Player_Health': std_fitness[1],
            'Min_Player_Health': min_fitness[1],
            'Max_Player_Health': max_fitness[1],
            'Avg_Time': avg_fitness[2],
            'Std_Time': std_fitness[2],
            'Min_Time': min_fitness[2],
            'Max_Time': max_fitness[2]
        }
        writer.writerow(row)


def main():
    # Initialize the population
    pop = toolbox.population(n=POP_SIZE)

    # Run the genetic algorithm for a specified number of generations
    for gen in range(1, N_GEN + 1):
        print(f"Generation {gen}")

        # Evaluate the individuals with an initial fitness evaluation
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Log the generation statistics
        log_generation_statistics(
            pop=pop,
            gen=gen,
            logbook=None,  # Assuming you are not using a logbook for now
            pareto_front=None,  # Assuming no Pareto front at this point
            env=env,  # Pass the environment instance
            experiment_name=experiment_name
        )

# Load the corrupted CSV file (without headers)
df = pd.read_csv('experiment_nsga2/generation_statistics.csv', header=None)

# Assign correct headers manually
df.columns = [
    'Generation', 'Evaluations',
    'Avg_Enemy_Health', 'Std_Enemy_Health', 'Min_Enemy_Health', 'Max_Enemy_Health',
    'Avg_Player_Health', 'Std_Player_Health', 'Min_Player_Health', 'Max_Player_Health',
    'Avg_Time', 'Std_Time', 'Min_Time', 'Max_Time'
]

# Save the cleaned CSV file with proper headers
df.to_csv('experiment_nsga2/generation_statistics_cleaned.csv', index=False)

print("CSV file cleaned and saved as 'generation_statistics_cleaned.csv'")

if __name__ == "__main__":
    main()