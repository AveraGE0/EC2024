import numpy as np
import random
import os
from deap import base, creator, tools
from evoman.environment import Environment
from demo_controller import player_controller
import pandas as pd
import math

# Define the experiment name and create the directory if it doesn't exist
experiment_name = 'experiment_nsga2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Neural Network parameters
num_inputs = 20
num_outputs = 5
n_hidden_neurons = 10

# Calculate the number of weights in the neural network
num_weights = (
        (num_inputs * n_hidden_neurons) +
        (n_hidden_neurons * num_outputs) +
        n_hidden_neurons +
        num_outputs
)

# Initialize the environment for multiple enemies
env = Environment(
    experiment_name=experiment_name,
    enemies=[2, 4, 7],
    multiplemode="yes",
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    contacthurt='player',
    speed="fastest"
)

# Genetic Algorithm parameters
POP_SIZE = 50
N_GEN = 10
CXPB = 0.9
MUTPB = 0.1

# Define a multi-objective fitness function (minimize all objectives)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Evaluation function
def evaluate(individual):
    """
    Evaluates the individual in the environment and returns the fitness.
    The fitness is calculated as: player_life - enemy_life - log(time)
    """
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))

    # Calculate gain (player life - enemy life)
    total_enemy_life = sum(enemy_life) if isinstance(enemy_life, list) else enemy_life
    gain = player_life - total_enemy_life

    # Logarithm of time (handling any zero-time edge cases)
    log_time = math.log(time) if time > 0 else 0

    # Calculate overall fitness as: player_life - enemy_life - log(time)
    overall_fitness = player_life - total_enemy_life - log_time

    # Return the three objectives for the multi-objective optimization (enemy life, -player life, log time)
    return total_enemy_life, -player_life, log_time


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=15)
toolbox.register("mutate", tools.mutPolynomialBounded, low=-1.0, up=1.0, eta=15, indpb=1.0 / num_weights)
toolbox.register("select", tools.selNSGA2)


# Configure statistics
def configure_statistics():
    """
    This function configures which statistics to track.
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    return stats


# Initialize logbook
def initialize_logbook():
    """
    Initializes the logbook with headers for tracking generations.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'avg_gain', 'avg_log_time', 'avg_fitness'] + ['avg_enemy_life',
                                                                                     'avg_player_life',
                                                                                     'std_enemy_life',
                                                                                     'std_player_life'] + [
                         'min_enemy_life', 'min_player_life', 'max_enemy_life', 'max_player_life']
    return logbook


# Update the logbook after each generation
def update_logbook(logbook, gen, invalid_ind, population, stats):
    """
    Updates the logbook with statistics for the current generation.
    """
    # Gather statistics for enemy life, player life, log time, gain, and fitness
    gains = [ind.fitness.values[3] for ind in population]
    log_times = [ind.fitness.values[2] for ind in population]
    fitnesses = [ind.fitness.values[4] for ind in population]

    avg_gain = np.mean(gains)
    avg_log_time = np.mean(log_times)
    avg_fitness = np.mean(fitnesses)

    # Compile stats for enemy life and player life
    record = stats.compile(population)

    # Log the results for this generation
    logbook.record(gen=gen, nevals=len(invalid_ind), avg_gain=avg_gain, avg_log_time=avg_log_time,
                   avg_fitness=avg_fitness, **record)
    print(logbook.stream)


# Export logbook to CSV
def export_logbook_to_csv(logbook, experiment_name):
    """
    Exports the logbook data to a CSV file after the experiment is completed.
    """
    filepath = os.path.join(experiment_name, "logbook_statistics.csv")
    df = pd.DataFrame(logbook)  # Convert logbook to pandas DataFrame
    df.to_csv(filepath, index=False)
    print(f"Logbook data saved to {filepath}")


# Main genetic algorithm function
def main():
    # Initialize the population
    pop = toolbox.population(n=POP_SIZE)

    # Initialize logbook and Pareto front
    stats = configure_statistics()  # Configure the statistics to track
    logbook = initialize_logbook()
    pareto_front = tools.ParetoFront()

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

        # Update the Pareto front with current population
        pareto_front.update(pop)

        # Gather statistics and update the logbook
        update_logbook(logbook, gen, invalid_ind, pop, stats)

    # Export logbook data to CSV at the end of the run
    export_logbook_to_csv(logbook, experiment_name)


if __name__ == "__main__":
    main()
