import numpy as np
import random
import os
from deap import base, creator, tools
from environment import Environment
from demo_controller import player_controller

# Define the experiment name and create the directory if it doesn't exist
experiment_name = 'experiment_nsga2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Number of hidden neurons in the neural network controller
n_hidden_neurons = 10  # Adjust as needed

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

# Neural Network parameters
num_inputs = 20                # Number of inputs (depends on the game)
num_outputs = 5                # Number of outputs (actions)
n_hidden_neurons = 10          # Number of hidden neurons in the neural network

# Genetic Algorithm parameters
POP_SIZE = 200      # Population size
N_GEN = 20          # Number of generations
CXPB = 0.9          # Crossover probability
MUTPB = 0.1         # Mutation probability

from deap import creator

# Define a multi-objective fitness function (minimize all objectives)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))

# Create the Individual class based on list with the defined fitness function
creator.create("Individual", list, fitness=creator.FitnessMulti)


toolbox = base.Toolbox()

# Attribute generator: Each gene is a float between -1 and 1
toolbox.register("attr_float", random.uniform, -1, 1)

# Structure initializers: Define 'individual' and 'population'
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    """
    Evaluates an individual by playing the game and returning the objectives.
    Objectives:
    - Minimize enemy life (i.e., maximize damage to enemy)
    - Maximize player life (minimize negative player life)
    - Minimize time spent
    """
    # Play the game with the individual's weights
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    # Return the objectives (enemy life, negative player life, time)
    return enemy_life, -player_life, time

toolbox.register("evaluate", evaluate)

# Crossover operator: Simulated Binary Crossover (SBX)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=15)

# Mutation operator: Polynomial mutation
toolbox.register("mutate", tools.mutPolynomialBounded, low=-1.0, up=1.0, eta=15, indpb=1.0/n_weights)

# Selection operator: NSGA-II selection
toolbox.register("select", tools.selNSGA2)

def main():
    random.seed()  # Seed the random number generator for reproducibility

    # Initialize population
    pop = toolbox.population(n=POP_SIZE)
    # Assign crowding distance and rank to individuals (necessary for NSGA-II)
    pop = toolbox.select(pop, len(pop))

    # Prepare the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: np.mean(fits, axis=0))
    stats.register("std", lambda fits: np.std(fits, axis=0))
    stats.register("min", lambda fits: np.min(fits, axis=0))
    stats.register("max", lambda fits: np.max(fits, axis=0))

    # Logbook to record statistics
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # Evaluate the initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    print(f"Evaluating {len(invalid_ind)} individuals in the initial population...")
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Record statistics for initial population
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, N_GEN + 1):
        # Selection (parents)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        for ind in offspring:
            if random.random() <= MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print(f"Evaluating {len(invalid_ind)} individuals in generation {gen}...")
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Combine the parent and offspring population
        pop = toolbox.select(pop + offspring, POP_SIZE)

        # Record statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

    # Save statistics to file
    with open(os.path.join(experiment_name, "statistics.txt"), "w") as f:
        for entry in logbook:
            f.write(str(entry) + '\n')

    # Extract the Pareto front
    pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

    # Save the Pareto front
    with open(os.path.join(experiment_name, "pareto_front.txt"), "w") as f:
        for ind in pareto_front:
            f.write(f"{ind.fitness.values}\t{list(ind)}\n")

if __name__ == "__main__":
    main()


