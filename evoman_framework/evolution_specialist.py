"""File to run the neural evolution"""
import os
import random
import numpy as np
from tqdm import tqdm
from deap import base, creator, tools
from evoman.environment import Environment
from neural_controller import NeuralController


def main(toolbox, config: dict):
    """Main evolution function. Adjusted from:
    https://deap.readthedocs.io/en/master/overview.html [18.09.2024]

    Args:
        toolbox (_type_): _description_
        config (dict): _description_

    Returns:
        _type_: _description_
    """
    pop = toolbox.population(n=config["population_size"])
    n_generations = config["generations"]
    crossover_prob = config["p_crossover"]
    mutation_prob = config["p_mutation"]

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    max_fitness = max([ind.fitness.values for ind in pop])
    pbar = tqdm(range(n_generations))
    for g in pbar:
        # Select the next generation individuals (half of population)
        offspring = toolbox.select(pop, len(pop)//2)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The bottom population is entirely replaced by the offspring
        pop.sort(reverse=True, key=lambda x: x.fitness.values)
        pop[-len(offspring):] = offspring

        max_fitness = max(max_fitness, *[ind.fitness.values for ind in pop])
        pbar.set_postfix(message=f"current max fitness: {float(max_fitness[0])}")

    return pop


if __name__ == '__main__':
    # create directory
    EXPERIMENT_NAME = "../experiments/test_evo"

    # initialize directories for running the experiment
    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    nc = NeuralController(n_inputs=20, n_outputs=5, hidden_size=5)
    env = Environment(
        experiment_name=EXPERIMENT_NAME,  # this is actually a path!
        multiplemode="no",
        enemies=[1],
        player_controller=nc,
        visuals=False,
    )

    def evaluate(individual: list):
        np.random.seed(42)
        default_fitness, p_life, e_life, time = env.play(
            pcont=individual
        )  # pcont is actually the genome (bad naming)
        return default_fitness,

    config = {
        "population_size": 200,
        "individual_size": nc.get_genome_size(),
        "generations": 10,
        "p_crossover": 0.5,
        "p_mutation": 0.2,
    }
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # individuals + initialization
    toolbox = base.Toolbox()
    toolbox.register("attribute", lambda: random.uniform(-10, 10))  # allow positive and negative values
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attribute, n=config["individual_size"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # mating, mutation selection (for mating and mutation), evaluation function
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    final_population = main(toolbox, config)

    # sort descending
    final_population.sort(reverse=True, key=lambda x: x.fitness.values)
    best_individual = final_population[0]
    #np.random.seed(42)
    env.visuals = True
    env.speed = "normal"

    np.random.seed(42)
    final_fitness, *_ = env.play(pcont=best_individual)
    print(f"final fitness: {final_fitness}")
