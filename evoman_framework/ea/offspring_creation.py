"""Module to setup the offspring creation."""
import random
import numpy as np
from deap import base, tools


def create_offspring(
        population: list,
        toolbox: base.Toolbox,
        crossover_prob: float,
        mutation_prob: float
    ) -> tuple[list, list]:
    """Function to handle offspring creation. Will return a list with new individuals
    which have no fitness yet. This list can be used for parallel computation to
    distribute the individuals to evaluate on processes evenly. Also returns the
    offspring for the given population.

    Args:
        population (list): Population for which offspring is created (can be island).
        toolbox (base.Toolbox): Toolbox used for offspring creation.
        crossover_prob (float): Probability for an individual to be crossed over.
        mutation_prob (float): Probability for an individual to be mutated.

    Returns:
        tuple[list, list]: individuals with undetermined fitness, offspring which can bse used for
                           the next generation.
    """
    offspring = toolbox.select_parents(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_prob:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutation_prob:
            toolbox.mutate(np.clip(mutant, -10, 10))
            del mutant.fitness.values

    undetermined_fitness_individuals = [ind for ind in offspring if not ind.fitness.valid]

    for ind in undetermined_fitness_individuals:
        toolbox.constrain_individual(ind)

    return undetermined_fitness_individuals, offspring


def register_recombination(toolbox: base.Toolbox, config: dict) -> None:
    """Function to register recombination functions for the run.

    Args:
        toolbox (base.Toolbox): Toolbox used.
        config (dict): Config dict containing configuration parameters.
    """
    toolbox.register(
        "mate",
        tools.cxSimulatedBinary,
        eta=config["SBX_eta"]
    )

    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=config["polynomial_eta"],
        low=config["polynomial_low"],
        up=config["polynomial_up"],
        indpb=config["polynomial_indpb"]
    )
