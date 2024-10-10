"""Module for population instantiation."""
import random
from deap import base, tools


def register_population(creator, toolbox: base.Toolbox, config: dict) -> None:
    """Function to register the population in the given toolbox. The population
    includes registering the individuals, individuals attributes and whole population.

    Args:
        creator (_type_): Creator used for creating classes.
        toolbox (base.Toolbox): Toolbox where everything is registered.
        config (dict): Config dict containing configuration parameters.
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMax,
        multi_gain=None,
        n_defeated=None,
        defeated=None,
        player_life=None,
        play_time=None
    )

    # individual + initialization
    toolbox.register(
        "attribute",
        random.uniform,
        config["init_low"],  # lower bound
        config["init_up"]  # upper bound
    )

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attribute,
        n=config["individual_size"]
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
