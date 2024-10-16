"""Module for setting values on individuals."""
import numpy as np
from ea.fitness_weighting import FitnessWeighting
from competition_metrics import multi_gain, defeated_enemies


def set_fitness_multi_objective(
    population: list[list],
    fitness_weighter: FitnessWeighting,
    fitness_sharing: callable,
) -> None:
    """Function to set the fitness (weighted!) on all individuals, given the population
    and the fitness weight scaler. The fitness weighter can produce any kind of aggregated
    values (also mean).

    Args:
        individuals (list[list]): Population of individuals.
        scheduled_weights (FitnessWeighting): Fitness weighting object.
    """
    for ind in population:
        ind.fitness.values = (fitness_weighter.get_weighted_fitness(ind.fitnesses),)

    if not fitness_sharing:
        return

    shared_fitnesses = fitness_sharing(population)
    for ind, shared_fitness in zip(population, shared_fitnesses):
        ind.fitness.values = (shared_fitness,)


def set_individual_properties(ind, metrics: dict[str, np.ndarray]) -> None:
    """Sets the properties of an individual given the metrics of a simulation
    run. The metrics are arrays of fitness, player life, enemy life, time,
    where each array contains one value for each enemy simulated.
    The fitness itself is not set since it might change over time.

    Args:
        ind (_type_): Individual.
        metrics (dict): Metrics dictionary (fitness, player_life, enemy_life, time).
        scheduled_weights (FitnessWeighting): Weighted adjusting the fitness average.
    """
    ind.multi_gain = multi_gain(metrics["player_life"], metrics["enemy_life"])
    ind.n_defeated = defeated_enemies(metrics["enemy_life"])
    ind.player_life = metrics["player_life"].sum()
    ind.play_time = metrics["time"].sum()
    ind.defeated = np.where(metrics["enemy_life"] == 0, 1, 0)
    ind.fitnesses = metrics["fitness"]


def enforce_individual_bounds(individual: list, lower_bound: float, upper_bound: float):
    """Function to enforce the maximal size for alleles. Clamps all values of offspring
    to the given range.

    Args:
        individual (list): Individual as list.
        lower_bound (float): Lower bound for single values in the list.
        upper_bound (float): Upper_bound for single values in the list.
    """
    clipped_values = np.clip(individual, a_min=lower_bound, a_max=upper_bound)
    individual[:] = clipped_values
