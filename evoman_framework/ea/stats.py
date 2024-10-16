"""Module containing utility for the stats setup."""
import numpy as np
from deap import tools
from ea.fitness_sharing import mean_euclidean_distance, mean_hamming_distance


def create_population_stats() -> tools.MultiStatistics:
    """Function to create the stats recorded during a run of the EA.

    Returns:
        tools.MultiStatistics: MultiStatistics used in the run.
    """
    fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    fitness_stats.register("avg", np.mean)
    fitness_stats.register("std", np.std)
    fitness_stats.register("max", np.max)

    raw_fitness_stats = tools.Statistics(key=lambda ind: ind.fitnesses)
    raw_fitness_stats.register("avg", np.mean, axis=0)
    raw_fitness_stats.register("std", np.std, axis=0)
    raw_fitness_stats.register("max", np.max, axis=0)

    gain_stats = tools.Statistics(key=lambda ind: ind.multi_gain)
    gain_stats.register("avg_sum", np.mean)
    gain_stats.register("max_sum", np.max)

    defeated_stats = tools.Statistics(key=lambda ind: ind.n_defeated)
    defeated_stats.register("avg", np.mean)
    defeated_stats.register("max", np.max)

    life_stats = tools.Statistics(key=lambda ind: ind.player_life)
    life_stats.register("avg_sum", np.mean)
    life_stats.register("max_sum", np.max)

    time_stats = tools.Statistics(key=lambda ind: ind.play_time)
    time_stats.register("avg_sum", np.mean)
    time_stats.register("min_sum", np.min)

    enemies_defeated_stats = tools.Statistics(key=lambda ind: ind.defeated)
    enemies_defeated_stats.register("avg_def", np.mean, axis=0)
    enemies_defeated_stats.register("is_def", np.max, axis=0)

    return tools.MultiStatistics(
        fitness=fitness_stats,
        fitnesses=raw_fitness_stats,
        gain=gain_stats,
        defeated=defeated_stats,
        life=life_stats,
        time=time_stats,
        enemies=enemies_defeated_stats
    )


def create_island_stats() -> tools.MultiStatistics:
    """Function to create the stats recorded during a run of the EA.

    Returns:
        tools.MultiStatistics: MultiStatistics used in the run.
    """
    fitness_stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    fitness_stats.register("avg", np.mean)
    fitness_stats.register("std", np.std)
    fitness_stats.register("max", np.max)

    gain_stats = tools.Statistics(key=lambda ind: ind.multi_gain)
    gain_stats.register("avg_sum", np.mean)
    gain_stats.register("max_sum", np.max)

    defeated_stats = tools.Statistics(key=lambda ind: ind.n_defeated)
    defeated_stats.register("avg", np.mean)
    defeated_stats.register("max", np.max)

    life_stats = tools.Statistics(key=lambda ind: ind.player_life)
    life_stats.register("avg_sum", np.mean)
    life_stats.register("max_sum", np.max)

    time_stats = tools.Statistics(key=lambda ind: ind.play_time)
    time_stats.register("avg_sum", np.mean)
    time_stats.register("min_sum", np.min)

    enemies_defeated_stats = tools.Statistics(key=lambda ind: ind.defeated)
    enemies_defeated_stats.register("avg_def", np.mean, axis=0)
    enemies_defeated_stats.register("is_def", np.max, axis=0)

    # Register custom diversity statistics on the genotypes
    diversity_stats = tools.Statistics()
    diversity_stats.register("euclidean_avg", mean_euclidean_distance)
    diversity_stats.register("hamming", mean_hamming_distance)

    return tools.MultiStatistics(
        fitness=fitness_stats,
        gain=gain_stats,
        defeated=defeated_stats,
        life=life_stats,
        time=time_stats,
        enemies=enemies_defeated_stats,
        diversity_stats=diversity_stats
    )
