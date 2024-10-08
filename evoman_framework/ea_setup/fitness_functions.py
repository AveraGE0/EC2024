"""Module to collect different fitness functions."""
import numpy as np


def default_fitness(self) -> float:
    """Default fitness function as provided in the environment class. Can be used to calculate
    the fitness against a single enemy.

    Returns:
        float: fitness score.
    """
    return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - np.log(self.get_time())


def mean_std_fitness(fitnesses: np.array, std_weight: float) -> float:
    """Function to calculate the fitness based on the mean and
    standard deviation of the fitness values. The reasoning is
    that if we penalize a high std in the fitness values we
    encourage learning strategies that work on all enemies.

    Args:
        sim_metrics (tuple): Metric returned by a simulation runs
        on multiple enemies. [fitnesses, player lives, enemy lives, times]
        std_weight (float): Weight of the standard deviation penalty

    Returns:
        float: Fitness as a single float (mean of fitness penalized by standard deviation).
    """
    return fitnesses.mean() - std_weight * fitnesses.std()


def clipped_harmonic_mean(values: np.ndarray) -> float:
    """Function to calculate the clipped harmonic mean of values.
    If this is used on the fitness scores, the lowest value will
    impact the final score the most.

    Args:
        values (np.ndarray): _description_

    Returns:
        float: _description_
    """
    return len(values) / np.sum(1.0 / np.where(values <= 0, 1e-7, values))