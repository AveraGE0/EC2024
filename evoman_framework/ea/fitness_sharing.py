"""Module for fitness sharing (and distance calculation)."""
import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean


def mean_euclidean_distance(population_genotypes: list[list]) -> float:
    """Calculates the mean pairwise euclidean distances of a population.

    Args:
        population_genotypes (list[list]): Population.

    Returns:
        float: Mean euclidean distance.
    """
    # Compute pairwise Euclidean distances
    pairwise_distances = pdist(population_genotypes, metric='euclidean')

    # Calculate the mean Euclidean distance
    mean_distance = np.mean(pairwise_distances)

    return mean_distance


def mean_hamming_distance(population_genotypes: list[list], threshold=0.5) -> float:
    """_summary_

    Args:
        population_genotypes (list[list]): _description_
        threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        float: _description_
    """
    # Compute pairwise Hamming distances with a threshold using pdist
    pairwise_hamming_distances = pdist(
        population_genotypes,
        metric=lambda u, v: hamming(u, v, threshold=threshold)
    )
    # Calculate the mean Hamming distance (as the raw count of mismatches)
    return np.mean(pairwise_hamming_distances)


def hamming(u: np.ndarray, v: np.ndarray, threshold: int = 0.5) -> int:
    """Function to calculate the hamming distance for continuous
    vectors (arrays).

    Args:
        u (np.ndarray): Individual 1
        v (np.ndarray): Individual 2
        threshold (float, optional): Difference for determining if two values are
                                     different. Defaults to 0.5.

    Returns:
        int: Hamming distance: number of positions where the difference is greater
             than the given threshold.
    """
    return np.sum(np.abs(u - v) > threshold)


def same_loss(u: np.ndarray, v: np.ndarray) -> int:
    """Function to calculate a distance based on the number of enemies that
    are not defeated by both given individuals (u and v). The more enemies
    both defeat the larger their distance will be. Also if they dont lose
    against the same enemy, they will be further apart.

    Args:
        u (np.ndarray): Array of indicating which enemies Individual 1
                        won against.
        v (np.ndarray): Array of indicating which enemies Individual 2
                        won against.

    Returns:
        int: The distance as number of total enemies (8) minus the amount
             of enemies both lose against.
    """
    return int(8 - np.where((u + v)==0, 1, 0).sum())


def compute_distance_matrix(
        population_genotypes: list[list],
        distance_func: callable,
        distance_property: str = None,
        distance_parameters: dict =None
    ) -> np.ndarray:
    """Function to calculate the pairwise distances (matrix) using a given distance function.
    The distance function receives two properties of individuals (can be the genomes) and
    returns a distance value. The distance_parameters parameter allows to pass any keyword
    arguments to this distance function. The distance property choses the property that is used.
    If left empty (None), it will default to using the genome.

    Args:
        population_genotypes (list[list]): Population where distance is calculated.
        distance_func (callable): Function calculating the distance.
        distance_parameters (dict, optional): Keyword arguments for the distance function.
                                              Defaults to None.

    Returns:
        np.ndarray: Distance matrix using given property, population and distance function.
    """
    if not distance_parameters:
        distance_parameters = {}

    population_genotypes = list(map(
        lambda x: getattr(x, distance_property) if distance_property != "genotype" else x,
        population_genotypes
    ))

    pairwise_distances = pdist(
        population_genotypes,
        metric=lambda u, v: distance_func(u, v, **distance_parameters)
    )
    # Returning a matrix
    return squareform(pairwise_distances)


def share_fitness(
        population: list[list],
        sigma: float,
        distance_func: callable,
        distance_property: str = None
    ) -> list[float]:
    """Function to perform fitness sharing on a population.

    Args:
        population (list[list]): Population.
        sigma (float): Minimal distance to share fitness.
        distance_func (callable): Function to calculate the fitness.
        distance_property (str, optional): Property used for distance calculation
        Can be None (genotype), defeated, etc. Defaults to None.

    Returns:
        list[float]: shared fitness values for each of the individuals in the population.
    """
    shared_fit = []
    dist_matrix = compute_distance_matrix(population, distance_func, distance_property)

    for i, ind_i in enumerate(population):
        neighbors_fitness_sum = 0
        for j, _ in enumerate(population):
            if dist_matrix[i,j] <= sigma:
                neighbors_fitness_sum += 1 - (dist_matrix[i,j] / sigma)

        shared_fit.append(ind_i.fitness.values[0]/neighbors_fitness_sum)

    return shared_fit
