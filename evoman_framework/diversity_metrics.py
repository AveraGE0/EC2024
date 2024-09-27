import numpy as np
from scipy.spatial.distance import pdist, squareform
import random 


def euclidean_distance(population_genotypes: list[list]) -> float:
    # Compute pairwise Euclidean distances
    pairwise_distances = pdist(population_genotypes, metric='euclidean')

    # Calculate the mean Euclidean distance
    mean_distance = np.mean(pairwise_distances)

    return mean_distance


def hamming_distance(population_genotypes: list[list], threshold=0.5) -> float:
    # Define a custom Hamming distance with a threshold (without normalization)
    def hamming_threshold(u, v, threshold=0.1):
        # Count the number of positions where the absolute difference exceeds the threshold
        return np.sum(np.abs(u - v) > threshold)

    # Compute pairwise Hamming distances with a threshold using pdist
    pairwise_hamming_distances = pdist(population_genotypes, metric=lambda u, v: hamming_threshold(u, v, threshold=threshold))

    # Calculate the mean Hamming distance (as the raw count of mismatches)
    return np.mean(pairwise_hamming_distances)

def hamming_distance_mat(population_genotypes: list[list], threshold=0.5):
    # Define a custom Hamming distance with a threshold (without normalization)
    def hamming_threshold(u, v, threshold=0.1):
        # Count the number of positions where the absolute difference exceeds the threshold
        return np.sum(np.abs(u - v) > threshold)

    # Compute pairwise Hamming distances with a threshold using pdist
    pairwise_hamming_distances = pdist(population_genotypes, metric=lambda u, v: hamming_threshold(u, v, threshold=threshold))

    # Returning a matrix
    return squareform(pairwise_hamming_distances)

def fitness_sharing(individuals, k, tournsize, sigma):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    shared_fit = []
    ham_mat = hamming_distance_mat(individuals)

    for i in range(len(individuals)):
        distance = 0
        for j in range(len(individuals)):
            if ham_mat[i,j] <= sigma:
                distance += 1-ham_mat[i,j]/sigma
        shared_fit.append(individuals[i].fitness.values[0]/distance)
    
    chosen = []
    fit_zipped = list(zip(shared_fit, individuals))
    for i in range(k):
        aspirants = random.choices(fit_zipped, k=tournsize)
        chosen.append(max(aspirants, key=lambda x:x[0])[1])
    return chosen