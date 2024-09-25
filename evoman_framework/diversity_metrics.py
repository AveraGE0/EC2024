import numpy as np
from scipy.spatial.distance import pdist, squareform


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