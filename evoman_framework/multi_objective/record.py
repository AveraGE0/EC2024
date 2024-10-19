import numpy as np
from pymoo.core.callback import Callback
from ea.fitness_sharing import mean_euclidean_distance


class RecordCallback(Callback):

    def __init__(self):
        super().__init__()
        self.data = []

    def notify(self, algorithm):
        # Save data for each generation
        generation_data = {
            "generation": algorithm.n_gen,
            "best_fitness": -np.min(np.mean(algorithm.pop.get("F"), axis=1)),
            "avg_fitness": -np.mean(algorithm.pop.get("F")),
            "std_fitness": np.std(algorithm.pop.get("F")),
            "avg_euclidean": np.mean(mean_euclidean_distance(algorithm.pop.get("X"))),
            "avg_defeated": np.mean(algorithm.pop.get("def")),
            "max_defeated": np.max(algorithm.pop.get("def"))
        }
        self.data.append(generation_data)
