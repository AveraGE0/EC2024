import pickle
import numpy as np
from pymoo.core.sampling import Sampling


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return np.random.uniform(-1, 1, size=(n_samples, 265))


class PreTrainedPopulation(Sampling):
    def __init__(self, pop_path: str) -> None:
        super().__init__()
        self.pop_path = pop_path

    def _do(self, problem, n_samples, **kwargs):
        with open(self.pop_path, mode="rb") as p_file:
            population = pickle.load(p_file)

        total_population = np.empty(shape=(0, 265))
        for island in population:
            total_population = np.concatenate([total_population, island])

        if n_samples > len(total_population):
            raise ValueError(
                f"Initial population is too small ({len(population)}), needed: {n_samples}"
            )
        
        return total_population
