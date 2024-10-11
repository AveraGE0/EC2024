import numpy as np
from pymoo.core.sampling import Sampling


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return np.random.uniform(-1, 1, size=(n_samples, 265))
