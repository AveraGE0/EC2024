import numpy as np
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output


class CustomOutput(Output):

    def __init__(self):
        super().__init__()
        self.f_mean = Column("Fitness AVG", width=10)
        self.f_std = Column("Fitness STD", width=10)
        self.f_max = Column("Fitness MAX", width=10)
        self.columns += [self.f_mean, self.f_max, self.f_std]

    def update(self, algorithm):
        super().update(algorithm)
        self.f_mean.set(-np.mean(algorithm.pop.get("F")))
        self.f_max.set(-np.min(np.mean(algorithm.pop.get("F"), axis=1)))
        self.f_std.set(np.std(algorithm.pop.get("F")))
