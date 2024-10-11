import numpy as np
from pymoo.core.problem import Problem
from parallel_environment import ParallelEnvironment


class MyProblem(Problem):

    def __init__(self, par_environment: ParallelEnvironment, individual_size=265):
        super().__init__(n_var=265, n_obj=8, n_ieq_constr=0, xl=-10, xu=10)
        self.indidvidual_size = individual_size
        self.weight_bounds = (-10, 10)
        self.par_environment = par_environment

    def _evaluate(self, x, out, *args, **kwargs):
        """Function to evaluate.

        Args:
            x (np.ndarray): 2-D numpy array containing all population.
            out (dict): Out dict, contains fitness.
        """
        results_dict = self.par_environment.get_results(x)
        results = np.array([-res["fitness"] for res in results_dict])
        out["F"] = results