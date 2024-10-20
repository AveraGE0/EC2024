import numpy as np
from pymoo.core.problem import Problem
from parallel_environment import ParallelEnvironment
from competition_metrics import multi_gain, defeated_enemies


class GeneralistProblem(Problem):

    def __init__(
            self,
            par_environment: ParallelEnvironment,
            individual_size=265,
            n_obj=8,
            allele_lower_bound=-10,
            allele_upper_bound=10
        ):
        super().__init__(n_var=individual_size, n_obj=n_obj, n_ieq_constr=0, xl=allele_lower_bound, xu=allele_upper_bound)
        self.indidvidual_size = individual_size
        self.par_environment = par_environment

    def _evaluate(self, x, out, *args, **kwargs):
        """Function to evaluate.

        Args:
            x (np.ndarray): 2-D numpy array containing all population.
            out (dict): Out dict, contains fitness.
        """
        results_dict = self.par_environment.get_results(x)
        results = np.array([-res["fitness"] for res in results_dict])  # np.array([-res["fitness"] for res in results_dict])
        defeated = np.array([defeated_enemies(res["enemy_life"]) for res in results_dict])
        out["F"] = results
        out["def"] = defeated
