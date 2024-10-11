import yaml
import os
import numpy as np
from multi_objective.output import MyOutput
from parallel_environment import ParallelEnvironment

from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize

from multi_objective.problem import MyProblem
from multi_objective.sampling import MySampling
from pymoo.visualization.scatter import Scatter

from ea.fitness_functions import default_fitness
from neural_controller import NeuralController

from visualize import show_run


if __name__ == "__main__":
    #np.set_printoptions(precision=3, suppress=True)
    CONFIG_NAME = "config_competition_test.yaml"

    # Load the configuration from a YAML file
    with open(f"../{CONFIG_NAME}", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    experiment_name = os.path.join("../experiments/", config["name"])

    # initialize directories for running the experiment
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    nc = NeuralController(
        n_inputs=config["n_inputs"],
        n_outputs=config["n_outputs"],
        hidden_size=config["hidden_size"]
    )
    # add correct individual size to config
    config["individual_size"] = nc.get_genome_size()

    par_env = ParallelEnvironment(15, config, default_fitness)
    par_env.start_processes()

    print("started!")

    algorithm = SMSEMOA(
        pop_size=400,
        sampling=MySampling(),

    )

    my_problem = MyProblem(par_env)

    res = minimize(
        my_problem,
        algorithm,
        ('n_gen', 400),
        seed=42,
        output=MyOutput(),
        verbose=True
    )

    print(f"Pareto optimal solutions: {res.X.shape}")
    print(f"Best fitness given by framework (sum={res.f.sum()}):{res.f}")

    i_best = res.F.sum(axis=1).argmin()
    best = res.X[i_best]
    print(f"Best fitness in pareto front (sum={res.F[i_best].sum()}): {res.F[i_best]}")
    show_run(best, [1, 2, 3, 4, 5, 6, 7, 8], config)
