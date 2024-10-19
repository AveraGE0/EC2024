import yaml
import os
import numpy as np
import time
from multi_objective.output import CustomOutput
from parallel_environment import ParallelEnvironment
import pickle

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize

from multi_objective.problem import GeneralistProblem
from multi_objective.sampling import RandomSampling, PreTrainedPopulation
from pymoo.visualization.scatter import Scatter

from ea.fitness_functions import default_fitness
from neural_controller import NeuralController
from multi_objective.record import RecordCallback


def run_experiment(config: dict):
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

    par_env = ParallelEnvironment(config["n_processes"], config, default_fitness)
    par_env.start_processes()

    nsga2 = NSGA2(
        pop_size=config["population_size"],
        sampling=RandomSampling(lower=config["init_low"], upper=config["init_up"]),
    )

    # Instantiate the callback
    callback = RecordCallback()

    generalist = GeneralistProblem(par_env, n_obj=len(config["train_enemy"]))

    res = minimize(
        generalist,
        nsga2,
        ('n_gen', config["generations"]),
        seed=42,
        output=CustomOutput(),
        verbose=True,
        callback=callback
    )
    
    with open(os.path.join(experiment_name, "config.yaml"), "w", encoding="utf-8") as c_file:
        yaml.dump(config, c_file, default_flow_style=False)

    with open(os.path.join(experiment_name, "logs.pkl"), "wb") as l_file:
        pickle.dump(callback.data, l_file,)

    # Saving the best solution (res.X gives you the decision variables of the best solution)
    best_solution = res.X
    best_fitness = res.F
    np.savetxt(os.path.join(experiment_name, "best_fitness_solution.csv"), np.hstack((best_solution, best_fitness)), delimiter=",", header="X1,X2,...,Fitness")

    # Saving the entire population
    population = res.pop.get("X")  # Decision variables of the entire population
    population_fitness = res.pop.get("F")  # Fitness values of the entire population
    np.savetxt(os.path.join(experiment_name, "population.csv"), np.hstack((population, population_fitness)), delimiter=",", header="X1,X2,...,Fitness")

    i_best = res.F.sum(axis=1).argmin()
    best = res.X[i_best]
    np.savetxt(os.path.join(experiment_name, "best_gain_solution.csv"), best, delimiter=",")

    par_env.stop_processes()


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    CONFIG_NAME = "config_nsga_all.yaml"
    # Load the configuration from a YAML file
    with open(f"../{CONFIG_NAME}", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # run experiment
    for i_run, seed in zip(list(range(config['n_repetitions'])), config['seeds']):
        config["name"] = CONFIG_NAME.split(".")[0] + f"_{i_run}"
        config["seed"] = seed
        run_experiment(config)
        # let cpu cool down
        time.sleep(1)
