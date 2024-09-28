import os
import pickle
import numpy as np
from evoman.environment import Environment
from neural_controller import NeuralController


def get_gain(env, enemy: int, individuals: list, n_runs=5) -> list:
    np.random.seed(42)

    env.enemies = [enemy]
    gain = np.zeros(shape=(len(individuals), n_runs))
    for i_ind, individual in enumerate(individuals):
        for run in range(n_runs):
            default_fitness, p_life, e_life, time = env.play(
                pcont=individual
            )
            gain[i_ind, run] = p_life-e_life
    return gain.mean(axis=1)


def get_gain_values(env, algorithms: dict, experiment_base_path: str, repeat=5):
    gains = {}

    for alg_name, enemies in algorithms.items():
        gains[alg_name] = {}
        for enemy, base_name in enemies.items():
            experiment_paths = [os.path.join(experiment_base_path, path) for path in os.listdir(experiment_base_path) if base_name in path]

            individuals = []
            for experiment_path in experiment_paths:
                with open(os.path.join(experiment_path, "best_gain_individual.pkl"), mode="rb") as individual_file:
                    individuals.append(pickle.load(individual_file))
            gains[alg_name][enemy] = get_gain(env, enemy=int(enemy[-1]), individuals=individuals, n_runs=repeat)
    return gains
    

if __name__ == '__main__':
    EXPERIMENT_NAME = "../experiments/test"

    # initialize directories for running the experiment
    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    config = {}
    config["n_inputs"] = 20
    n_outputs=config["n_outputs"] = 5
    hidden_size=config["hidden_size"] = 5

    nc = NeuralController(
        n_inputs=config["n_inputs"],
        n_outputs=config["n_outputs"],
        hidden_size=config["hidden_size"]
    )
    # add correct individual size to config
    config["individual_size"] = nc.get_genome_size()

    env = Environment(
        experiment_name=EXPERIMENT_NAME,  # this is actually a path!
        multiplemode="no",
        enemies=[2],
        player_controller=nc,
        visuals=False,
        level=2,
    )

    algorithms = {
        "high_g":{
            "enemy2": "test_final_high_g_enemy=2_25092024_183036",
            "enemy5": "test_final_high_g_enemy=5_25092024_182952"
        }
    }
        
    experiment_base_path = "../experiments"
    
    gains = get_gain_values(env, algorithms, experiment_base_path, repeat=5)
    print(gains)
    
    