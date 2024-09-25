import graphviz
import matplotlib.pyplot as plt
from evoman.environment import Environment
from neural_controller import NeuralController
import numpy as np
import os


def show_run(individual: list, enemies: list, config: dict):
    nc = NeuralController(
        n_inputs=config["n_inputs"],
        n_outputs=config["n_outputs"],
        hidden_size=config["hidden_size"]
    )

    # add correct individual size to config
    config["individual_size"] = nc.get_genome_size()

    EXPERIMENT_NAME ="../experiments/test"

    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    env = Environment(
        experiment_name=EXPERIMENT_NAME,  # this is actually a path!
        multiplemode="no",
        visuals = True,
        speed = "normal",
        enemies=[enemies[0]],
        player_controller=nc,
        level=2,
    )
    np.random.seed(42)

    for enemy in enemies:
        env.enemies = [enemy]
        final_fitness, p_energy, e_energy, _ = env.play(pcont=individual)
        print(f"Enemy {enemy}")
        print(f"final fitness: {final_fitness}")
        print(f"final_gain: {p_energy - e_energy}")
