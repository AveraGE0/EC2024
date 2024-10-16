"""Small script for visualizing an individuals (NN) behavior with Evoman."""
import os
import pickle
import yaml
import numpy as np
from evoman.environment import Environment
from neural_controller import NeuralController


def show_run(individual: list, enemies: list, config: dict, speed="fastest"):
    """Function to show a run of a specific individual given a config (the config it was
    trained on).

    Args:
        individual (list): Individual (NN weights) that is simulated.
        enemies (list): Enemies that should be played against.
        config (dict): Configuration of the run.
        speed (str, optional): Simulation speed. Should be "normal" or "fastest".
                               Defaults to "fastest".
    """
    if speed not in ["fastest", "normal"]:
        raise ValueError(
            "Got wrong speed setting, should be 'fastest' or 'normal'"\
            f"but was: {speed}!"
        )

    nc = NeuralController(
        n_inputs=config["n_inputs"],
        n_outputs=config["n_outputs"],
        hidden_size=config["hidden_size"]
    )

    # add correct individual size to config
    config["individual_size"] = nc.get_genome_size()
    # dummy directory
    experiment_name ="../experiments/test"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = Environment(
        experiment_name=experiment_name,
        multiplemode="no",
        visuals=True,
        speed=speed,
        enemies=[enemies[0]],
        player_controller=nc,
        level=2,
    )

    np.random.seed(42)

    won = 0
    total_gain = 0
    total_fitness = 0

    for enemy in enemies:
        env.enemies = [enemy]
        final_fitness, p_energy, e_energy, _ = env.play(pcont=individual)

        total_fitness += final_fitness
        total_gain = total_gain + p_energy - e_energy
        won += int(e_energy == 0)

        print(
            f"Enemy {enemy}: fitness: {round(final_fitness, 3)},\t gain: {round(p_energy - e_energy, 3)},"\
            f"\t defeated: {'yes' if e_energy == 0 else 'no'}"
        )
    print(
        f"Total won: {won},"\
        f" total gain: {round(total_gain, 3)},"\
        f" avg fitness: {round(total_fitness/8, 3)}"
    )


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    RUN_DIR = "..\\experiments\\competition_test\\"

    with open(os.path.join(RUN_DIR, "best_individual_multi_gain.pkl"), mode="rb") as ind_file:
        ind = pickle.load(ind_file)

    with open(os.path.join(RUN_DIR, "config.yaml"), mode="r", encoding="utf-8") as config_file:
        ind_config = yaml.safe_load(config_file)

    show_run(ind, list(range(1, 9)), ind_config, speed="normal")
