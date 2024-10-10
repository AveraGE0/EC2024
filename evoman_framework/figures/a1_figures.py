"""Module to generate the plots for Assignment 1."""
import os
import pickle
import matplotlib.pyplot as plt
from neural_controller import NeuralController
from evoman.environment import Environment
from figures.plots import multirun_plots, multirun_plots_diversity, plot_final
from gain import get_gain_values


if __name__ == "__main__":
    base_names = {
        "lphg": {   
            "enemy2": "high_g_enemy=2_27092024_170302",
            "enemy5": "high_g_enemy=5_27092024_172657",
            "enemy7": "high_g_enemy=7_27092024_180456"
        },
        "hplg": {
            "enemy2": "low_g_enemy=2_27092024_183050",
            "enemy5": "low_g_enemy=5_27092024_185837",
            "enemy7": "low_g_enemy=7_27092024_194355"
        }
    }
    # wheat and cadetblue for the first configuration
    # orange and red for the second configuration
    colors = [
        {
            "max": "#B8860B",
            "avg": "#5F9EA0",
            "std": "#5F9EA0",
            "euclidean": "#F5DEB3",
            "hamming": "#2F4F4F"
        },
        {
            "max": "#FFA500",
            "avg": "#FF4500",
            "std": "#FF4500",
            "euclidean": "#800080",
            "hamming": "#FFA500"
        }
    ]
    experiment_base_path = "../experiments"
    plots_path = "../summary_plots"

    # replace each base name with all logs of the run with that basename
    logs = {}

    for name, enemies in base_names.items():
        logs[name] = {}
        for enemy, base_name in enemies.items():
            logs[name].update({enemy: []})
            # load all runs of the algorithm on specific enemy
            experiment_files = list(filter(
                lambda f_name: base_name in f_name,
                os.listdir(experiment_base_path)
            ))
            experiment_paths = [
                os.path.join(experiment_base_path, path) for path in experiment_files
            ]
            # append all logs to the algorithm, enemy combination
            for experiment_path in experiment_paths:
                with open(os.path.join(experiment_path, "logbook.pkl"), mode="rb") as log_file:
                    logs[name][enemy].append(pickle.load(log_file))

    for enemy in base_names["lphg"].keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

        fig.suptitle(f'Enemy {enemy[-1]}', fontsize=16)

        ax_fit = multirun_plots(
            ax1,{
                "LPHG":logs["lphg"][enemy],
                "HPLG": logs["hplg"][enemy],
            },
            colors
        )

        ax_div = multirun_plots_diversity(
            ax2,{
                "LPHG": logs["lphg"][enemy],
                "HPLG": logs["hplg"][enemy],
            },
            colors,
        )
        ax_div.yaxis.set_ticks_position('right')
        ax_div.yaxis.set_label_position('right')

        ax_fit.legend(loc='center right', bbox_to_anchor=(1.0, 0.55))
        ax_div.legend(loc='center right')

        ax_fit.set_title("Population's Average and Best\nFitness Averaged Over 10 Runs", fontsize=14)
        ax_div.set_title("Population's Diversity Averaged\nOver 10 Runs", fontsize=14)
        
        fig.text(0.5, -0.04, 'Generations', ha='center', va='center')

        ax_div.yaxis.label.set_fontsize(14)
        ax_fit.yaxis.label.set_fontsize(14)
        # Adjust layout to minimize gaps
        plt.subplots_adjust(wspace=0)
        # Adjust layout
        fig.tight_layout()
        # Show the combined figure
        fig.savefig(os.path.join(plots_path, f"{enemy}_fitness_diversity.png"))

    # gain boxplot
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

    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = Environment(
        experiment_name=EXPERIMENT_NAME,  # this is actually a path!
        multiplemode="no",
        enemies=[2],
        player_controller=nc,
        visuals=False,
        level=2,
    )
        
    experiment_base_path = "../experiments"

    gains = get_gain_values(env, base_names, experiment_base_path, repeat=5)
    #print(gains)
    # Simulated data for 5 runs of two algorithms over 3 enemies
    data = [
        gains["lphg"]["enemy2"],  # Algorithm 1 for Enemy 1
        gains["hplg"]["enemy2"],  # Algorithm 2 for Enemy 1
        gains["lphg"]["enemy5"],  # Algorithm 1 for Enemy 2
        gains["hplg"]["enemy5"],  # Algorithm 2 for Enemy 2
        gains["lphg"]["enemy7"],  # Algorithm 1 for Enemy 3
        gains["hplg"]["enemy7"]   # Algorithm 2 for Enemy 3  
    ]

    labels = ['Enemy2', 'Enemy5', 'Enemy7']
    algorithm_names = ['Gain LPHG', 'Gain HPLG']

    # Plot for one enemy (can be repeated for other enemies)
    plot_final(data, labels, algorithm_names)
