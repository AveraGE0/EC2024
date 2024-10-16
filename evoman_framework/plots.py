"""Module for plotting stats"""
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from deap.tools import Statistics
from scipy.stats import ttest_ind
import pickle
from gain import get_gain_values
from neural_controller import NeuralController
from evoman.environment import Environment


def plot_stats(logs: Statistics, ylog=False) -> Figure:
    """ Plots the population's average (including std) and best fitness."""
    logs = logs.chapters['fitness']
    generation = logs.select("gen")
    best_fitness = np.array(logs.select("max"))
    avg_fitness = np.array(logs.select("avg"))
    stdev_fitness = np.array(logs.select("std"))

    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(generation, avg_fitness, 'darkslategrey', label="mean", markersize=8)

    ax.plot(generation, avg_fitness - stdev_fitness, 'cadetblue', label="-1 sd", markersize=8)
    ax.plot(generation, avg_fitness + stdev_fitness, 'cadetblue', label="+1 sd", markersize=8)
    plt.fill_between(
        generation,
        avg_fitness - stdev_fitness,
        avg_fitness + stdev_fitness,
        color='cadetblue',
        alpha=0.2,
        #label='Standard Deviation'
    )

    ax.plot(generation, best_fitness, 'darkgoldenrod', label="max", markersize=8)

    # Customize the plot
    ax.set_title("Population's Mean and Maximum Over Ten Runs")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")
    ax.grid(True)
    ax.legend(loc="best")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 0)) 
    ax.spines['bottom'].set_position(('outward', 0)) 

    if ylog:
        ax.set_yscale('symlog')

    # Adjust the layout
    fig.tight_layout()
    return fig


def multirun_plots(ax, experiment_logs: dict[str, list], colors: list, ylog=False):
    """ Plots the population's average (including std) and best fitness.

    Args:
        experiment_logs (dict[str, list]): Name: logs for each algorithm that should be plotted
        ylog (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    for (name, logs), palette in zip(experiment_logs.items(), colors):
        metrics = {"gen": np.array([]), "max": np.array([]), "avg": np.array([]), "std": np.array([])}

        for log in logs:
            log = log.chapters['fitness']
            for metric in metrics.keys():
                values = np.expand_dims(np.array(log.select(metric)), axis=0)
                if metrics[metric].size == 0:
                    metrics[metric] = values
                else:
                    metrics[metric] = np.concatenate([metrics[metric], values], axis = 0)
        runs = 0
        # average over runs
        for metric in metrics.keys():
            runs = metrics[metric].shape[0]
            metrics[metric] = np.mean(metrics[metric], axis=0)
       
        # Plot the data
        ax.plot(metrics["gen"], metrics["avg"], "-", color=f'{palette["avg"]}', label=f"Average {name}", markersize=8)

        #ax.plot(metrics["gen"], metrics["avg"] - metrics["std"], 'g-.', label=f"-1 sd {name}", markersize=8)
        #ax.plot(metrics["gen"], metrics["avg"] + metrics["std"], 'g-.', label=f"+1 sd {name}", markersize=8)
        ax.fill_between(
            metrics["gen"],
            metrics["avg"] - metrics["std"],
            metrics["avg"] + metrics["std"],
            color=palette["std"],
            alpha=0.2,
            #label='Standard Deviation'
        )

        ax.plot(metrics["gen"], metrics["max"], "-", color=f'{palette["max"]}', label=f"Best {name}", markersize=8)

    # Customize the plot
    ax.set_title(f"Population's Average and Best Fitness Averaged Over {runs} Runs")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")
    ax.grid(True)
    ax.legend(loc="best")

    if ylog:
        ax.set_yscale('symlog')

    return ax


def multirun_plots_diversity(ax, experiment_logs: dict[str, list], colors: list, ylog=False):
    """ Plots the population's average (including std) and best fitness.

    Args:
        experiment_logs (dict[str, list]): Name: logs for each algorithm that should be plotted
        ylog (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    for (name, logs), palette in zip(experiment_logs.items(), colors):
        metrics = {"gen": np.array([]), "std": np.array([]), "euclidean_avg": np.array([]), "hamming": np.array([])}

        for log in logs:
            log_d = log.chapters['diversity']
            log_f = log.chapters['fitness']
            for metric in metrics.keys():
                if metric == "std":
                    log = log_f
                else:
                    log = log_d
                values = np.expand_dims(np.array(log.select(metric)), axis=0)
                if metrics[metric].size == 0:
                    metrics[metric] = values
                else:
                    metrics[metric] = np.concatenate([metrics[metric], values], axis = 0)
        runs = 0
        # average over runs
        for metric in metrics.keys():
            runs = metrics[metric].shape[0]
            metrics[metric] = np.mean(metrics[metric], axis=0)
       
        # Plot the data
        ax.plot(metrics["gen"], metrics["euclidean_avg"], "-", color=f'{palette["euclidean"]}', label=f"Average Euclidean {name}", markersize=8)
        ax.plot(metrics["gen"], metrics["hamming"], "-", color=f'{palette["hamming"]}', label=f"Average Hamming {name}", markersize=8)
        ax.plot(metrics["gen"], metrics["std"], "-", color=f'{palette["std"]}', label=f"Fitness STD {name}", markersize=8)
        #plt.show()
        #pass
    # Customize the plot
    ax.set_title(f"Population's Diversity Over {runs} Runs for Enemy {enemy[-1]}")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Distance")
    ax.grid(True)
    ax.legend(loc="best")
    #fig.show()
    #plt.show()
    if ylog:
        ax.set_yscale('symlog')
    return ax


def plot_final(data, labels, algorithm_names):
    """
    Plots box plots comparing two algorithms per enemy and performs a t-test.
    
    Parameters:
    - data: A list of lists or arrays, each representing the 5 runs for a given algorithm.
            Should be in the form [[alg1_runs_enemy1], [alg2_runs_enemy1], [alg1_runs_enemy2], [alg2_runs_enemy2], ...]
    - labels: List of labels corresponding to the 'enemy' for each pair (e.g., ['enemy1', 'enemy1', 'enemy2', 'enemy2']).
    - algorithm_names: List of names of the algorithms being compared (e.g., ['Algorithm1', 'Algorithm2']).
    - enemy_name: The name of the current enemy being plotted.
    
    Output:
    - Box plot comparing the performance of two algorithms for the given enemy.
    - Prints the p-value from the t-test.
    """
    # Getting the right positions
    positions = [1, 2, 4, 5, 7, 8]

    t_stat = []
    p_value = []
    for i in [0,2,4]:
        t_stat.append(round(ttest_ind(data[i], data[i+1])[0], 4))
        p_value.append(round(ttest_ind(data[i], data[i+1])[1], 4))

    plt.figure(figsize=(8, 4))
    
    # Create box plot
    box = plt.boxplot(data, positions=positions, widths=0.9, patch_artist=True)
    plt.title(f'Gain on Different Enemies for Two Configurations')
    plt.ylabel('Gain')
    plt.xticks([1.5, 4.5, 7.5], labels)
    plt.grid(True)

    # Define color schemes
    color_scheme_1 = 'wheat'
    color_scheme_2 = 'cadetblue'
    median_color1 = 'darkgoldenrod'
    median_color2 = 'darkslategrey'

    # Apply colors to each boxplot
    for i in range(len(data)):
        if i % 2 == 0:  # 1st, 3rd, 5th boxplots
            plt.setp(box['boxes'][i], facecolor=color_scheme_1, edgecolor=median_color1)
            plt.setp(box['medians'][i], color=median_color1, linewidth=2)
            plt.setp(box['whiskers'][2 * i:2 * i + 2], color=median_color1, linewidth=1) 
            plt.setp(box['caps'][2 * i:2 * i + 2], color=median_color1, linewidth=1) 
        else:  # 2nd, 4th, 6th boxplots
            plt.setp(box['boxes'][i], facecolor=color_scheme_2, edgecolor=median_color2)
            plt.setp(box['medians'][i], color=median_color2, linewidth=2)
            plt.setp(box['whiskers'][2 * i:2 * i + 2], color=median_color2, linewidth=1) 
            plt.setp(box['caps'][2 * i:2 * i + 2], color=median_color2, linewidth=1)

    handles = [plt.Line2D([0], [0], color=color_scheme_1, lw=4),
               plt.Line2D([0], [0], color=color_scheme_2, lw=4)]
    plt.legend(handles, algorithm_names, loc='best')

    # Printing the t-stat and p-value
    for i, j in zip(range(len(t_stat)), [1.5, 4.5, 7.5]):
        plt.text(j, 45, f't-stat: {t_stat[i]}\n p-value: {p_value[i]}', ha='center', va='top')
    plt.tight_layout()
    plt.savefig("../summary_plots/boxplot.png")
    #plt.show()




if __name__ == '__main__':
    # base_names = {
    #     "lphg": {   
    #         "enemy2": "high_g_enemy=2_25092024_210118",
    #         "enemy5": "high_g_enemy=5_25092024_211033",
    #         "enemy7": "high_g_enemy=7_25092024_212353"
    #     },
    #     "hplg": {
    #         "enemy2": "low_g_enemy=2_25092024_213352",
    #         "enemy5": "low_g_enemy=5_25092024_214406",
    #         "enemy7": "low_g_enemy=7_25092024_220135"
    #     }
    # }
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
    colors = [
        {"max": "#B8860B", "avg": "#5F9EA0", "std": "#5F9EA0", "euclidean": "#F5DEB3", "hamming": "#2F4F4F"},  # wheat and cadetblue for the first configuration
        {"max": "#FFA500", "avg": "#FF4500", "std": "#FF4500", "euclidean": "#800080", "hamming": "#FFA500"}   # orange and red for the second configuration
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
            experiment_paths = [os.path.join(experiment_base_path, path) for path in os.listdir(experiment_base_path) if base_name in path]
            # append all logs to the algorithm, enemy combination
            for experiment_path in experiment_paths:
                with open(os.path.join(experiment_path, "logbook.pkl"), mode="rb") as log_file:
                    logs[name][enemy].append(pickle.load(log_file))
    # TODO: we need both algorithms per enemy!
    
    
    for enemy in base_names["lphg"].keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))#, gridspec_kw={'wspace': 0})
        fig.suptitle(f'Enemy {enemy[-1]}', fontsize=16)
        ax_fit = multirun_plots(
            ax1,{
                "LPHG":logs["lphg"][enemy],
                "HPLG": logs["hplg"][enemy],
            },
            colors
        )
        #fig.savefig(os.path.join(plots_path, f"{enemy}_fitness.png"))

        ax_div = multirun_plots_diversity(
            ax2,{
                "LPHG": logs["lphg"][enemy],
                "HPLG": logs["hplg"][enemy],
            },
            colors,
        )
        ax_div.yaxis.set_ticks_position('right')
        ax_div.yaxis.set_label_position('right')
        #fig.savefig(os.path.join(plots_path, f"{enemy}_diversity.png"))

        # Remove the left y-axis from the right plot
        #ax2.spines['left'].set_visible(False)
          # Move the label to the right
        # Optionally, add legends
        ax_fit.legend(loc='center right', bbox_to_anchor=(1.0, 0.55))
        ax_div.legend(loc='center right')

        ax_fit.set_title("Population's Average and Best\nFitness Averaged Over 10 Runs", fontsize=14)
        ax_div.set_title("Population's Diversity Averaged\nOver 10 Runs", fontsize=14)
        
        fig.text(0.5, -0.04, 'Generations', ha='center', va='center')
        #ax1.grid(True)
        #ax2.grid(True)
        ax_div.yaxis.label.set_fontsize(14)
        ax_fit.yaxis.label.set_fontsize(14)
        # Adjust layout to minimize gaps
        plt.subplots_adjust(wspace=0)

        # Adjust layout
        fig.tight_layout()

        # Show the combined figure
        fig.savefig(os.path.join(plots_path, f"{enemy}_fitness_diversity.png"))

    # GAIN BOXPLOT
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
