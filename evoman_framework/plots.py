"""Module for plotting stats"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from deap.tools import Statistics
from scipy.stats import ttest_ind
import pickle


def plot_island_metric(logs: Statistics, metric: str, chapter: str = "fitness", ylog=False):
    """ Plots the population's average (including std) and best fitness for each island and overall.
    
    Args:
        logs (Logbook): The DEAP logbook that contains overall stats (mean, std, max).
        island_stats (list of Logbooks): List of logbooks for each island.
        n_islands (int): Number of islands.
        metric (str): Metric that should be plotted (must be in logbook!).
        ylog (bool): If true, will set y-axis to symlog scale.
    """
    # Extract overall population stats from the logbook
    logs = logs.chapters[chapter]
    generation = np.array(logs.select("gen"))
    island = np.array(logs.select("island"))
    metric_values = np.array(logs.select(metric))

    n_islands = len(set(island))

    fig, ax = plt.subplots()

    for i in list(range(n_islands)) + [None]:
        indices = np.where(island==i)[0]  # Gives the indexes in A for which value = 2
        gen_filtered = generation[indices]
        metric_filtered = metric_values[indices]

        # Plot the overall population mean and standard deviation
        ax.plot(gen_filtered, metric_filtered, label=f"i{i} {metric}", markersize=8)

    # Add vertical red lines to indicate migration events every 10 generations
    for gen in range(0, len(set(generation)), 10):
        ax.axvline(x=gen, color='red', linestyle=':', label="Migration" if gen == 0 else "")

    # Customize the plot
    ax.set_title("Island Fitness Over Generations with Migration Events")
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

        ax.fill_between(
            metrics["gen"],
            metrics["avg"] - metrics["std"],
            metrics["avg"] + metrics["std"],
            color=palette["std"],
            alpha=0.2,
            #label='Standard Deviation'
        )

        ax.plot(
            metrics["gen"],
            metrics["max"],
            "-", 
            color=f'{palette["max"]}',
            label=f"Best {name}",
            markersize=8
        )

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
        metrics = {
            "gen": np.array([]),
            "std": np.array([]), 
            "euclidean_avg": np.array([]), 
            "hamming": np.array([])
        }

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
        # Average over runs
        for metric in metrics.keys():
            metrics[metric] = np.mean(metrics[metric], axis=0)

        # Plot the data
        ax.plot(metrics["gen"], metrics["euclidean_avg"], "-", color=f'{palette["euclidean"]}', label=f"Average Euclidean {name}", markersize=8)
        ax.plot(metrics["gen"], metrics["hamming"], "-", color=f'{palette["hamming"]}', label=f"Average Hamming {name}", markersize=8)
        ax.plot(metrics["gen"], metrics["std"], "-", color=f'{palette["std"]}', label=f"Fitness STD {name}", markersize=8)
    # Customize the plot
    ax.set_xlabel("Generations")
    ax.set_ylabel("Distance")
    ax.grid(True)
    ax.legend(loc="best")

    if ylog:
        ax.set_yscale('symlog')
    return ax


def plot_final(data, labels, algorithm_names):
    """
    Plots box plots comparing two algorithms per enemy and performs a t-test.
    
    Parameters:
    - data: A list of lists or arrays, each representing the 5 runs for a given algorithm.
            Should be in the form [[alg1_runs_enemy1], [alg2_runs_enemy1],
                                  [alg1_runs_enemy2], [alg2_runs_enemy2], ...]
    - labels: List of labels corresponding to the 'enemy' for each pair (e.g.,
              ['enemy1', 'enemy1', 'enemy2', 'enemy2']).
    - algorithm_names: List of names of the algorithms being compared (e.g.,
                       ['Algorithm1', 'Algorithm2']).
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
    plt.title('Gain on Different Enemies for Two Configurations')
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
        if i % 2 == 0:  # 1st, 3rd, 5th box-plots
            plt.setp(box['boxes'][i], facecolor=color_scheme_1, edgecolor=median_color1)
            plt.setp(box['medians'][i], color=median_color1, linewidth=2)
            plt.setp(box['whiskers'][2 * i:2 * i + 2], color=median_color1, linewidth=1) 
            plt.setp(box['caps'][2 * i:2 * i + 2], color=median_color1, linewidth=1) 
        else:  # 2nd, 4th, 6th box-plots
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
    plt.savefig("../summary_plots/box_plot.png")


if __name__ == '__main__':
    with open("../experiments/competition_test/logbook_islands.pkl", mode="rb") as log_file:
        logs_islands = pickle.load(log_file)
    plot_island_metric(logs_islands, "avg").savefig("../experiments/competition_test/islands_avg.png")
    plot_island_metric(logs_islands, "max").savefig("../experiments/competition_test/islands_max.png")
   