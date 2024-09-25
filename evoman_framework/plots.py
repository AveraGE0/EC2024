"""Module for plotting stats"""
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from deap.tools import Statistics
from scipy.stats import ttest_ind
import pickle


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
    ax.grid(False)
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


def multirun_plots(experiment_logs: dict[str, list], ylog=False):
    """ Plots the population's average (including std) and best fitness."""
    plots = []
    for name, logs in experiment_logs.items():
        fig, ax = plt.subplots()
        metrics = {"gen": np.array([]), "max": np.array([]), "avg": np.array([]), "std": np.array([])}

        for log in logs:
            for metric in metrics.keys():
                values = np.expand_dims(np.array(log.select(metric)), axis=0)
                if metrics[metric].size == 0:
                    metrics[metric] = values
                else:
                    metrics[metric] = np.concat([metrics[metric], values], axis = 0)
        
        # average over runs
        for metric in metrics.keys():
            metrics[metric] = np.mean(metrics[metric], axis=0)
       
        # Plot the data
        ax.plot(metrics["gen"], metrics["avg"], 'b-', label=f"average {name}", markersize=8)

        ax.plot(metrics["gen"], metrics["avg"] - metrics["std"], 'g-.', label=f"-1 sd {name}", markersize=8)
        ax.plot(metrics["gen"], metrics["avg"] + metrics["std"], 'g-.', label=f"+1 sd {name}", markersize=8)
        plt.fill_between(
            metrics["gen"],
            metrics["avg"] - metrics["std"],
            metrics["avg"] + metrics["std"],
            color='g',
            alpha=0.2,
            #label='Standard Deviation'
        )

        ax.plot(metrics["gen"], metrics["max"], 'r-', label=f"{name} best", markersize=8)

        # Customize the plot
        ax.set_title(f"Population's Average and Best Fitness in Averaged over {len(experiment_logs)} Runs")
        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.grid(True)
        ax.legend(loc="best")

        if ylog:
            ax.set_yscale('symlog')

        # Adjust the layout
        fig.tight_layout()
        plots.append(fig)
    return plots


def plot_final(data, labels, algorithm_names, enemy_name):
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
    # Split data into two groups for the algorithms
    #alg1_data = [x for i, x in enumerate(data) if labels[i] == enemy_name and i % 2 == 0]
    #alg2_data = [x for i, x in enumerate(data) if labels[i] == enemy_name and i % 2 == 1]
    
    # Combine data for boxplot
    #combined_data = [alg1_data, alg2_data]

    # Getting the right positions
    positions = [1, 2, 4, 5, 7, 8]

    t_stat = []
    p_value = []
    for i in [0,2,4]:
        t_stat.append(round(ttest_ind(data[i], data[i+1])[0], 4))
        p_value.append(round(ttest_ind(data[i], data[i+1])[1], 4))

    # Find the maximum y value in the data
    y_max = max([max(d) for d in data])  
    text_y_pos = y_max + 5 

    plt.figure(figsize=(8, text_y_pos+2))
    
    # Create box plot
    box = plt.boxplot(data, positions=positions, widths=0.9, patch_artist=True)
    plt.title(f'Comparison of Performance on Different Enemies with Large and Small Population Size')
    plt.ylabel('Performance')
    plt.xticks([1.5, 4.5, 7.5], ['Enemy 1', 'Enemy 2', 'Enemy 3'])

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
        plt.text(j, text_y_pos, f't-stat: {t_stat[i]}\n p-value: {p_value[i]}', ha='center')

    plt.show()




if __name__ == '__main__':
    base_names = {
        "enemy2": "test_final_high_g_enemy=2_25092024_183036",
        "enemy5": "test_final_high_g_enemy=5_25092024_182952"
    }
    
    experiment_base_path = "../experiments"
    
    logs = {}

    for name, base_name in base_names.items():
        experiment_paths = [os.path.join(experiment_base_path, path) for path in os.listdir(experiment_base_path) if base_name in path]
        logs[name] = []
        for experiment_path in experiment_paths:
            with open(os.path.join(experiment_path, "logbook.pkl"), mode="rb") as log_file:
                logs[name].append(pickle.load(log_file))
    # TODO: we need both algorithms per enemy!
    figs = multirun_plots(logs)
    for i, plot in enumerate(figs):
        plot.savefig(os.path.join(experiment_base_path, f"test{i}.png"))
    plt.show()
    # Example Usage
    # Simulated data for 5 runs of two algorithms over 3 enemies
    data = [
        np.random.normal(50, 10, 5),  # Algorithm 1 for Enemy 1
        np.random.normal(55, 10, 5),  # Algorithm 2 for Enemy 1
        np.random.normal(60, 10, 5),  # Algorithm 1 for Enemy 2
        np.random.normal(62, 10, 5),  # Algorithm 2 for Enemy 2
        np.random.normal(65, 10, 5),  # Algorithm 1 for Enemy 3
        np.random.normal(63, 10, 5)   # Algorithm 2 for Enemy 3  
    ]

    labels = ['enemy1', 'enemy1', 'enemy2', 'enemy2', 'enemy3', 'enemy3']
    algorithm_names = ['Large Population', 'Small Population']

    # Plot for one enemy (can be repeated for other enemies)
    plot_final(data, labels, algorithm_names, 'enemy1')
