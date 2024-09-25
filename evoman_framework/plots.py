"""Module for plotting stats"""
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from deap.tools import Statistics
import pickle


def plot_stats(logs: Statistics, ylog=False) -> Figure:
    """ Plots the population's average (including std) and best fitness."""

    generation = logs.select("gen")
    best_fitness = np.array(logs.select("max"))
    avg_fitness = np.array(logs.select("avg"))
    stdev_fitness = np.array(logs.select("std"))

    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(generation, avg_fitness, 'b-', label="average", markersize=8)

    ax.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd", markersize=8)
    ax.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd", markersize=8)
    plt.fill_between(
        generation,
        avg_fitness - stdev_fitness,
        avg_fitness + stdev_fitness,
        color='g',
        alpha=0.2,
        #label='Standard Deviation'
    )

    ax.plot(generation, best_fitness, 'r-', label="best", markersize=8)

    # Customize the plot
    ax.set_title("Population's Average and Best Fitness")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")
    ax.grid(True)
    ax.legend(loc="best")

    if ylog:
        ax.set_yscale('symlog')

    # Adjust the layout
    fig.tight_layout()
    return fig


def multirun_plots(experiment_logs: list, ylog=False):
    """ Plots the population's average (including std) and best fitness."""

    metrics = {"gen": np.array([]), "max": np.array([]), "avg": np.array([]), "std": np.array([])}

    for log in experiment_logs:
        for metric in metrics.keys():
            values = np.expand_dims(np.array(log.select(metric)), axis=0)
            if metrics[metric].size == 0:
                metrics[metric] = values
            else:
                metrics[metric] = np.concat([metrics[metric], values], axis = 0)
    
    # average over runs
    for metric in metrics.keys():
        metrics[metric] = np.mean(metrics[metric], axis=0)

    fig, ax = plt.subplots()

    
    
    # Plot the data
    ax.plot(metrics["gen"], metrics["avg"], 'b-', label="average", markersize=8)

    ax.plot(metrics["gen"], metrics["avg"] - metrics["std"], 'g-.', label="-1 sd", markersize=8)
    ax.plot(metrics["gen"], metrics["avg"] + metrics["std"], 'g-.', label="+1 sd", markersize=8)
    plt.fill_between(
        metrics["gen"],
        metrics["avg"] - metrics["std"],
        metrics["avg"] + metrics["std"],
        color='g',
        alpha=0.2,
        #label='Standard Deviation'
    )

    ax.plot(metrics["gen"], metrics["max"], 'r-', label="best", markersize=8)

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
    return fig


if __name__ == '__main__':
    base_name = "test_final_low_g_25092024_155622"
    experiment_path = "../experiments"
    experiment_paths = [os.path.join(experiment_path, path) for path in os.listdir(experiment_path) if base_name in path]

    logs = []
    for experiment_path in experiment_paths:
        with open(os.path.join(experiment_path, "logbook.pkl"), mode="rb") as log_file:
            logs.append(pickle.load(log_file))

    multirun_plots(logs)
    plt.show()