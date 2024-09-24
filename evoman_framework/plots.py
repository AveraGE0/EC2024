"""Module for plotting stats"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from deap.tools import Statistics


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