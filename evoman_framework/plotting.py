import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(logbooks, group_names):
    """Plot metrics over generations for each group."""
    for logbook, group_name in zip(logbooks, group_names):
        gen = logbook.select('gen')
        avg_enemy_life = logbook.select('avg_enemy_life')
        avg_player_life = logbook.select('avg_player_life')
        avg_gain = logbook.select('avg_gain')
        avg_time = logbook.select('avg_time')
        avg_distance = logbook.select('avg_distance')

        fig, axs = plt.subplots(5, 1, figsize=(10, 25))

        axs[0].plot(gen, avg_enemy_life, label='Avg Enemy Life')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Average Enemy Life')
        axs[0].legend()

        axs[1].plot(gen, avg_player_life, label='Avg Player Life')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Average Player Life')
        axs[1].legend()

        axs[2].plot(gen, avg_gain, label='Avg Gain')
        axs[2].set_xlabel('Generation')
        axs[2].set_ylabel('Average Gain')
        axs[2].legend()

        axs[3].plot(gen, avg_time, label='Avg Time')
        axs[3].set_xlabel('Generation')
        axs[3].set_ylabel('Average Time')
        axs[3].legend()

        axs[4].plot(gen, avg_distance, label='Avg Diversity')
        axs[4].set_xlabel('Generation')
        axs[4].set_ylabel('Average Diversity')
        axs[4].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(EXPERIMENT_NAME, f'metrics_plot_{group_name}.png'))
        plt.close()


def analyze_results(csv_file):
    # Load the CSV data
    df = pd.read_csv(csv_file)

    # 1. Plot average fitness metrics over generations
    plt.figure(figsize=(10, 6))

    # Update column names based on actual CSV structure
    plt.plot(df['gen'], df['avg_enemy_life'], label='Avg Enemy Health')
    plt.plot(df['gen'], df['avg_player_life'], label='Avg Player Health')
    plt.plot(df['gen'], df['avg_time'], label='Avg Time')

    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Average Fitness Over Generations')
    plt.legend()
    plt.savefig('fitness_over_generations.png')
    plt.show()

    # 2. Plot Pareto front for player health vs enemy health
    plt.scatter(df['avg_player_life'], df['avg_enemy_life'], c=df['gen'], cmap='viridis')
    plt.colorbar(label='Generation')
    plt.xlabel('Player Health')
    plt.ylabel('Enemy Health')
    plt.title('Pareto Front Across Generations')
    plt.savefig('pareto_front.png')
    plt.show()

    # 3. Generate Summary Stats
    summary = df[['avg_enemy_life', 'avg_player_life', 'avg_time']].describe()
    print(summary)

    # Save summary to file
    summary.to_csv('summary_statistics.csv')


# Example usage
analyze_results('experiment_nsga2/generation_statistics.csv')


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
        # label='Standard Deviation'
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
                    metrics[metric] = np.concatenate([metrics[metric], values], axis=0)
        runs = 0
        # average over runs
        for metric in metrics.keys():
            runs = metrics[metric].shape[0]
            metrics[metric] = np.mean(metrics[metric], axis=0)

        # Plot the data
        ax.plot(metrics["gen"], metrics["avg"], "-", color=f'{palette["avg"]}', label=f"Average {name}", markersize=8)

        # ax.plot(metrics["gen"], metrics["avg"] - metrics["std"], 'g-.', label=f"-1 sd {name}", markersize=8)
        # ax.plot(metrics["gen"], metrics["avg"] + metrics["std"], 'g-.', label=f"+1 sd {name}", markersize=8)
        ax.fill_between(
            metrics["gen"],
            metrics["avg"] - metrics["std"],
            metrics["avg"] + metrics["std"],
            color=palette["std"],
            alpha=0.2,
            # label='Standard Deviation'
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
                    metrics[metric] = np.concatenate([metrics[metric], values], axis=0)
        runs = 0
        # average over runs
        for metric in metrics.keys():
            runs = metrics[metric].shape[0]
            metrics[metric] = np.mean(metrics[metric], axis=0)

        # Plot the data
        ax.plot(metrics["gen"], metrics["euclidean_avg"], "-", color=f'{palette["euclidean"]}',
                label=f"Average Euclidean {name}", markersize=8)
        ax.plot(metrics["gen"], metrics["hamming"], "-", color=f'{palette["hamming"]}', label=f"Average Hamming {name}",
                markersize=8)
        ax.plot(metrics["gen"], metrics["std"], "-", color=f'{palette["std"]}', label=f"Fitness STD {name}",
                markersize=8)
        # plt.show()
        # pass
    # Customize the plot
    ax.set_title(f"Population's Diversity Over {runs} Runs for Enemy {enemy[-1]}")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Distance")
    ax.grid(True)
    ax.legend(loc="best")
    # fig.show()
    # plt.show()
    if ylog:
        ax.set_yscale('symlog')
    return ax


# Plot Gain against Each Enemy
def plot_gain_per_enemy(data):
    enemies = [f'gain_enemy_{i}' for i in range(1, 9)]
    groups = data['group']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot gains for each enemy
    for enemy in enemies:
        ax.plot(groups, data[enemy], marker='o', label=enemy.replace('_', ' ').title())

    ax.set_xlabel('Group')
    ax.set_ylabel('Gain')
    ax.set_title('Gain per Enemy for Each Group')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.show()

"""# Function to generate all plots from the data
def plot_all_results(data):
    plot_total_gain_and_defeated_enemies(data)
    plot_gain_per_enemy(data)
    plot_player_vs_enemy_life(data)

# Assuming 'data' is a pandas DataFrame that has been loaded from the CSV file
# Call the plot_all_results(data) function to generate all the plots"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# --------------------

# Plotting from GPT

# ------------------

# 1. Fitness Metrics: Plot average, maximum, and minimum fitness across generations
def plot_fitness_metrics(generations, avg_fitness, min_fitness, max_fitness):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, label='Average Fitness', color='blue')
    plt.fill_between(generations, min_fitness, max_fitness, color='lightblue', alpha=0.5)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness Metrics Over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fitness_metrics.png')
    plt.show()


# 2. Gain Metrics: Gain per enemy, average gain across enemies
def plot_gain_per_enemy(data):
    enemies = [f'gain_enemy_{i}' for i in range(1, 9)]
    groups = data['group']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot gains for each enemy
    for enemy in enemies:
        ax.plot(groups, data[enemy], marker='o', label=enemy.replace('_', ' ').title())

    ax.set_xlabel('Group')
    ax.set_ylabel('Gain')
    ax.set_title('Gain per Enemy for Each Group')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.savefig('gain_per_enemy.png')
    plt.show()


# 3. Player vs. Enemy Life: Bar plot comparison
def plot_player_vs_enemy_life(data):
    fig, axs = plt.subplots(4, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i in range(1, 9):
        ax = axs[i - 1]
        ax.bar(data['group'], data[f'player_life_enemy_{i}'], color='green', width=0.4, label='Player Life')
        ax.bar(data['group'], data[f'enemy_life_enemy_{i}'], color='red', width=0.4, label='Enemy Life')
        ax.set_title(f'Player Life vs Enemy Life (Enemy {i})')
        ax.set_xlabel('Group')
        ax.set_ylabel('Life')
        ax.legend()

    plt.tight_layout()
    plt.savefig('player_vs_enemy_life.png')
    plt.show()


# 4. Diversity Over Generations: Track diversity metrics
def plot_diversity_over_generations(generations, avg_diversity):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_diversity, label='Average Diversity', color='purple')
    plt.xlabel('Generations')
    plt.ylabel('Diversity')
    plt.title('Diversity Over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('diversity_over_generations.png')
    plt.show()


# 5. Generation-to-Generation Improvements: Improvement in gain, fitness, or diversity
def plot_generation_improvements(generations, avg_gain, avg_fitness, avg_diversity):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Average Gain', color='blue')
    ax1.plot(generations, avg_gain, label='Average Gain', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Fitness', color='green')
    ax2.plot(generations, avg_fitness, label='Average Fitness', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Generation-to-Generation Improvement (Gain and Fitness)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('generation_improvements.png')
    plt.show()


# 6. Comparison of Best Individuals across Groups
def plot_comparison_best_individuals(data):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    groups = data['group']

    ax1.bar(groups, data['total_gain'], color='skyblue', label='Total Gain', width=0.4, align='center')
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Total Gain', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax2 = ax1.twinx()
    ax2.bar(groups, data['defeated_enemies'], color='orange', label='Defeated Enemies', width=0.3, align='edge')
    ax2.set_ylabel('Defeated Enemies', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('Comparison of Best Individuals Across Groups')
    plt.tight_layout()
    plt.savefig('comparison_best_individuals.png')
    plt.show()


# Main function to call all plotting functions
def plot_all_results(data, generations, avg_fitness, min_fitness, max_fitness, avg_gain, avg_diversity):
    # Plot fitness metrics
    plot_fitness_metrics(generations, avg_fitness, min_fitness, max_fitness)

    # Plot gain per enemy
    plot_gain_per_enemy(data)

    # Plot player life vs enemy life
    plot_player_vs_enemy_life(data)

    # Plot diversity over generations
    plot_diversity_over_generations(generations, avg_diversity)

    # Plot generation-to-generation improvements
    plot_generation_improvements(generations, avg_gain, avg_fitness, avg_diversity)

    # Plot comparison of best individuals across groups
    plot_comparison_best_individuals(data)

# Example usage:
# Assuming you have loaded the data and relevant metrics (e.g., generations, avg_fitness, etc.)
# Call plot_all_results with the required data

# Example:
# plot_all_results(data, generations, avg_fitness, min_fitness, max_fitness, avg_gain, avg_diversity)