"""Plots for assignment 2"""
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd


def box_plot_gain(
        df_data: pd.DataFrame,
        title_size: int = 20,
        ticks_size: int = 16,
        axes_label_size: int = 20,
        legend_size: int = 20) -> Figure:
    """Creates and returns a box-plot with the gain values of the different algorithms.
    Expects a dataframe with [algorithm,run,gain_sum] as columns. The results will be grouped
    after the algorithm column, and using the results from the gain_sum.!

    Args:
        df_data (pd.DataFrame): Dataframe with algorithm, run, gain_sum columns
        The labels of the boxes should be given by the "algorithm" column. Overwrite
        before to change
    
    Returns:
        Axus: Created subfigure to save.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create a boxplot per group (algorithm column defines the groups)
    box = df_data.boxplot(
        column='gain_sum',
        by='algorithm',
        ax=ax,
        patch_artist=True,
        showfliers=True
    )

    # Define color schemes
    color_scheme_1 = 'wheat'
    color_scheme_2 = 'cadetblue'
    median_color1 = 'darkgoldenrod'
    median_color2 = 'darkslategrey'

    # Apply colors to each boxplot
    # Retrieve the boxes, medians, whiskers, and caps from the Axes object
    for i, box in enumerate(ax.patches):
        if i % 2 == 0:  # 1st, 3rd
            box.set_facecolor(color_scheme_1)
            box.set_edgecolor(median_color1)
            ax.lines[6 * i + 4].set_color(median_color1)  # median
            ax.lines[6 * i + 5].set_color(median_color1)  # median
        else:  # 2nd, 4th
            box.set_facecolor(color_scheme_2)
            box.set_edgecolor(median_color2)
            ax.lines[6 * i + 4].set_color(median_color2)  # median
            ax.lines[6 * i + 5].set_color(median_color2)  # median

    # Adding legend to differentiate the groups
    algorithm_names = df_data['algorithm'].unique()
    handles = [plt.Line2D([0], [0], color=color_scheme_1, lw=4),
               plt.Line2D([0], [0], color=color_scheme_2, lw=4)]
    plt.legend(handles, algorithm_names, loc='best', fontsize=legend_size)

    # Adding titles and labels
    ax.set_title('Box-plot of the Sum of Gain for All Enemies by Algorithm', fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=ticks_size)
    plt.suptitle('')  # Remove the default 'by' title
    ax.set_xlabel('Algorithm/Enemies Trained on', fontsize=axes_label_size)
    ax.set_ylabel('Sum of Gain for all Enemies', fontsize=axes_label_size)

    return fig


def plot_fitness(
        ax: Axis,
        df_data: pd.DataFrame,
        palette: dict,
        title_size: int = 20,
        ticks_size: int = 16,
        axes_label_size: int = 20,
        legend_size: int = 20
    ) -> Axis:
    """Function to plot the fitness values of multiple runs. Dataframe needs to contain
    avg, max, std, algorithm, generation as columns.

    Args:
        ax (Axis): Axis of the plot to add the data.
        df_data (pd.DataFrame): DataFrame with requested columns.
        palette (dict): Dict with mean, std and max as keys and colors as values

    Returns:
        Axis: Plotted data on axis.
    """

    # Group by the 'algorithm' column
    grouped = df_data.groupby('algorithm')

    # Loop over each algorithm group
    for (algorithm, group), palette in zip(grouped, palette):
        # Extract relevant data for this group
        generations = group['generation']
        avg_fitness = group['avg']
        max_fitness = group['max']
        std_fitness = group['std']
        #max_beaten = (group["max_defeated"] / 8) * 100

        # Plot the standard deviation as a shaded region
        ax.fill_between(generations, avg_fitness - std_fitness, avg_fitness + std_fitness,
                        color=palette['std'], alpha=0.2, label=f"Std Dev ({algorithm})")

        # Plot the mean fitness as a line
        ax.plot(generations, avg_fitness, label=f"Mean Fitness ({algorithm})", color=palette['mean'])

        # Plot the max fitness as a dashed line
        ax.plot(generations, max_fitness, label=f"Max Fitness ({algorithm})", linestyle='--', color=palette['max'])
        
        #ax.plot(generations, max_beaten, label=f"Max defeated ({algorithm})", linestyle="-", color=palette["defeated"])


    # Adding labels and title
    ax.set_xlabel('Generations', fontsize=axes_label_size)
    ax.set_ylabel('Fitness', fontsize=axes_label_size)
    ax.set_title('Mean, Max, and Std Fitness by Algorithm', fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=ticks_size)
    ax.legend(loc='best', fontsize=legend_size)
    ax.grid(True)

    return ax


def plot_defeated(
        ax: Axis,
        df_data: pd.DataFrame,
        palette: dict,
        title_size: int = 20,
        ticks_size: int = 16,
        axes_label_size: int = 20,
        legend_size: int = 20
    ) -> Axis:
    # Group by the 'algorithm' column
    grouped = df_data.groupby('algorithm')

    # Loop over each algorithm group
    for (algorithm, group), palette in zip(grouped, palette):
        # Extract relevant data for this group
        generations = group['generation']
        avg_defeated = group["defeated"]
        max_defeated = group["max_defeated"]

        # Plot the mean fitness as a line
        ax.plot(generations, avg_defeated, label=f"Mean Defeated ({algorithm})", color=palette['defeated'])

        # Plot the max fitness as a dashed line
        ax.plot(generations, max_defeated, label=f"Max Defeated ({algorithm})", linestyle='--', color=palette['max_defeated'])

    # Adding labels and title
    ax.set_xlabel('Generations', fontsize=axes_label_size)
    ax.set_ylabel('Defeated', fontsize=axes_label_size)
    ax.set_ylim(0, 8)
    ax.set_title('Defeated Enemies by Algorithm', fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=ticks_size)
    ax.legend(loc='best', fontsize=legend_size)
    ax.grid(True)

    # Return the figure
    return ax