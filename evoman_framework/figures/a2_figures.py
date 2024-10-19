"""Module to generate the final plots."""
import pandas as pd
import matplotlib.pyplot as plt
from figures.plots_a2 import box_plot_gain, plot_defeated, plot_fitness


if __name__ == '__main__':
    # this will be the gain plot!
    df_data = pd.concat([
        pd.read_csv("../a2_plots/islands_gain.csv"),  # add other metrics here
        pd.read_csv("../a2_plots/nsga_gain.csv"),
    ])
    df_data["algorithm"] = df_data["algorithm"].replace("island_all_enemies", "Island/1-8")
    df_data["algorithm"] = df_data["algorithm"].replace("island_sub_enemies", "Island/1-4")
    df_data["algorithm"] = df_data["algorithm"].replace("nsga_all_enemies", "NSGA II/1-8")
    df_data["algorithm"] = df_data["algorithm"].replace("nsga_sub_enemies", "NSGA II/1-4")
    fig = box_plot_gain(
        df_data,
        title_size=20, axes_label_size=20,
        ticks_size=16, legend_size=20
    )
    fig.savefig("../a2_plots/gain_box_plot.png", dpi=400)

    # This will be the fitness plot
    df_data = pd.concat([
        pd.read_csv("../a2_plots/averaged_fitness_data_island.csv"),
        pd.read_csv("../a2_plots/averaged_fitness_data_nsga.csv"),
    ])
    df_data["algorithm"] = df_data["algorithm"].replace("island_all_enemies", "Island/1-8")
    df_data["algorithm"] = df_data["algorithm"].replace("island_sub_enemies", "Island/1-4")
    df_data["algorithm"] = df_data["algorithm"].replace("nsga_all_enemies", "NSGA-II/1-8")
    df_data["algorithm"] = df_data["algorithm"].replace("nsga_sub_enemies", "NSGA-II/1-4")
    colors = [
        {
            "max": "#B8860B",
            "mean": "#5F9EA0",
            "std": "#5F9EA0",
            "defeated": "#5F9EA0",
            "max_defeated": "#B8860B",
            "euclidean": "#F5DEB3",
            "hamming": "#2F4F4F"
        },
        {
            "max": "#FFA500",
            "mean": "#FF4500",
            "std": "#FF4500",
            "defeated": "#FF4500",
            "max_defeated": "#FFA500",
            "euclidean": "#800080",
            "hamming": "#FFA500"
        },
        {
            "max": "#B8860B",
            "mean": "#5F9EA0",
            "std": "#5F9EA0",
            "defeated": "#5F9EA0",
            "max_defeated": "#B8860B",
            "euclidean": "#F5DEB3",
            "hamming": "#2F4F4F"
        },
        {
            "max": "#FFA500",
            "mean": "#FF4500",
            "std": "#FF4500",
            "defeated": "#FF4500",
            "max_defeated": "#FFA500",
            "euclidean": "#800080",
            "hamming": "#FFA500"
        }
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax_fitness = plot_fitness(
        ax1, df_data, colors,
        title_size=20, axes_label_size=20,
        ticks_size=16, legend_size=20
    )
    #fig.savefig("../a2_plots/fitness_plot.png", dpi=400)

    ax_defeated = plot_defeated(
        ax2, df_data, colors,
        title_size=20, axes_label_size=20,
        ticks_size=16, legend_size=20
    )
    #fig.savefig("../a2_plots/defeated_plot.png", dpi=400)

    fig.suptitle('Fitness and Defeated Metric for Algorithms', fontsize=16)

    ax_defeated.yaxis.set_ticks_position('right')
    ax_defeated.yaxis.set_label_position('right')

    ax_fitness.legend(loc='lower right', fontsize=14)
    ax_defeated.legend(loc='lower right', fontsize=14)

    ax_fitness.set_title("Population's Average and Best\nFitness Averaged Over 10 Runs", fontsize=14)
    ax_defeated.set_title("Population's Average and Highest Number\n of Defeated Enemies Over 10 Runs", fontsize=14)
    
    fig.text(0.5, -0.04, 'Generations', ha='center', va='center')

    # Adjust layout to minimize gaps
    plt.subplots_adjust(wspace=0)
    # Adjust layout
    fig.tight_layout()
    # Show the combined figure
    fig.savefig("../a2_plots/fitness_defeated.png", dpi=400)