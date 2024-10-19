import os
import pickle
import pandas as pd
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller


if __name__ == "__main__":
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    # # Box-plot data first: take best solution, simulate against enemy and obtain the sum of the gain
    # df_gain_results = pd.DataFrame({"algorithm": [], "run": [], "gain_sum": []})
    
    # island_all = [
    #     ("island_all_enemies", os.path.join("../experiments", directory))
    #     for directory in os.listdir("../experiments") if "config_island_final" in directory
    # ]
    # island_sub = [
    #     ("island_sub_enemies", os.path.join("../experiments", directory))
    #     for directory in os.listdir("../experiments") if "config_island_sel_enemy" in directory
    # ]

    # for algorithm, experiment_path in island_all + island_sub:
    #     best_ind_path = os.path.join(experiment_path, "best_individual_multi_gain.pkl")
    #     #if not os.path.isfile(best_ind_path):
    #     #    continue

    #     with open(best_ind_path, mode="rb") as b_file:
    #         individual = np.array(pickle.load(b_file))
        
    #     total_gain = 0
    #     for enemy in range(1, 9):
    #         env = Environment(
    #                 experiment_name="test",
    #                 enemies=[enemy],
    #                 playermode="ai",
    #                 player_controller=player_controller(10),
    #                 enemymode="static",
    #                 level=2,
    #                 speed="fastest")
    #         f, p, e, t = env.play(pcont=individual)
    #         total_gain = total_gain + p - e
        
    #     df_gain_results = pd.concat([
    #         df_gain_results,
    #         pd.DataFrame({"algorithm": [algorithm], "run": [experiment_path[-1]], "gain_sum": [total_gain]})
    #     ])
    
    # df_gain_results.to_csv("../a2_plots/islands_gain.csv")

    # # Next: data for plotting the average/std of the mean and maximum fitness across the generations using a line-plot
    # df_fitness_stats = pd.DataFrame()
    # for experiment_base, algorithm in [("config_island_final", "island_all_enemies"), ('config_island_sel_enemy', "island_sub_enemies")]:
    #     experiment_paths = [os.path.join("../experiments", directory) for directory in os.listdir("../experiments") if experiment_base in directory]

    #     avg_fitness = np.empty(shape=(0, 350))
    #     max_fitness = np.empty(shape=(0, 350))
    #     std_fitness = np.empty(shape=(0, 350))
    #     avg_defeated = np.empty(shape=(0, 350))
    #     max_defeated = np.empty(shape=(0, 350))
        
    #     for experiment_path in experiment_paths:
    #         logbook_path = os.path.join(experiment_path, "logbook.pkl")
    #         #if not os.path.isfile(logbook_path):
    #         #    continue

    #         with open(logbook_path, mode="rb") as log_file:
    #             multi_logs = pickle.load(log_file)

    #         unweighted_fitness = multi_logs.chapters["fitnesses"]
    #         defeated_stats = multi_logs.chapters["defeated"]

    #         avg_fitness = np.concatenate(
    #             [
    #                 avg_fitness,
    #                 np.expand_dims(np.array(unweighted_fitness.select("avg")).mean(axis=1), axis=0)
    #             ],
    #             axis=0
    #         )
    #         max_fitness = np.concatenate(
    #             [
    #                 max_fitness,
    #                 np.expand_dims(np.array(unweighted_fitness.select("max")).mean(axis=1), axis=0)
    #             ],
    #             axis=0
    #         )
    #         std_fitness = np.concatenate(
    #             [
    #                 std_fitness,
    #                 np.expand_dims(np.array(unweighted_fitness.select("std")).mean(axis=1), axis=0)
    #             ],
    #             axis=0
    #         )
    #         avg_defeated = np.concatenate(
    #             [
    #                 avg_defeated,
    #                 np.expand_dims(np.array(defeated_stats.select("avg")), axis=0)
    #             ],
    #             axis=0
    #         )
    #         max_defeated = np.concatenate(
    #             [
    #                 max_defeated,
    #                 np.expand_dims(np.array(defeated_stats.select("max")), axis=0)
    #             ],
    #             axis=0
    #         )
    #         generations = multi_logs.select("gen")

    #     df_fitness_stats = pd.concat([
    #         df_fitness_stats,
    #         pd.DataFrame({
    #             "algorithm": [algorithm] * len(generations),
    #             "generation": generations,
    #             "avg": avg_fitness.mean(axis=0),
    #             "max": max_fitness.mean(axis=0),
    #             "std": std_fitness.mean(axis=0),
    #             "defeated": avg_defeated.mean(axis=0),
    #             "max_defeated": max_defeated.mean(axis=0)
    #     })])

    # df_fitness_stats.to_csv("../a2_plots/averaged_fitness_data_island.csv")


    ## NSGA-II
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    # # Box-plot data first: take best solution, simulate against enemy and obtain the sum of the gain
    # df_gain_results = pd.DataFrame({"algorithm": [], "run": [], "gain_sum": []})
    
    # nsga_all = [
    #     ("nsga_all_enemies", os.path.join("../experiments", directory))
    #     for directory in os.listdir("../experiments") if "config_nsga_all" in directory
    # ]
    # nsga_sub = [
    #     ("nsga_sub_enemies", os.path.join("../experiments", directory))
    #     for directory in os.listdir("../experiments") if "config_nsga_sub" in directory
    # ]

    # for algorithm, experiment_path in nsga_all + nsga_sub:
    #     individual = np.loadtxt(os.path.join(experiment_path, "best_gain_solution.csv"))
    #     print(individual.shape)
        
    #     total_gain = 0
    #     for enemy in range(1, 9):
    #         env = Environment(
    #                 experiment_name="test",
    #                 enemies=[enemy],
    #                 playermode="ai",
    #                 player_controller=player_controller(10),
    #                 enemymode="static",
    #                 level=2,
    #                 speed="fastest")
    #         f, p, e, t = env.play(pcont=individual)
    #         total_gain = total_gain + p - e
        
    #     df_gain_results = pd.concat([
    #         df_gain_results,
    #         pd.DataFrame({"algorithm": [algorithm], "run": [experiment_path[-1]], "gain_sum": [total_gain]})
    #     ])
    
    # df_gain_results.to_csv("../a2_plots/nsga_gain.csv")


    # Next: data for plotting the average/std of the mean and maximum fitness across the generations using a line-plot
    df_fitness_stats = pd.DataFrame()
    for experiment_base, algorithm in [("config_nsga_all", "nsga_all_enemies"), ('config_nsga_sub', "nsga_sub_enemies")]:
        experiment_paths = [os.path.join("../experiments", directory) for directory in os.listdir("../experiments") if experiment_base in directory]

        avg_fitness = np.empty(shape=(0, 3))
        max_fitness = np.empty(shape=(0, 3))
        std_fitness = np.empty(shape=(0, 3))
        avg_defeated = np.empty(shape=(0, 3))
        max_defeated = np.empty(shape=(0, 3))
        avg_euclidean = np.empty(shape=(0, 3))
        
        for experiment_path in experiment_paths:
            logbook_path = os.path.join(experiment_path, "logs.pkl")
            #if not os.path.isfile(logbook_path):
            #    continue
            with open(logbook_path, mode="rb") as log_file:
                df_multi_logs = pd.DataFrame(pickle.load(log_file))

            avg_fitness = np.concatenate(
                [
                    avg_fitness,
                    np.expand_dims(df_multi_logs["avg_fitness"].to_numpy(), axis=0)
                ],
                axis=0
            )
            max_fitness = np.concatenate(
                [
                    max_fitness,
                    np.expand_dims(df_multi_logs["best_fitness"].to_numpy(), axis=0)
                ],
                axis=0
            )
            std_fitness = np.concatenate(
                [
                    std_fitness,
                    np.expand_dims(df_multi_logs["std_fitness"].to_numpy(), axis=0)
                ],
                axis=0
            )
            avg_defeated = np.concatenate(
                [
                    avg_defeated,
                    np.expand_dims(df_multi_logs["avg_defeated"].to_numpy(), axis=0)
                ],
                axis=0
            )
            max_defeated = np.concatenate(
                [
                    max_defeated,
                    np.expand_dims(df_multi_logs["max_defeated"].to_numpy(), axis=0)
                ],
                axis=0
            )
            avg_euclidean = np.concatenate(
                [
                    avg_euclidean,
                    np.expand_dims(df_multi_logs["avg_euclidean"].to_numpy(), axis=0)
                ],
                axis=0
            )
            generations = df_multi_logs["generation"].to_numpy()

        df_fitness_stats = pd.concat([
            df_fitness_stats,
            pd.DataFrame({
                "algorithm": [algorithm] * len(generations),
                "generation": generations,
                "avg": avg_fitness.mean(axis=0),
                "max": max_fitness.mean(axis=0),
                "std": std_fitness.mean(axis=0),
                "defeated": avg_defeated.mean(axis=0),
                "max_defeated": max_defeated.mean(axis=0),
                "avg_euclidean": avg_euclidean.mean(axis=0)
        })])

    df_fitness_stats.to_csv("../a2_plots/averaged_fitness_data_nsga.csv")
