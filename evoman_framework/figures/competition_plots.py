import pickle
import numpy as np
from figures.plots import plot_island_metric


if __name__ == "__main__":
    # with open("../experiments/competition_test_best_8_496/best_individual_multi_gain.pkl", mode="rb") as ind_file:
    #     best_ind = pickle.load(ind_file)
    # np.savetxt("../experiments/competition_test_best_8_493/best_individual_multi_gain.txt", best_ind)

    # exit()
    
    with open("../experiments/config_island_final_6/logbook.pkl", mode="rb") as stats_file:
        stats = pickle.load(stats_file)
    gain_stats = stats.chapters["gain"]
    diver_stats = stats.chapters["diversity_stats"]
    #gen = stats.chapters["gen"]
    print(gain_stats)

    exit()


    with open("../experiments/competition_test_best_8_496/final_population.pkl", mode="rb") as p_file:
        population = pickle.load(p_file)
    
    for i, island in enumerate(population):
        print(f"Island {i} gain_mean: {np.array([ind.fitness.values[0] for ind in island]).mean()}")
        print(f"Island {i} gain_max: {np.array([ind.fitness.values[0] for ind in island]).max()}")