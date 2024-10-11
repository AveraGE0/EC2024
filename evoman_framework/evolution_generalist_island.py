"""File to run the neural evolution"""
import os
import pickle
import matplotlib.pyplot as plt
import yaml
import numpy as np
from tqdm import tqdm
from deap import base, creator, tools
from ea.fitness_sharing import share_fitness
from neural_controller import NeuralController
from figures.plots import plot_stats
from visualize import show_run
from parallel_environment import ParallelEnvironment
from island_model import IslandsModel
from fitness_weighting import fitness_weighting_factory
from ea.stats import create_island_stats, create_population_stats
from ea.fitness_functions import default_fitness
from ea.offspring_creation import create_offspring, register_recombination
from ea.replacement import replace
from ea.individual import set_fitness_multi_objective, set_individual_properties
from ea.selection import register_selection
from ea.populations import register_population
from figures.plots import plot_island_metric
from ea.fitness_sharing import hamming, euclidean, same_loss


def evolution(
        toolbox: base.Toolbox,
        config: dict[str, any],
        general_stats: tools.Statistics,
        island_stats: tools.Statistics,
        logbook: tools.Logbook,
        island_logbook: tools.Logbook
    ) -> tuple[IslandsModel, tools.Logbook, tools.Logbook]:
    """Main evolution function. Originally adjusted from
    https://deap.readthedocs.io/en/master/overview.html [18.09.2024].Creates a
    population, performs recombination and replacement as many times as specified
    in the config and returns the final island model.

    Args:
        toolbox (base.Toolbox): Toolbox to register EA functions.
        config (dict): Config containing EA parameters.
        stats (tools.Statistics): Stats object for creating stats.
        logbook (tools.Logbook): Logbook to store stats.

    Returns:
        tuple: Final islands, logbook, island logbook
    """
    fitness_weights = fitness_weighting_factory.get_weighter(
        config["fitness_weighting"],
        **{key.lstrip("fw_"): value for key, value in config.items() if "fw" in key}
    )

    fitness_sharing_func = None
    if hasattr(toolbox, "fitness_sharing") and callable(toolbox.fitness_sharing):
        fitness_sharing_func = toolbox.fitness_sharing

    mu = config["population_size"]

    islands = IslandsModel(
        toolbox,
        n_islands=config["islands"],
        total_population=mu,
    )

    n_elites = config["elitism_size"]
    crossover_prob = config["p_crossover"]
    mutation_prob = config["p_mutation"]

    # Evaluate the entire population
    islands.map_total_population(toolbox.simulate)

    islands.map_total_population(
        toolbox.calculate_fitness,
        parameters={
            "fitness_weighter": fitness_weights,
            "fitness_sharing": fitness_sharing_func
        }
    )

    progress_bar = tqdm(range(config["generations"]))

    for current_gen in progress_bar:

        all_new_individuals, islands_offspring = [], []
        for island_pop in islands.get_islands():
            new_offspring, offspring = create_offspring(
                island_pop,
                toolbox,
                crossover_prob,
                mutation_prob
            )
            all_new_individuals = all_new_individuals + new_offspring
            islands_offspring.append(offspring)

        toolbox.simulate(all_new_individuals)

        # update the fitness of all individuals since it might change
        for island, offspring in zip(islands.get_islands(), islands_offspring):
            # this needs to be done for fitness sharing, modified in place!
            toolbox.calculate_fitness(
                island + offspring,
                fitness_weighter=fitness_weights,
                fitness_sharing=fitness_sharing_func
            )
        #islands.map_total_population(
        #    toolbox.calculate_fitness,
        #    parameters={"scheduled_weights": fitness_weights}
        #)
        # evaluate new individuals (not in total population)
        #toolbox.calculate_fitness(all_new_individuals, fitness_weights)

        for i, offspring in enumerate(islands_offspring):
            population = islands.get_island(i)
            islands.set_island(
                i,
                replace(
                    population,
                    offspring,
                    toolbox,
                    n_elites,
                    config["elite_metric"],
                    len(population)
                )
            )

        # re-evaluate fitness in case of fitness sharing
        for island in islands.get_islands():
            toolbox.calculate_fitness(
                island,
                fitness_weighter=fitness_weights,
                fitness_sharing=fitness_sharing_func
            )

        fitness_weights.update(
            metrics = {
                "fitnesses": np.array(
                    [ind.fitnesses for ind in islands.get_total_population()]
                ).mean(axis=0),
                "defeated": np.array(
                    [ind.defeated for ind in islands.get_total_population()]
                ).mean(axis=0)
            }
        )

        if current_gen > 0 and current_gen % int(config["migration_interval"]) == 0:
            islands.migrate(
                toolbox,
                migration_rate=config["migration_rate"],
                replace_metric=config["migration_replace_metric"]
            )

        island_stat_list = islands.map_islands(island_stats.compile)
        for i, island_stat in enumerate(island_stat_list):
            island_logbook.record(gen=current_gen, island=i, **island_stat)
        # Record total population statistics
        total_stats = general_stats.compile(islands.get_total_population())
        # Log the stats for this generation
        logbook.record(gen=current_gen, **total_stats)

        # display stats above progress bar
        tqdm.write(logbook.stream)

        fig = plot_island_metric(
            island_logbook,
            "euclidean_avg", 
            config["migration_interval"],
            chapter="diversity_stats"
        )
        fig.savefig(os.path.join('../experiments', config["name"], "island_diversity.png"))
        plt.close(fig)
        fig = plot_island_metric(
            island_logbook,
            "avg",
            config["migration_interval"],
            chapter="fitness"
        )
        fig.savefig(os.path.join('../experiments', config["name"], "island_fitness.png"))
        plt.close(fig)
        plt.close()

    return islands, logbook, island_logbook


def run_experiment(config: dict) -> None:
    """Function to run an experiment given a config.
    Takes care of creating output dir, etc.

    Args:
        config (dict): config containing run parameters

    Returns:
        None: -
    """
    # cleanup toolbox
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    # create run directory
    experiment_name = os.path.join("../experiments/", config["name"])

    # initialize directories for running the experiment
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    nc = NeuralController(
        n_inputs=config["n_inputs"],
        n_outputs=config["n_outputs"],
        hidden_size=config["hidden_size"]
    )
    # add correct individual size to config
    config["individual_size"] = nc.get_genome_size()

    p_env = ParallelEnvironment(
        n_processes=16,
        config=config,
        fitness_func=default_fitness
    )

    p_env.start_processes()

    def simulate(individuals: list[list]) -> None:
        np.random.seed(42)
        results = p_env.get_results(individuals)

        for ind, metrics in zip(individuals, results):
            set_individual_properties(ind, metrics)

    toolbox = base.Toolbox()

    register_population(creator, toolbox, config)

    register_recombination(toolbox, config)

    register_selection(toolbox, config)

    #fitness_sharing,
    #distance_func=euclidean,  # same_loss,  # hamming_distance,
    #distance_property=None,  # "defeated"
    #sigma=config["sigma"]
    if config["fitness_sharing"]:
        toolbox.register(
            "fitness_sharing",
            share_fitness,
            distance_func=globals().get(config["distance_func"]),
            distance_property=config["distance_property"],
            sigma=config["sigma"]
        )

    toolbox.register("simulate", simulate)

    toolbox.register("calculate_fitness", set_fitness_multi_objective)

    multi_stats = create_population_stats()
    island_stats = create_island_stats()

    logbook = tools.Logbook()

    logbook.header = ("gen", "defeated", "fitness", "gain", "life", "time", "enemies")

    logbook.chapters["defeated"].header = ("avg", "max")
    logbook.chapters["fitness"].header = ("avg", "max", "std")
    logbook.chapters["gain"].header = ("avg_sum", "max_sum")
    logbook.chapters["life"].header = ("avg_sum", "max_sum")
    logbook.chapters["time"].header = ("avg_sum", "min_sum")
    logbook.chapters["enemies"].header = ("avg_def", "is_def")

    island_logbook = tools.Logbook()

    final_islands, logbook, island_logbook = evolution(
        toolbox,
        config,
        multi_stats,
        island_stats,
        logbook,
        island_logbook
    )

    # save logbook!
    with open(os.path.join(experiment_name, 'logbook.pkl'), 'wb') as lb_file:
        pickle.dump(logbook, lb_file)

    #print(logbook)

    # save island logbook!
    with open(os.path.join(experiment_name, 'logbook_islands.pkl'), 'wb') as lb_file:
        pickle.dump(island_logbook, lb_file)

    # save config!
    with open(os.path.join(experiment_name, "config.yaml"), "w", encoding="utf-8") as c_file:
        yaml.dump(config, c_file, default_flow_style=False)

    # make plots and save them!
    fig = plot_stats(logbook)
    fig.savefig(os.path.join(experiment_name, "stats.png"), format="png")

    # save final population
    final_islands_list = final_islands.get_islands()
    with open(os.path.join(experiment_name, 'final_population.pkl'), 'wb') as p_file:
        pickle.dump(final_islands_list, p_file)

    # get best individual (sort descending!)
    best_individual = sorted(
        final_islands.get_total_population(),
        reverse=True,
        key=lambda x: x.fitness.values[0]
    )[0]

    # save best individual (fitness)
    print(f"Saving best individual with fitness of={round(best_individual.fitness.values[0], 4)}")
    with open(os.path.join(experiment_name, 'fittest_individual.pkl'), 'wb') as i_file:
        pickle.dump(best_individual, i_file)


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    CONFIG_NAME = "config_competition_test.yaml"
    # Load the configuration from a YAML file
    with open(f"../{CONFIG_NAME}", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # run experiment
    run_experiment(config)

    with open(
        os.path.join('../experiments', config["name"], "fittest_individual.pkl"),
        mode="rb"
    ) as f_ind:
        individual = pickle.load(f_ind)

        show_run(individual=individual, enemies=[1, 2, 3, 4, 5, 6, 7, 8], config=config)
