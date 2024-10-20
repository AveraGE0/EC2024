"""File to run the neural evolution"""
import os
import pickle
import itertools
import time
import yaml
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from deap import base, creator, tools

from ea.fitness_functions import default_fitness
from ea.fitness_sharing import euclidean, hamming, same_loss, share_fitness
from ea.individual import (
    enforce_individual_bounds,
    set_fitness_multi_objective,
    set_individual_properties
)
from ea.offspring_creation import create_offspring, register_recombination
from ea.populations import register_population
from ea.replacement import replace
from ea.selection import register_selection
from ea.stats import create_island_stats, create_population_stats
from ea.fitness_weighting import fitness_weighting_factory
from ea.serialization import save_best_individual

from figures.plots import plot_island_metric, plot_population_metric, plot_stats

from island_model import IslandsModel
from neural_controller import NeuralController
from parallel_environment import ParallelEnvironment
from visualize import show_run


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

    fitness_sharing_func_sel = None
    if hasattr(toolbox, "fitness_sharing_sel") and callable(toolbox.fitness_sharing_sel):
        fitness_sharing_func_sel = toolbox.fitness_sharing_sel

    fitness_sharing_func_rec = None
    if hasattr(toolbox, "fitness_sharing_rec") and callable(toolbox.fitness_sharing_rec):
        fitness_sharing_func_rec = toolbox.fitness_sharing_rec

    mu = config["population_size"]

    islands = IslandsModel(
        toolbox,
        n_islands=config["islands"],
        seed=config["seed"],
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
            "fitness_sharing": fitness_sharing_func_sel
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

        if config["sharing_type"] == "global":
            # create a list containing one large population to share
            sharing_populations = [
                list(itertools.chain(islands.get_total_population(), *islands_offspring))
            ]
        elif config["sharing_type"] == "islands":
            # create a population for each island
            sharing_populations = [
                island + offspring for island, offspring in
                zip(islands.get_islands(), islands_offspring)
            ]
        else:
            raise ValueError(f"Fitness sharing type: {config['sharing_type']} does not exist")
        # update the fitness of all individuals since it might change
        for share_pop in sharing_populations:
            # this needs to be done for fitness sharing, modified in place!
            toolbox.calculate_fitness(
                share_pop,
                fitness_weighter=fitness_weights,
                fitness_sharing=fitness_sharing_func_rec
            )

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

        if config["sharing_type"] == "global":
            # create a list containing one large population to share
            sharing_populations = [islands.get_total_population()]
        elif config["sharing_type"] == "islands":
            # create a population for each island
            sharing_populations = islands.get_islands()
        else:
            raise ValueError(f"Fitness sharing type: {config['sharing_type']} does not exist")
        # re-evaluate fitness in case of fitness sharing
        for share_pop in sharing_populations:
            toolbox.calculate_fitness(
                share_pop,
                fitness_weighter=fitness_weights,
                fitness_sharing=fitness_sharing_func_sel
            )

        fitness_weights.update(
            metrics = {
                "fitnesses": np.array(
                    [ind.fitnesses for ind in islands.get_total_population()]
                ).mean(axis=0),
                "defeated": np.array(
                    [ind.defeated for ind in islands.get_total_population()]
                ).mean(axis=0)
            },
            progress=current_gen/config["generations"]
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

        for chapter, metric, plot_name in config["island_plots"]:
            fig = plot_island_metric(
                island_logbook,
                metric,
                config["migration_interval"],
                chapter=chapter
            )
            fig.savefig(
                os.path.join('../experiments', config["name"], plot_name),
                dpi=config["plot_dpi"]
            )
            plt.close(fig)

        for chapter, metric, plot_name in config["population_plots"]:
            fig = plot_population_metric(
                logbook,
                metric,
                config["migration_interval"],
                chapter=chapter
            )
            fig.savefig(
                os.path.join('../experiments', config["name"], plot_name),
                dpi=config["plot_dpi"]
            )
            plt.close(fig)

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
        n_processes=config["n_processes"],
        config=config,
        fitness_func=globals().get(config["fitness_func"])
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

    if config["fitness_sharing_sel"]:
        toolbox.register(
            "fitness_sharing_sel",
            share_fitness,
            distance_func=globals().get(config["distance_func"]),
            distance_property=config["distance_property"],
            sigma=config["sigma"]
        )

    if config["fitness_sharing_rec"]:
        toolbox.register(
            "fitness_sharing_rec",
            share_fitness,
            distance_func=globals().get(config["distance_func"]),
            distance_property=config["distance_property"],
            sigma=config["sigma"]
        )

    toolbox.register("simulate", simulate)

    toolbox.register("calculate_fitness", set_fitness_multi_objective)

    toolbox.register(
        "constrain_individual",
        enforce_individual_bounds,
        lower_bound=config["allele_lower_limit"],
        upper_bound=config["allele_upper_limit"]
    )

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

    for save_by_metric, best in config["save_best_individual"]:
        save_best_individual(
            experiment_name,
            final_islands.get_total_population(),
            metric=save_by_metric,
            best=best
        )
    
    p_env.stop_processes()


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    CONFIG_NAME = "config_island_sel_enemy.yaml"
    # Load the configuration from a YAML file
    with open(f"../{CONFIG_NAME}", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # run experiment
    for i_run, seed in zip(list(range(config['n_repetitions'])), config['seeds']):
        config["name"] = CONFIG_NAME.split(".")[0] + f"_{i_run}"
        config["seed"] = seed
        run_experiment(config)
        # let cpu cool down
        time.sleep(60)

    #for metric, metric_best in config["save_best_individual"]:
    #    with open(
    #        os.path.join('../experiments', config["name"], f"best_individual_{metric}.pkl"),
    #        mode="rb"
    #    ) as f_ind:
    #        individual = pickle.load(f_ind)
    #
    #        show_run(individual=individual, enemies=[1, 2, 3, 4, 5, 6, 7, 8], config=config)
