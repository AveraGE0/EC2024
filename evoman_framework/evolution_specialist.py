"""File to run the neural evolution"""
import os
import random
import pickle
import yaml
import numpy as np
from tqdm import tqdm
from deap import base, creator, tools
from evoman.environment import Environment
from neural_controller import NeuralController
from naming import get_timed_name
from plots import plot_stats


def evolution(
        toolbox: base.Toolbox,
        config: dict,
        stats: tools.Statistics,
        logbook: tools.Logbook
    ) -> tuple:
    """Main evolution function. Adjusted from:
    https://deap.readthedocs.io/en/master/overview.html [18.09.2024]

    Args:
        toolbox (_type_): _description_
        config (dict): _description_

    Returns:
        _type_: returns the population as well as the logbook
    """
    mu = config["population_size"]
    n_elites = config["elitism_size"]
    pop = toolbox.population(n=mu)
    n_generations = config["generations"]
    crossover_prob = config["p_crossover"]
    mutation_prob = config["p_mutation"]

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    max_fitness = max([ind.fitness.values for ind in pop])
    pbar = tqdm(range(n_generations))

    # evaluate initial population
    logbook.record(gen=0, **stats.compile(pop))

    for n_generation in pbar:
        # Select the next generation individuals (half of population)
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # elitism with len(pop)-len(offspring) elites
        # pop.sort(reverse=True, key=lambda x: x.fitness.values)
        # pop[-len(offspring):] = offspring

        # lambda + mu replacement
        #pop = sorted(pop + offspring, key=lambda x: x.fitness.values, reverse=True)[:len(pop)]

        # Tournament selection (with elitism)
        elite = tools.selBest(pop, n_elites)
        pop = toolbox.replace(pop + offspring, k=mu-n_elites)
        pop.extend(elite)

        # make sure population size stays constant
        assert mu == len(pop)

        max_fitness = max(max_fitness, *[ind.fitness.values for ind in pop])
        # record value for generations
        logbook.record(gen=(n_generation+1), **stats.compile(pop))

        pbar.set_postfix(current_max_fitness=f"{float(max_fitness[0])}")

    return pop, logbook


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
    EXPERIMENT_NAME = os.path.join("../experiments/", config["name"])

    # initialize directories for running the experiment
    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    nc = NeuralController(
        n_inputs=config["n_inputs"],
        n_outputs=config["n_outputs"],
        hidden_size=config["hidden_size"]
    )

    # add correct individual size to config
    config["individual_size"] = nc.get_genome_size()

    env = Environment(
        experiment_name=EXPERIMENT_NAME,  # this is actually a path!
        multiplemode="no",
        enemies=config["train_enemy"],
        player_controller=nc,
        visuals=False,
    )

    def evaluate(individual: list) -> tuple[float]:
        """Function to get a fitness score for a single individual.

        Args:
            individual (list): individual (list representation)

        Returns:
            tuple: the fitness (as tuple with one value)
        """
        np.random.seed(42)
        default_fitness, p_life, e_life, time = env.play(
            pcont=individual  # pcont is actually the genome (bad naming)
        )
        return default_fitness,

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # individual + initialization
    toolbox.register(
        "attribute",
        random.uniform,
        config["init_low"],  # lower bound
        config["init_up"]  # upper bound
    )

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attribute,
        n=config["individual_size"]
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # mating, mutation, selection (for mating and mutation),
    # replacement selection and evaluation function

    #toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mate",
        tools.cxSimulatedBinary,
        eta=config["SBX_eta"]
    )
    #toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=config["polynomial_eta"],
        low=config["polynomial_low"],
        up=config["polynomial_up"],
        indpb=config["polynomial_indpb"]
    )

    toolbox.register("select", tools.selTournament, tournsize=config["sel_tournament_size"])
    toolbox.register("replace", tools.selTournament, tournsize=config["rep_tournament_size"])
    toolbox.register("evaluate", evaluate)

    # defining statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    final_population, logbook = evolution(
        toolbox,
        config,
        stats,
        tools.Logbook()
    )

    # save logbook!
    with open(os.path.join(EXPERIMENT_NAME, 'logbook.pkl'), 'wb') as lb_file:
        pickle.dump(logbook, lb_file)

    print(logbook)

    # save config!
    with open(os.path.join(EXPERIMENT_NAME, "config.yaml"), "w", encoding="utf-8") as c_file:
        yaml.dump(config, c_file, default_flow_style=False)

    # make plots and save them!
    fig = plot_stats(logbook)
    fig.savefig(os.path.join(EXPERIMENT_NAME, "stats.png"), format="png")

    # get best individual (sort descending!)
    final_population.sort(reverse=True, key=lambda x: x.fitness.values)
    best_individual = final_population[0]

    # replay the trained individual
    #env.visuals = True
    #env.speed = "normal"
    #env.multiplemode = "yes"
    #env.enemies = config["test_enemies"]
    #np.random.seed(42)

    #final_fitness, *_ = env.play(pcont=best_individual)
    #print(f"final fitness: {final_fitness}")


if __name__ == '__main__':

    for config_name in ["config_low_g.yaml", "config_high_g.yaml"]:
        # Load the configuration from a YAML file
        with open(f"../{config_name}", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # get same postfix for all runs
        EXPERIMENT_NAME = get_timed_name(prefix=config['name'])

        # run experiments
        for run in range(config["repeat"]):
            config["name"] = f"{EXPERIMENT_NAME}_{run}"
            run_experiment(config)
