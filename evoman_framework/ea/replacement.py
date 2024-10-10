"""Module to provide population replacement functions"""
from deap import tools, base


def replace(
        population: list,
        offspring: list,
        toolbox: base.Toolbox,
        n_elites: int,
        elite_metric: str,
        mu: int
    ) -> list:
    """Function to replace the current population (can be island) given the current population and
    offspring.

    Args:
        population (list): Population being replaced (also for replacement selection).
        offspring (list): Offspring for replacement selection.
        toolbox (base.Toolbox): Toolbox with selection and replacement function.
        n_elites (int): Number of elites (best individuals) that are not replaced.
        elite_metric (str): Metric used to select elites. If multiple are given, the elite
                            is split by the metrics.
        mu (int): Size of the replacement population (should be population size in most cases).

    Returns:
        list: Replacement population (next generation).
    """
    previous_population_size = len(population)

    #elite = []
    #for metric_name in elite_metrics:
    #    elite.append(tools.selBest(population, n_elites//len(elite_metrics), fit_attr=metric_name))

    #if len(elite) != n_elites:
    #    raise ValueError(
    #        f"Elite metrics can not be divided by the given n_elites: {len(elite_metrics)}, {n_elites}"
    #    )

    elite = tools.selBest(population, n_elites, fit_attr=elite_metric)

    population = toolbox.select_replacement(population + offspring, k=mu-n_elites)
    population.extend(elite)

    # make sure population size stays constant
    assert previous_population_size == len(population)
    return population
