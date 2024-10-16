"""Module to provide population replacement functions"""
from deap import tools, base


def replace(
        population: list,
        offspring: list,
        toolbox: base.Toolbox,
        n_elites: int,
        elite_metrics: list[str],
        mu: int
    ) -> list:
    """Function to replace the current population (can be island) given the current population and
    offspring.

    Args:
        population (list): Population being replaced (also for replacement selection).
        offspring (list): Offspring for replacement selection.
        toolbox (base.Toolbox): Toolbox with selection and replacement function.
        n_elites (int): Number of elites (best individuals) that are not replaced.
        elite_metrics (list[str]): Metrics used to select elites. If multiple are given, the elite
                                   is split by the metrics, so that n_elites will be elites.
        mu (int): Size of the replacement population (should be population size in most cases).

    Returns:
        list: Replacement population (next generation).
    """
    previous_population_size = len(population)

    #elite = tools.selBest(population, n_elites, fit_attr=elite_metric)
    elite = []
    for metric_name in elite_metrics:
        elite += tools.selBest(population, n_elites//len(elite_metrics), fit_attr=metric_name)

    if len(elite) != n_elites:
        raise ValueError(
            f"Elite metrics can not be divided by the given n_elites: {len(elite_metrics)},"\
            f" {n_elites}"
        )

    population = toolbox.select_replacement(population + offspring, k=mu-n_elites)
    population.extend(elite)

    # make sure population size stays constant
    if not previous_population_size == len(population):
        raise ValueError(
            f"Population changes size >:(, before: {previous_population_size},"\
            f" after: {len(population)}"
        )
    return population
