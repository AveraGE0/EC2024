"""Module for the selection utility functions."""
from deap import base, tools
from diversity_metrics import fitness_sharing
from scipy.spatial.distance import euclidean


def register_selection(toolbox: base.Toolbox, config: dict) -> None:
    """Function to register the selection mechanism (parent, selection, migration).

    Args:
        toolbox (base.Toolbox): Toolbox where functions are registered
        config (dict): Config dict containing configuration parameters.
    """
    toolbox.register("select_parents", tools.selTournament, tournsize=config["sel_tournament_size"])

    toolbox.register(
        "select_replacement",
        #fitness_sharing,
        #distance_func=euclidean,  # same_loss,  # hamming_distance,
        #distance_property=None,  # "defeated"
        #tournsize=config["rep_tournament_size"],
        #sigma=config["sigma"]
        tools.selTournament,
        tournsize=config["rep_tournament_size"]
    )

    toolbox.register(
        "select_migrants",
        tools.selTournament,
        tournsize=config["migration_tournament_size"]
    )