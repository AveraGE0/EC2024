"""Module for the selection utility functions."""
from deap import base, tools


def register_selection(toolbox: base.Toolbox, config: dict) -> None:
    """Function to register the selection mechanism (parent, selection, migration).

    Args:
        toolbox (base.Toolbox): Toolbox where functions are registered
        config (dict): Config dict containing configuration parameters.
    """
    toolbox.register(
        "select_parents", 
        tools.selTournament,
        tournsize=config["sel_tournament_size"],
        fit_attr=config["sel_metric"])

    toolbox.register(
        "select_replacement",
        tools.selTournament,
        tournsize=config["rep_tournament_size"],
        fit_attr=config["rep_metric"]
    )

    toolbox.register(
        "select_migrants",
        tools.selTournament,
        tournsize=config["migration_tournament_size"],
        fit_attr=config["mig_metric"]
    )