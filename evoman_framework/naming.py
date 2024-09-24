"""Module for generating names by time"""
import datetime


def get_timed_name(prefix: str) -> str:
    """Function to make a prefix based on the current time and
    a given prefix.

    Args:
        prefix (str): prefix of the name (non-time part)

    Returns:
        str: prefix combined with the current time
    """
    return f"{prefix}_{datetime.datetime.now().strftime('%d%m%Y_%H%M%S')}"
