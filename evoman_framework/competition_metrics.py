"""Module that provides measures of the competition"""
import numpy as np

# 1 number of defeated enemies
# 1.1 sum of player life (higher better)
# 1.2 time taken (lower=better)
def defeated_enemies(enemy_lives: np.array) -> int:
    """Function to get  the number of defeated enemies.

    Args:
        enemy_lives (np.array): Enemy lives.

    Returns:
        int: Number of enemies defeated
    """
    n_defeated = np.sum(np.where(enemy_lives == 0, 1, 0))
    return n_defeated


# 2.Gain measure
# sum of player life - sum of enemy life
def multi_gain(player_lives: np.array, enemy_lives: np.array) -> int:
    """Function to calculate the gain for the enemies given
    all scores of one individual for all enemies.

    Args:
        player_lives (np.array): Array of player lives.
        enemy_lives (np.array): Array of enemy lives.

    Returns:
        int: Total gain over all enemies.
    """
    return np.sum(player_lives) - np.sum(enemy_lives)
