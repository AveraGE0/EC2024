"""Module presenting a class to represent a variable changing by a schedule."""
from abc import ABC, abstractmethod
import numpy as np


class FitnessWeighting(ABC):
    """Interface for fitness weighting."""

    @abstractmethod
    def update(self, metrics: dict[str, np.ndarray], progress: float) -> None:
        """Update function to update the weights dynamically.

        Args:
            metrics dict[str, np.ndarray]: Dict with metrics from the simulation run
                     that can be used to update the weighter.
            progress (float): Percentual progress in relation to n_generations.

        Raises:
            NotImplementedError: Abstract methods, an Exception will be thrown if not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def get_weighted_fitness(self, fitness_ind: np.ndarray) -> float:
        """Gets the fitness for an individual based on the fitness values returned
        to the specific individual. The fitness values will be weighted for each
        enemy with the current weights.

        Args:
            fitness_ind (np.ndarray): List of fitnesses for a single individual.

        Raises:
            NotImplementedError: Abstract methods, an Exception will be thrown if not implemented.

        Returns:
            float: Fitness value for individual.
        """
        raise NotImplementedError


class FitnessChangeWeighting(FitnessWeighting):
    """Schedule for weights for the fitness function. To encourage increasing the fitness against
    all enemies, the fitness values of the enemies are weighted depending on the last time they
    have been changed."""
    def __init__(
            self,
            n_enemies: int,
            relative_increase: float = 0.05,
            decrease_threshold: float = 70,
            min_weight: float = 0.01,
            max_weight: float = 10.0
        ) -> None:
        """Constructor for dynamic weight scheduler.

        Args:
            n_enemies (int): amount of enemies (weights) we will have
            relative_increase (float, optional): How much a weight is decreased/increased.
                                                 Defaults to 0.1.
            decrease_threshold (float): The average fitness that has to be reached in order
                                        to decrease the weight. Defaults to 0.5.
            min_weight (float): The minimal value a weight can take (especially to 
                                avoid negative weights). Defaults to 0.01.
            max_weight (float): The maximal value a weight can take. Defaults to 10.0.
        """
        self.weights = np.array([1.0] * n_enemies)
        self.relative_increase = relative_increase
        self.last_fitness = np.zeros_like(self.weights)

        self.decrease_threshold = decrease_threshold
        self.min_weight = min_weight
        self.max_weight = max_weight

    def update(self, metrics: dict[str, np.ndarray], progress: float) -> None:
        """Gets the next weighted sum of fitness values.

        Args:
            metrics: dict[str, np.ndarray]: Metrics to update the weighter
                                            (fitness means are used).
            progress (float): Percentual progress in relation to n_generations.
        """
        fitness_avg = metrics["fitnesses"]
        change = fitness_avg - self.last_fitness
        self.last_fitness = fitness_avg

        added_weight = np.where(change < 0, self.relative_increase, -self.relative_increase)
        if abs(added_weight.sum()) == self.relative_increase * len(self.weights):
            added_weight = np.zeros_like(added_weight)

        added_weight[(fitness_avg < self.decrease_threshold) & (added_weight < 0)] = 0.0

        self.weights += added_weight
        self.weights = np.clip(self.weights, self.min_weight, self.max_weight)

    def get_weighted_fitness(self, fitness_ind: np.ndarray) -> float:
        """Method to get the weighted average fitness for one individual.

        Args:
            fitness_ind (np.ndarray): Fitnesses of enemies for one individual.

        Returns:
            float: Weighted Average Fitness with normalized weights (weights sum to one).
        """
        # weights are normalized to avoid extreme values
        return ((self.weights / self.weights.sum()) * fitness_ind).sum()


class FitnessProportionalWeighting(FitnessWeighting):
    """Schedule for weights for the fitness function. To encourage increasing the fitness against
    all enemies, the fitness values of the enemies are weighted depending on the last time they
    have been changed."""
    def __init__(
            self,
            n_enemies: int,
            step_size: float = 0.2,
            step_decay: str = "linear",
            min_step: float = 0.05,
            min_weight: float = 0.01,
            max_weight: float = 10.0
        ) -> None:
        """Constructor for dynamic weight scheduler.

        Args:
            n_enemies (int): Amount of enemies (weights) we will have.
            step_size (float): Percentage of step going towards the current fitness weights.
            min_weight (float): The minimal value a weight can take (especially to 
                                avoid negative weights). Defaults to 0.01.
            max_weight (float): The maximal value a weight can take. Defaults to 10.0.
        """
        self.weights = np.array([1.0] * n_enemies)

        self.step_size = step_size
        self.step_decay_func = {
            "linear": lambda progress: self.min_step + (self.step_size - self.min_step) * (1-progress),
            "None": lambda progress: self.step_size
        }[step_decay]
        self.min_step = min_step
        self.min_weight = min_weight
        self.max_weight = max_weight

    def update(self, metrics: dict[str, np.ndarray], progress: float) -> None:
        """Gets the next weighted sum of fitness values.

        Args:
            metrics (dict[str, np.ndarray]): Metrics to update the weighter
                                             (fitness means are used).
            progress (float): Percentual progress in relation to n_generations.
        """
        fitness_avg = metrics["fitnesses"]
        # squared_avg = fitness_avg * fitness_avg
        #current_weight = (1 - fitness_avg / 100)**2
        #current_weight = 0.5 * (1 - fitness_avg / 100) + 0.5 * (1 - fitness_avg / 100)**2
        current_weight = 1 - ((fitness_avg - fitness_avg.min()) / (fitness_avg.max() - fitness_avg.min()))
        current_step_size = self.step_decay_func(progress)
        self.weights = (1 - current_step_size) * self.weights + current_step_size * current_weight
        self.weights = np.clip(self.weights, self.min_weight, self.max_weight)

    def get_weighted_fitness(self, fitness_ind: np.ndarray) -> float:
        """Method to get the weighted average fitness for one individual.

        Args:
            fitness_ind (np.ndarray): Fitnesses of enemies for one individual.

        Returns:
            float: Weighted Average Fitness with normalized weights (weights sum to one).
        """
        # weights are normalized to avoid extreme values
        return ((self.weights / self.weights.sum()) * fitness_ind).sum()


class DefeatedProportionalWeighting(FitnessWeighting):
    """Fitness weighter that scales the fitness scores according to the amount
    of percentage of representations that defeat this enemy. Ideally this increases
    the weights in a way that hard to beat enemies are also regarded."""
    def __init__(
            self,
            n_enemies: int,
            step_size: int = 0.5,
            min_weight: float = 0.01,
            max_weight: float = 10.0
        ) -> None:
        self.weights = np.ones(shape=(n_enemies,)) / n_enemies
        self.step_size = step_size

        self.min_weight = min_weight
        self.max_weight = max_weight

    def update(self, metrics: dict[str, np.ndarray], progress: float) -> None:
        """Updates the fitness weights based on the percentage at which
        enemies are defeated at the moment.

        Args:
            metrics (dict[str, np.ndarray]): Metrics to update the weighter
                                            (defeated means are used).
            progress (float): Percentual progress in relation to n_generations.

        """
        defeated = metrics["defeated"]
        # min_defeated = defeated.min()
        proportional_change = 1 / (defeated + 1e-2)

        new_weights = np.clip(
            self.weights + proportional_change * self.step_size,
            self.min_weight,
            self.max_weight
        )

        self.weights =  new_weights / new_weights.sum()

    def get_weighted_fitness(self, fitness_ind: np.ndarray) -> float:
        return (self.weights * fitness_ind).sum()


class MeanWeighting:
    """Simple mean fitness weighter."""
    def __init__(self) -> None:
        pass

    def update(self, metrics: dict[str, np.ndarray], progress: float) -> None:
        """Unused function since we dont depend on any parameters."""
        return

    def get_weighted_fitness(self, fitness_ind: np.ndarray) -> float:
        """Method to get the mean of the finesses."""
        return fitness_ind.mean()


class MinWeighting:
    """Class to represent the fitness of an individual by its worst performance"""
    def __init__(self) -> None:
        pass

    def update(self, metrics: dict[str, np.ndarray], progress: float) -> None:
        """Unused function since we dont depend on any parameters."""
        return

    def get_weighted_fitness(self, fitness_ind: np.ndarray) -> float:
        """Method to get the mean of the finesses."""
        return fitness_ind.min()


class FitnessWeightingFactory():
    """Factory to get the fitness weighters."""
    def __init__(self) -> None:
        self.weighters = {}

    def register(self, fw_name: str, fw: FitnessWeighting) -> None:
        """Registers an existing class to the Factory.

        Args:
            fw_name (str): Name of the weighter.
            fw (FitnessWeighting): Class to instantiate the weighter.
        """
        self.weighters.update({fw_name: fw})

    def get_weighter(self, name: str, **kwargs) -> FitnessWeighting:
        """Method to get the fitness weighter class.

        Args:
            name (str): Name of the registered fitness weighter.
            kwargs (dict): parameters with which the weighter will be instantiated.

        Returns:
            FitnessWeighting: Fitness weighter object.
        """
        return self.weighters[name](**kwargs)


fitness_weighting_factory = FitnessWeightingFactory()

fitness_weighting_factory.register("mean", MeanWeighting)
fitness_weighting_factory.register("change", FitnessChangeWeighting)
fitness_weighting_factory.register("proportional", FitnessProportionalWeighting)
fitness_weighting_factory.register("defeated_proportional", DefeatedProportionalWeighting)
