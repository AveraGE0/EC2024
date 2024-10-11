"""Module providing a representation to work easier with island models."""
from deap import base


class IslandsModel:
    """
    Representation of an island model. Manages the different islands populations automatically.
    """
    def __init__(
        self,
        toolbox: base.Toolbox,
        n_islands: int,
        total_population: int =None,
        island_population: int =None,
    ) -> None:
        """Initializer for the representation of Islands.

        Args:
            toolbox (base.Toolbox): DEAP toolbox (to initialize the population).
            n_islands (int): number of islands created.
            total_population (int, optional): Size of the total population. Alternative to
            island_population parameter. Defaults to None.
            island_population (int, optional): Size of an individual island. Alternative to
            total_population parameter. Defaults to None.

        Raises:
            ValueError: If the number of island is smaller equal 0.
            ValueError: If we have either both total_population and island_population
            or none of them.
            ValueError: If we have more islands than individuals in the population.
        """
        self.n_islands = n_islands

        if self.n_islands <= 0:
            raise ValueError("Error, there must be at least one island!")

        # we need one and only one of them
        if (total_population and island_population) or\
           (not total_population and not island_population):
            raise ValueError("Please give either a total population or an island population")

        total_population = total_population if total_population else island_population * n_islands
        if total_population < n_islands:
            raise ValueError("Error, the population size in combination with the amount of islands"\
                             f"does not make sense: {total_population} in {self.n_islands}!?")

        self.islands_merged = toolbox.population(n=total_population)
        self.islands_borders = list(
            range(0, total_population, (total_population//n_islands))
        ) + [len(self.islands_merged)]

    def get_island(self, island_index: int) -> list:
        """Method to retrieve the island at the given index.

        Args:
            island_index (int): index of island

        Raises:
            IndexError: Raise if invalid index is given

        Returns:
            list: island, which is a subpopulation
        """
        if not 0 <= island_index < self.n_islands:
            raise IndexError(f"The given island index is out of range: {island_index}")

        return self.islands_merged[
            self.islands_borders[island_index]:
            self.islands_borders[island_index+1]
        ]

    def get_islands(self) -> list[list]:
        """Method to get all islands in a single list.

        Returns:
            list[list]: List of island (lists).
        """
        return [self.get_island(i) for i in range(self.n_islands)]

    def get_total_population(self) -> list:
        """Returns the population of all islands as a single list

        Returns:
            list: whole population
        """
        return self.islands_merged

    def set_island(self, island_index: int, new_island: list) -> None:
        """Replaces island at given index with newly given list (island).

        Args:
            island_index (int): index of island to be replaced
            new_island (list): replacement for current island
        """
        if not 0 <= island_index < self.n_islands:
            raise IndexError(f"The given island index is out of range: {island_index}")

        self.islands_merged[
            self.islands_borders[island_index]:
            self.islands_borders[island_index+1]
        ] = new_island

    def map_islands(self, func: callable, parameters: dict=None) -> any:
        """Method to run a function on each island individually.

        Args:
            func (callable): _description_
            parameters (dict, optional): Parameters of the function. Defaults to {}.

        Returns:
            list[list]: List of the returns of the func function.
        """
        if not parameters:
            parameters = {}

        results = []
        for island_index in range(self.n_islands):
            results.append(
                func(self.get_island(island_index), **parameters)
            )

        return results

    def map_total_population(self, func: callable, parameters: dict =None) -> any:
        """Method to run a function on the whole population. This could be
        something like retrieving statistics or evaluating fitness for example.

        Args:
            func (callable): function that maps a population (list of individuals)
            parameters (dict, optional): Parameters of the function. Defaults to None.

        Returns:
            any: the output of the func function.
        """
        if not parameters:
            parameters = {}

        return func(self.islands_merged, **parameters)

    def migrate(self, toolbox: base.Toolbox, replace_metric: str, migration_rate: float = 0.1) -> None:
        """Method to perform migration between islands. The migration rate is
        the percentage of individuals exchanged.

        Args:
            toolbox (base.Toolbox): toolbox
            migration_rate (float, optional): percentage of swapped individuals.
            Defaults to 0.1.
        """
        island_size = len(self.get_total_population()) // self.n_islands
        swap_size = int(island_size * migration_rate)

        # collect migrants
        emigrants = []
        for i in range(self.n_islands):
            #i_second_island = (i + 1) % self.n_islands
            island_emigrants = toolbox.select_migrants(self.get_island(i), k=swap_size)
            #immigrants = toolbox.select_migrants(self.get_island(i_second_island), k=swap_size)
            # we have to copy otherwise we dont swap
            emigrants.append(list(map(toolbox.clone, island_emigrants)))
            #immigrants = list(map(toolbox.clone, immigrants))

        for i in range(self.n_islands):
            self.set_island(
                i,
                sorted(
                    self.get_island(i),
                    key=lambda ind: getattr(
                        ind,
                        replace_metric
                    ) if replace_metric != "fitness" else ind.fitness.values[0],
                    reverse=True
                )
            )

        # perform performance based replacement
        for i in range(self.n_islands):
            i_second_island = (i + 1) % self.n_islands
            # Replace worst with emigrants of next island
            self.islands_merged[
                self.islands_borders[i+1] - swap_size:self.islands_borders[i+1]
            ] = emigrants[i_second_island]
