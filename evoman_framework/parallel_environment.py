"""Module that provides a wrapper around the environment class in order
to provide a multithreading version that is easily usable"""
import os
import multiprocessing
import numpy as np
from environment_client import env_process


class ParallelEnvironment:
    """
    Class the allows to run an arbitrary number of processes running the environment
    in parallel.
    """
    def __init__(self, n_processes: int, config: dict, fitness_func: callable) -> None:
        """Init method

        Args:
            n_processes (int): amount of processes created
            config (dict): the configuration parameters used within all environments
            fitness_func (callable): fitness function that is used to calculate the fitness array
        """
        # disable any video from opening
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.config = config

        self.fitness_func = fitness_func
        self.processes = []
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.n_processes = n_processes

    def start_processes(self) -> None:
        """Method to start all processes within the environment"""
        for i in range(self.n_processes):
            p = multiprocessing.Process(
                target=env_process,
                args=(i, self.task_queue, self.result_queue, self.config, self.fitness_func)
            )
            p.start()
            self.processes.append(p)

        print(f"Started {self.n_processes} worker processes!")

    def queue_tasks(self, sub_populations: list[list]) -> None:
        """Add tasks to the queue.

        Args:
            sub_populations (list[list[list]]): a list of subpopulations
        """
        for idx, sub_population in enumerate(sub_populations):
            self.task_queue.put((idx, sub_population))

    def stop_processes(self) -> None:
        """Terminates all processes previously created"""
        for process in self.processes:
            process.terminate()

        print("Closed all subprocesses successfully.")

    def get_results(self, population: list) -> list:
        """Method to asynchronously (parallel) get results on a population using
        multiple processes. Do not start the Parallel environment in a script without
        running it in the if __main__ == ... branch, otherwise exceptions might be
        raised.

        Args:
            population (list): population being evaluated.

        Returns:
            list: returns a list with all evaluation scores obtained from
            calling env.play(). Results are in form of a dict with
            fitness, player_life, enemy_life, time keys.
        """
        # split and queue data for processing
        splits = np.array_split(population, self.n_processes)
        self.queue_tasks(splits)

        # wait for as many results as we queued (order will be messed up)
        results = [self.result_queue.get() for _ in range(len(splits))]
        results.sort(key=lambda x: x[0])

        # merge the subpopulations, removing the indices of them
        merged_results = []
        for _, partial_result in results:
            merged_results = merged_results + partial_result

        return merged_results
