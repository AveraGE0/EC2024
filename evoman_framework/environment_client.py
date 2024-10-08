"""Module for specifying a process (function) able to run an environment and
simulate individuals in it."""
import os
import types
import multiprocessing as mp
from evoman.environment import Environment
from neural_controller import NeuralController


def env_process(
        worker_id: int,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        config: dict,
        fitness_func: callable
    ) -> None:
    """Function for process running environment for simulation.
    Environment is created, the function waits for new tasks in the
    queue and works on them as soon as they are scheduled.

    Args:
        worker_id (int): id of the process running this function.
        task_queue (mp.Queue): Queue for open tasks (individual evaluation).
        result_queue (mp.Queue): Queue where results will be stored.
        config (dict): Configuration as dict.
        fitness_func (callable): The fitness function used in the environment.
    """
    print(f"Worker {worker_id} starting, initializing environment...")

    experiment_name = os.path.join("../experiments/", config["name"])
    # create experiment dir if it does not exist
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize the environment
    nc = NeuralController(
        n_inputs=config["n_inputs"],
        n_outputs=config["n_outputs"],
        hidden_size=config["hidden_size"]
    )

    env = Environment(
        experiment_name=experiment_name,  # this is actually a path!
        multiplemode=config["multiplemode"],
        enemies=config["train_enemy"],
        player_controller=nc,
        visuals=False,
        level=config["level"],
    )
    # this function transforms the output vectors to z-scores we want the actual scores, we
    # change it to return the original values
    env.cons_multi = lambda x: x
    # adjust the fitness calculated per enemy
    env.fitness_single = types.MethodType(fitness_func, env)
    #env.fitness_single = fitness_func

    while True:
        idx, sub_population = task_queue.get()
        # Send the result back to the parent process via result queue as a list of dicts
        # so the results are easier to understand!
        results = []
        for genome in sub_population:
            result = env.play(pcont=genome)
            results.append({
                "fitness": result[0],
                "player_life": result[1],
                "enemy_life": result[2],
                "time": result[3]
            })

        result_queue.put((idx, results))
