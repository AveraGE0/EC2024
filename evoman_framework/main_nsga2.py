
import numpy as np
import random
import os
import yaml
import logging
from logging.config import dictConfig
from deap import base, creator, tools, algorithms
from evoman.environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import sys
from typing import List, Tuple, Dict

# --------------------------
# Load Configuration
# --------------------------

def load_config(config_file='config_nsga2.yaml'):
    """Load the configuration from a YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Ensure the experiment directory exists
def create_experiment_directory(experiment_name):
    """Create a directory for the experiment if it doesn't already exist."""
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    return experiment_name



# --------------------------
# Setup Logging
# --------------------------

def setup_logging(logging_config):
    dictConfig(logging_config)

setup_logging(config['logging'])

logger = logging.getLogger('evolutionary_algorithm')

def setup_logging_for_run(run_id, enemy_group, experiment_directory):
    """Set up a logger for a specific run and enemy group, saving logs in the experiment directory."""
    enemy_group_str = '_'.join(map(str, enemy_group))
    logger_name = f'evolutionary_algorithm_run_{run_id}_group_{enemy_group_str}'
    logger_instance = logging.getLogger(logger_name)
    logger_instance.setLevel(logging.INFO)
    # Add handlers if not already added
    if not logger_instance.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger_instance.addHandler(ch)
        # File handler
        log_filename = os.path.join(experiment_directory, f'experiment_run_{run_id}_group_{enemy_group}.log')
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger_instance.addHandler(fh)
    return logger_instance

# --------------------------
# Parameters from Config
# --------------------------

experiment_name = config['experiment']['name']
num_inputs = config['experiment']['num_inputs']
num_outputs = config['experiment']['num_outputs']
n_hidden_neurons = config['experiment']['n_hidden_neurons']

num_weights = (
    num_inputs * n_hidden_neurons +  # Input to hidden layer weights
    n_hidden_neurons * num_outputs +  # Hidden to output layer weights
    n_hidden_neurons +  # Hidden layer biases
    num_outputs  # Output layer biases
)

enemy_groups = config['experiment']['enemy_groups']
pop_size = config['experiment']['pop_size']
n_gen = config['experiment']['n_gen']
cxpb = config['experiment']['cxpb']
mutpb = config['experiment']['mutpb']
tournament_size = config['experiment']['tournament_size']
elitism_rate = config['experiment']['elitism_rate']
random_seed = config['experiment']['random_seed']
selection_method = config['experiment']['selection_method']  # Use 'selection_method' instead

parallel_enable = config['parallel']['enable']
parallel_processes = config['parallel']['processes']
multiple_runs = config['parallel']['multiple_runs']

# uncomment if you want to obtain the same results for each run
# random.seed(random_seed)
# np.random.seed(random_seed)

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Function to parse 'parallel_processes' parameter
def parse_parallel_processes(processes_config, reserved_cores=0):
    if isinstance(processes_config, int):
        processes = processes_config
    elif isinstance(processes_config, str):
        if processes_config == 'max':
            max_procs = cpu_count()
            processes = max_procs - reserved_cores
        else:
            raise ValueError(f"Invalid processes config: {processes_config}")
    else:
        raise ValueError(f"Invalid processes config: {processes_config}")
    if processes < 1:
        raise ValueError(f"Number of processes ({processes}) must be at least 1.")
    return processes

# Retrieve 'reserved_cores' from the configuration
reserved_cores = config['parallel'].get('reserved_cores', 0)

# Parse the 'processes' parameter using the updated function
parallel_processes = parse_parallel_processes(config['parallel']['processes'], reserved_cores)

# --------------------------
# DEAP Setup
# --------------------------

def setup_deap():
    """Setup DEAP toolbox and creators."""
    # Clear any existing creators (to prevent errors when re-running)
    if 'FitnessMulti' in creator.__dict__:
        del creator.FitnessMulti
    if 'Individual' in creator.__dict__:
        del creator.Individual

    # Weights: (1.0 for maximizing 100 - enemy_life, 1.0 for maximizing player_life, -1.0 for minimizing log(time))
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0), force=True)
    creator.create("Individual", list, fitness=creator.FitnessMulti, force=True)

    # Register attributes, individual, and population
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox

# --------------------------
# Environment Setup
# --------------------------

def initialize_environment(experiment_name, enemies, multiplemode, n_hidden_neurons):
    """Initialize the Evoman environment."""
    try:
        return Environment(
            experiment_name=experiment_name,
            enemies=enemies,
            multiplemode=multiplemode,
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest"
        )
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        sys.exit(1)

# --------------------------
# Evaluation Functions
# --------------------------

def evaluate(individual, env):
    """Evaluate an individual against all enemies in the current group simultaneously."""
    # Play the game with the current individual's parameters
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    # Return the three objectives:
    # 1. Maximize reduction in enemy_life (100 - enemy_life)
    # 2. Maximize player_life
    # 3. Minimize log(time)
    return 100 - enemy_life, player_life, np.log(time + 1),

def evaluate_individual_on_single_enemies(individual, env):
    """Evaluate an individual against each enemy separately."""
    results = {}

    # Loop over each enemy (1 to 8)
    for enemy_id in range(1, 9):
        # Update the environment to evaluate the individual against the specific enemy
        env.update_parameter('enemies', [enemy_id])

        # Get fitness, player life, enemy life, and time for the current enemy
        fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))

        fitness_of_individual = fitness_value(enemy_life, player_life, time)

        # Store the results for each enemy (multi-objective metrics)
        results[f'enemy_{enemy_id}'] = {
            'player_life': player_life,
            'enemy_life': enemy_life,
            'time': time,
            'objectives': (100 - enemy_life, player_life, np.log(time + 1))  # The three objectives
        }
    return results

def fitness_value(enemy_life: float, player_life: float, time: float) -> float:
    """Calculate a fitness score for comparison with other models."""
    return 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time + 1)

def calculate_population_diversity_euclidian(population: List) -> float:
    """
    Calculate the diversity of the population as the average Euclidean distance between individuals.
    If the population has only one individual, diversity is set to 0.
    """
    population_array = np.array([ind for ind in population])

    if len(population_array) > 1:
        # Calculate pairwise Euclidean distances
        distances = np.linalg.norm(population_array[:, None] - population_array, axis=2)
        # Exclude self-distances (distance of an individual to itself)
        distances = distances[distances > 0]
        avg_distance = np.mean(distances)
    else:
        avg_distance = 0  # No diversity if there's only one individual

    return avg_distance

def calculate_population_std(population: List) -> float:
    """
    Calculate the standard deviation of the population.
    If the population has only one individual, std is set to 0.
    """
    # Convert population to a numpy array for easier manipulation
    population_array = np.array([ind for ind in population])

    if len(population_array) > 1:
        # Calculate standard deviation across the population
        std_dev = np.std(population_array, axis=0)  # std of each parameter
        avg_std_dev = np.mean(std_dev)  # Average std across all parameters
    else:
        avg_std_dev = 0  # No std if there's only one individual

    return avg_std_dev

# --------------------------
# Genetic Operators
# --------------------------

def register_genetic_operators(toolbox):
    """Register genetic operators in the toolbox."""
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=15)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=-1.0, up=1.0, eta=15, indpb=1.0 / num_weights)
    # Register selection methods from the config file, you can choose there which option is applied
    selection_method_lower = selection_method.lower()  # Convert to lowercase for consistency
    if selection_method_lower == 'nsga2':
        toolbox.register("select", tools.selNSGA2)
    elif selection_method_lower == 'tournament':
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    elif selection_method_lower == 'deterministiccrowding':
        toolbox.register("select", select_with_deterministic_crowding)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

# --------------------------
# Deterministic Crowding Functions
# --------------------------
# Defined here since it isn't included standard in DEAP framework

def select_with_deterministic_crowding(parents, offspring):
    """Perform deterministic crowding between parents and offspring."""
    new_population = []
    for parent, child in zip(parents, offspring):
        # Calculate Euclidean distance
        if euclidean_distance(parent, child) < 0.01:
            # If parent and child are similar, choose the one with better fitness
            if dominates(child.fitness.values, parent.fitness.values):
                new_population.append(child)
            else:
                new_population.append(parent)
        else:
            # If dissimilar, child replaces parent
            new_population.append(child)
    return new_population

def dominates(fitness1, fitness2):
    """Check if fitness1 dominates fitness2."""
    return tools.emo.isDominated(fitness2, fitness1)

def euclidean_distance(ind1, ind2):
    """Calculate Euclidean distance between two individuals."""
    return np.linalg.norm(np.array(ind1) - np.array(ind2))

# --------------------------
# Logging Functions
# --------------------------

def log_generation_metrics(population: List, generation: int, logger_instance, experiment_directory, run_id,
                           enemy_group) -> None:
    """
    Log generation metrics including scalar fitness values and diversity.

    Parameters:
    - population: List of individuals in the current population.
    - generation: Current generation number.
    - logger_instance: Logger object for logging information.
    - experiment_directory: Directory where experiment data is stored.
    - run_id: Identifier for the current run.
    - enemy_group: List representing the current enemy group.
    """

    # Define the scalar fitness formula
    def calculate_scalar_fitness(individual_fitness):
        """
        Calculate scalar fitness based on the formula:
        fitness = (100 - enemy_life) + player_life - log(t)

        Parameters:
        - individual_fitness: Tuple containing (100 - enemy_life, player_life, log(t + 1))

        Returns:
        - Scalar fitness value as a float.
        """
        enemy_life_component = individual_fitness[0]  # 100 - enemy_life
        player_life = individual_fitness[1]  # player_life
        log_time = individual_fitness[2]  # log(t + 1)
        return enemy_life_component + player_life - log_time

    # Calculate scalar fitness for each individual
    scalar_fitness_values = [calculate_scalar_fitness(ind.fitness.values) for ind in population]

    # Calculate average and maximum scalar fitness
    avg_fitness = np.mean(scalar_fitness_values)
    max_fitness = np.max(scalar_fitness_values)

    # Log the results for this generation
    logger_instance.info(
        f"Generation {generation}: Avg Scalar Fitness = {avg_fitness:.4f}, Max Scalar Fitness = {max_fitness:.4f}"
    )

    # Optionally, store the metrics to a CSV file for later aggregation
    store_generation_metrics(run_id, enemy_group, generation, avg_fitness, max_fitness, experiment_directory)


def store_generation_metrics(run_id, enemy_group, generation, avg_fitness, max_fitness, experiment_directory):
    """
    Store generation metrics to a CSV file.

    Parameters:
    - run_id: Identifier for the current run.
    - enemy_group: List representing the current enemy group.
    - generation: Current generation number.
    - avg_fitness: Average scalar fitness of the population.
    - max_fitness: Maximum scalar fitness in the population.
    - experiment_directory: Directory where experiment data is stored.
    """
    import os
    import csv

    # Create a unique filename based on run ID and enemy group
    enemy_group_str = '_'.join(map(str, enemy_group))
    metrics_filename = os.path.join(experiment_directory, f'metrics_run_{run_id}_group_{enemy_group_str}.csv')

    # Check if the file exists; if not, write the header
    file_exists = os.path.isfile(metrics_filename)
    with open(metrics_filename, 'a', newline='') as csvfile:
        fieldnames = ['Generation', 'Avg_Scalar_Fitness', 'Max_Scalar_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        # Write the current generation's metrics
        writer.writerow({
            'Generation': generation,
            'Avg_Scalar_Fitness': f"{avg_fitness:.4f}",
            'Max_Scalar_Fitness': f"{max_fitness:.4f}"
        })


# --------------------------
# Analysis Functions
# --------------------------

def analyze_results(all_results, logger_instance):
    """Analyze the results of the evaluation."""
    avg_gains = []
    defeated_enemies = []
    total_fitnesses = []

    for result in all_results:
        gains = [r['gain'] for r in result.values() if isinstance(r, dict)]
        avg_gain = sum(gains) / len(gains)
        avg_gains.append(avg_gain)
        defeated = result['defeated_enemies']
        defeated_enemies.append(defeated)
        total_fitnesses.append(result['total_gain'])  # Assuming total gain is used as total fitness

    logger_instance.info(f"Average Gain across all solutions: {np.mean(avg_gains):.2f}")
    logger_instance.info(f"Average Defeated Enemies: {np.mean(defeated_enemies):.2f}")
    logger_instance.info(f"Best Solution Defeated: {max(defeated_enemies)} enemies")
    logger_instance.info(f"Highest Total Gain: {max(total_fitnesses):.2f}")

# --------------------------
# Evolutionary Algorithm
# --------------------------

def run_evolutionary_algorithm(run_id, toolbox, train_env, enemy_group, logger_instance, experiment_directory):
    """Perform the evolutionary algorithm for one enemy group."""
    group_name = f"run_{run_id}"
    logger_instance.info(f"Starting evolutionary run: {group_name} with Enemy Group: {enemy_group}")

    population = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = toolbox.map(lambda ind: toolbox.evaluate(ind, train_env), population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Log initial metrics, including average and max fitness
    log_generation_metrics(population, 0, logger_instance, experiment_directory, run_id, enemy_group)  # Generation 0

    # Begin the evolution
    for gen in range(1, n_gen + 1):
        selection_method_lower = selection_method.lower()

        if selection_method_lower == 'deterministiccrowding':
            # Randomly shuffle population
            random.shuffle(population)
            # Apply variation to population to produce offspring
            offspring = []
            for parent1, parent2 in zip(population[::2], population[1::2]):
                # Clone the individuals
                child1, child2 = toolbox.clone(parent1), toolbox.clone(parent2)
                # Apply crossover
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                # Apply mutation
                toolbox.mutate(child1)
                toolbox.mutate(child2)
                # Remove fitness values
                del child1.fitness.values
                del child2.fitness.values
                offspring.extend([child1, child2])

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(lambda ind: toolbox.evaluate(ind, train_env), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Perform deterministic crowding replacement
            new_population = []
            for parent1, parent2, child1, child2 in zip(population[::2], population[1::2], offspring[::2], offspring[1::2]):
                # Calculate distances
                d_p1_c1 = euclidean_distance(parent1, child1)
                d_p1_c2 = euclidean_distance(parent1, child2)
                d_p2_c1 = euclidean_distance(parent2, child1)
                d_p2_c2 = euclidean_distance(parent2, child2)

                # Assign offspring to parents based on distance
                if (d_p1_c1 + d_p2_c2) <= (d_p1_c2 + d_p2_c1):
                    pairs = [(parent1, child1), (parent2, child2)]
                else:
                    pairs = [(parent1, child2), (parent2, child1)]

                for parent, child in pairs:
                    # Select between parent and child
                    if dominates(child.fitness.values, parent.fitness.values):
                        new_population.append(child)
                    else:
                        new_population.append(parent)

            population = new_population

        elif selection_method_lower == 'nsga2':
            # Apply variation (crossover and mutation)
            offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(lambda ind: toolbox.evaluate(ind, train_env), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            population = toolbox.select(population + offspring, pop_size)

        elif selection_method_lower == 'tournament':
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(lambda ind: toolbox.evaluate(ind, train_env), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            population[:] = offspring

        else:
            logger_instance.error(f"Unknown selection method: {selection_method}")
            sys.exit(1)

        # Log metrics for this generation, including average and max fitness
        log_generation_metrics(population, gen, logger_instance, experiment_directory, run_id, enemy_group)  # Generation N

    # Save the best individual
    best_individual = tools.selBest(population, k=1)[0]
    best_filename = os.path.join(experiment_directory, f'best_solution_{group_name}.txt')
    np.savetxt(best_filename, best_individual)
    logger_instance.info(f"Best solution saved to {best_filename}")

    return population, best_individual, run_id


import os
import csv
import numpy as np


def aggregate_metrics(experiment_directory, enemy_groups, multiple_runs, num_generations):
    """
    Aggregate mean and max scalar fitness metrics across multiple runs for each enemy group.

    Parameters:
    - experiment_directory: Directory where experiment data is stored.
    - enemy_groups: List of enemy groups.
    - multiple_runs: Number of independent evolutionary runs.
    - num_generations: Total number of generations per run.

    Returns:
    - A dictionary structured as:
      {
          enemy_group_tuple: {
              'mean_fitness_mean': np.array([...]),
              'mean_fitness_std': np.array([...]),
              'max_fitness_mean': np.array([...]),
              'max_fitness_std': np.array([...])
          },
          ...
      }
    """
    aggregated_data = {}

    for group in enemy_groups:
        group_tuple = tuple(group)
        aggregated_data[group_tuple] = {
            'mean_fitness_runs': [],
            'max_fitness_runs': []
        }

        for run_id in range(1, multiple_runs + 1):
            group_str = '_'.join(map(str, group))
            metrics_filename = os.path.join(experiment_directory, f'metrics_run_{run_id}_group_{group_str}.csv')

            if not os.path.isfile(metrics_filename):
                print(f"Metrics file {metrics_filename} not found. Skipping run {run_id} for group {group_str}.")
                continue

            # Read the CSV file
            with open(metrics_filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                run_mean_fitness = []
                run_max_fitness = []
                for row in reader:
                    generation = int(row['Generation'])
                    if generation > num_generations:
                        continue
                    run_mean_fitness.append(float(row['Avg_Scalar_Fitness']))
                    run_max_fitness.append(float(row['Max_Scalar_Fitness']))

            # Ensure the run has data for all generations
            if len(run_mean_fitness) != num_generations + 1:
                print(
                    f"Incomplete data in {metrics_filename}. Expected {num_generations + 1} generations, got {len(run_mean_fitness)}.")
                continue

            aggregated_data[group_tuple]['mean_fitness_runs'].append(run_mean_fitness)
            aggregated_data[group_tuple]['max_fitness_runs'].append(run_max_fitness)

        # Convert lists to NumPy arrays for statistical computations
        mean_fitness_runs = np.array(aggregated_data[group_tuple]['mean_fitness_runs'])
        max_fitness_runs = np.array(aggregated_data[group_tuple]['max_fitness_runs'])

        # Compute mean and std deviation across runs for each generation
        aggregated_data[group_tuple]['mean_fitness_mean'] = np.mean(mean_fitness_runs, axis=0)
        aggregated_data[group_tuple]['mean_fitness_std'] = np.std(mean_fitness_runs, axis=0)
        aggregated_data[group_tuple]['max_fitness_mean'] = np.mean(max_fitness_runs, axis=0)
        aggregated_data[group_tuple]['max_fitness_std'] = np.std(max_fitness_runs, axis=0)

    return aggregated_data

# --------------------------
# Main Function for Single Run
# --------------------------

def single_run(run_id, enemy_group, config, experiment_directory, seed):
    """Function to execute a single evolutionary run."""
    # Set up logger for this run and group
    logger_instance = setup_logging_for_run(run_id, enemy_group, experiment_directory)

    # Assign the provided seed for this run
    unique_seed = seed
    random.seed(unique_seed)
    np.random.seed(unique_seed)

    # Setup DEAP toolbox
    toolbox = setup_deap()
    register_genetic_operators(toolbox)

    # Disable parallelism inside single_run if batch processing is using multiprocessing
    if multiple_runs > 1 and parallel_enable:
        inner_parallel_enable = False
    else:
        inner_parallel_enable = parallel_enable

    # Setup parallel map if enabled
    if inner_parallel_enable:
        pool = Pool(processes=parallel_processes)
        toolbox.register("map", pool.map)
        logger_instance.info(f"Multiprocessing pool with {parallel_processes} processes initialized.")
    else:
        toolbox.register("map", map)

    # Initialize environment for this run
    train_env = initialize_environment(
        experiment_name,
        enemy_group,  # Pass enemy group
        "yes" if len(enemy_group) > 1 else "no",  # Set multiple mode based on group size
        n_hidden_neurons
    )

    # Execute evolutionary algorithm
    population, best_individual, run_id = run_evolutionary_algorithm(run_id, toolbox, train_env, enemy_group, logger_instance, experiment_directory)

    if inner_parallel_enable:
        pool.close()
        pool.join()
        logger_instance.info("Multiprocessing pool closed.")

    return population, best_individual, run_id, enemy_group

# --------------------------
# Batch Processing for Multiple Runs
# --------------------------

def batch_processing(config, experiment_directory):
    """Run multiple independent evolutionary runs in parallel."""
    logger.info(f"Starting batch processing with {multiple_runs} runs.")
    run_ids = list(range(1, multiple_runs + 1))

    # Generate seeds dynamically: base_seed + run_id
    base_seed = config['experiment']['random_seed']
    seeds = [base_seed + run_id for run_id in run_ids]

    results = []

    # Iterate over each run and enemy group
    tasks = []
    for run_id, seed in zip(run_ids, seeds):
        for enemy_group in enemy_groups:
            # Append tasks with run_id and seed
            tasks.append((run_id, enemy_group, config, experiment_directory, seed))

    if parallel_enable and multiple_runs > 1:
        pool = Pool(processes=parallel_processes)

        # Modify single_run to accept seed
        results = pool.starmap(single_run, tasks)

        pool.close()
        pool.join()
    else:
        # Non-parallel execution
        for task in tasks:
            result = single_run(*task)  # Unpack the run_id, enemy_group, config, and seed
            results.append(result)

    logger.info("Batch processing complete.")
    return results

# --------------------------
# Test best individual
# --------------------------

def test_best_individual_against_all_enemies(best_individual, eval_env, logger_instance):
    """Test the best individual against all enemies and log results per enemy, including total gain."""
    results = evaluate_individual_on_single_enemies(best_individual, eval_env)

    # Log the results for each enemy
    logger_instance.info("Testing best individual against all enemies:")
    logger_instance.info(f"{'Enemy':<10} {'Player Life':<15} {'Enemy Life':<15} {'Gain':<10}")

    # Calculate total gain directly in the loop
    total_gain = 0
    for enemy_id, result in results.items():
        player_life = result['player_life']
        enemy_life = result['enemy_life']
        gain = player_life - enemy_life
        total_gain += gain

        # Log the results in a table format
        logger_instance.info(f"{enemy_id:<10} {player_life:<15} {enemy_life:<15} {gain:<10.4f}")

    # Log the total gain
    logger_instance.info(f"\nTotal Gain: {total_gain:.4f}")

    return results, total_gain

# --------------------------
# Helper Functions
# --------------------------

def evaluate_best_individuals(best_individuals, logger_instance):
    """Evaluate best individuals against all enemies and collect detailed statistics."""
    eval_env = initialize_test_environment()

    performance_table = []

    for item in best_individuals:
        population, best_individual, run_id, enemy_group = item
        # Evaluate the best individual
        results, total_gain = test_best_individual_against_all_enemies(best_individual, eval_env, logger_instance)

        # Calculate additional statistics
        total_defeated_enemies = sum(1 for result in results.values() if result['enemy_life'] == 0)
        total_player_life = sum(result['player_life'] for result in results.values())
        total_enemy_life = sum(result['enemy_life'] for result in results.values())

        # Collect detailed results for flexibility
        performance_table.append({
            'run_id': run_id,
            'enemy_group': enemy_group,
            'total_gain': total_gain,
            'total_defeated_enemies': total_defeated_enemies,
            'total_player_life': total_player_life,
            'total_enemy_life': total_enemy_life,
            'results': results,  # Detailed results per enemy
            'best_individual': best_individual
        })

    # Sort the performance_table based on total_gain
    performance_table_sorted = sorted(performance_table, key=lambda x: x['total_gain'], reverse=True)

    # Display summary statistics in a table
    print("\nPerformance of Best Individuals from Each Run:")
    print(f"{'Rank':<5} {'Run ID':<7} {'Enemy Group':<15} {'Total Gain':<10} {'Defeated Enemies':<18} {'Player Life':<12} {'Enemy Life':<12}")
    for idx, entry in enumerate(performance_table_sorted):
        # Convert enemy_group list to a formatted string
        enemy_group_str = ', '.join(map(str, entry['enemy_group']))
        print(f"{idx+1:<5} {entry['run_id']:<7} {enemy_group_str:<15} {entry['total_gain']:<10.4f} {entry['total_defeated_enemies']:<18} {entry['total_player_life']:<12.4f} {entry['total_enemy_life']:<12.4f}")

    # Optionally return the sorted performance table for further manipulation
    return performance_table_sorted

def initialize_test_environment():
    """Initialize the environment for testing the best individual against all enemies."""
    return initialize_environment(
        experiment_name,
        list(range(1, 9)),  # Enemies 1 to 8
        "no",  # Single mode
        n_hidden_neurons
    )

# --------------------------
# Main Function
# --------------------------

def main():
    # Load the config
    config = load_config()  # This only returns the config

    # Retrieve the experiment name from the config
    experiment_name = config['experiment']['name']

    # Create the experiment directory if it doesn't exist
    experiment_directory = create_experiment_directory(experiment_name)

    # Setup logging
    setup_logging(config['logging'])
    logger = logging.getLogger('evolutionary_algorithm')
    logger.info(f"Starting the experiment: {experiment_name}")

    # Batch process to run multiple evolutionary runs
    all_results = batch_processing(config, experiment_directory)

    # Evaluate the best individuals
    performance_table = evaluate_best_individuals(all_results, logger)

    # Print relevant details to the console
    print(f"\nEvolution Summary:")
    print(f"------------------")
    print(f"Generations per Run: {config['experiment']['n_gen']}")
    print(f"Enemy Groups: {len(config['experiment']['enemy_groups'])} groups - {config['experiment']['enemy_groups']}")
    print(f"Total Runs: {config['parallel']['multiple_runs']}")
    print(
        f"Best Overall Individual from Run {performance_table[0]['run_id']} with Total Gain: {performance_table[0]['total_gain']:.4f}")

    logger.info("Evolution complete. Results saved in the experiment directory.")


if __name__ == "__main__":
    main()
