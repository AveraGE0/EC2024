import numpy as np
import random
import os
import yaml
import logging
from logging.config import dictConfig
from deap import base, creator, tools, algorithms
from evoman.environment import Environment
from demo_controller import player_controller
import sys
import csv
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
        log_filename = os.path.join(experiment_directory, f'experiment_run_{run_id}_group_{enemy_group_str}.log')
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
selection_method = config['experiment']['selection_method']

parallel_enable = config['parallel']['enable']
parallel_processes = config['parallel']['processes']
multiple_runs = config['parallel']['multiple_runs']

# Function to parse 'parallel_processes' parameter
def parse_parallel_processes(processes_config, reserved_cores=0):
    if isinstance(processes_config, int):
        processes = processes_config
    elif isinstance(processes_config, str):
        if processes_config == 'max':
            max_procs = os.cpu_count()
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

    # Weights: (1.0 for maximizing total fitness, 0.0 for other objectives)
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,), force=True)
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
            speed="fastest",
            visuals=False
        )
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        sys.exit(1)

# --------------------------
# Evaluation Functions
# --------------------------

def evaluate(individual, env):
    """Evaluate an individual and store defeated enemies count."""
    # Override cons_multi to return individual values
    env.cons_multi = lambda x: x

    # Play the game with the current individual's parameters
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))

    # Ensure time is non-negative
    time = np.maximum(time, 0)

    # Count defeated enemies
    defeated_enemies = np.sum(np.array(enemy_life) <= 0)

    # Store defeated enemies count in the individual
    individual.defeated_enemies = defeated_enemies

    # Aggregate objectives
    total_fitness = np.sum(fitness)

    # Return the aggregated objectives
    return (total_fitness,)

# --------------------------
# Genetic Operators
# --------------------------

def register_genetic_operators(toolbox):
    """Register genetic operators in the toolbox."""
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=15)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=-1.0, up=1.0, eta=15, indpb=1.0 / num_weights)
    # Register selection methods from the config file
    selection_method_lower = selection_method.lower()
    if selection_method_lower == 'nsga2':
        toolbox.register("select", tools.selNSGA2)
    elif selection_method_lower == 'tournament':
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

# --------------------------
# Logging Functions
# --------------------------

def log_generation_metrics(population: List, generation: int, logger_instance, experiment_directory, run_id,
                           enemy_group, logbook) -> None:
    """
    Log generation metrics including scalar fitness values and diversity.
    """
    # Scalar fitness is total_fitness (first fitness value)
    scalar_fitness_values = [ind.fitness.values[0] for ind in population]

    # Fitness metrics
    avg_fitness = np.mean(scalar_fitness_values)
    max_fitness = np.max(scalar_fitness_values)
    std_fitness = np.std(scalar_fitness_values)

    # Defeated enemies metrics
    defeated_enemies_counts = [ind.defeated_enemies for ind in population]
    avg_defeated_enemies = np.mean(defeated_enemies_counts)
    max_defeated_enemies = np.max(defeated_enemies_counts)

    # Diversity metrics
    euclidean_diversity = calculate_population_diversity_euclidean(population)
    std_dev = calculate_population_std(population)

    # Log the results for this generation
    logger_instance.info(
        f"Generation {generation}: Avg Fitness = {avg_fitness:.4f}, Max Fitness = {max_fitness:.4f}, Std Dev = {std_fitness:.4f}"
    )
    logger_instance.info(
        f"Defeated Enemies - Avg: {avg_defeated_enemies:.2f}, Max: {max_defeated_enemies}"
    )
    logger_instance.info(
        f"Diversity - Euclidean: {euclidean_diversity:.4f}, Std Dev: {std_dev:.4f}"
    )

    # Update the logbook
    fitness_stats = {'avg': avg_fitness, 'max': max_fitness, 'std': std_fitness}
    defeated_enemies_stats = {'avg': avg_defeated_enemies, 'max': max_defeated_enemies}
    diversity_stats = {'euclidean': euclidean_diversity, 'std_dev': std_dev}

    logbook.record(gen=generation, fitness=fitness_stats, defeated_enemies=defeated_enemies_stats,
                   diversity=diversity_stats)

    # Optionally, store the metrics to a CSV file for later aggregation
    store_generation_metrics(run_id, enemy_group, generation, avg_fitness, max_fitness,
                             avg_defeated_enemies, max_defeated_enemies,
                             euclidean_diversity, std_dev, experiment_directory)

def store_generation_metrics(run_id, enemy_group, generation, avg_fitness, max_fitness,
                             avg_defeated_enemies, max_defeated_enemies,
                             euclidean_diversity, std_dev, experiment_directory):
    """
    Store generation metrics to a CSV file.
    """
    # Create a unique filename based on run ID and enemy group
    enemy_group_str = '_'.join(map(str, enemy_group))
    metrics_filename = os.path.join(experiment_directory, f'metrics_run_{run_id}_group_{enemy_group_str}.csv')

    # Check if the file exists; if not, write the header
    file_exists = os.path.isfile(metrics_filename)
    with open(metrics_filename, 'a', newline='') as csvfile:
        fieldnames = ['Generation', 'Avg_Scalar_Fitness', 'Max_Scalar_Fitness',
                      'Avg_Defeated_Enemies', 'Max_Defeated_Enemies',
                      'Euclidean_Diversity', 'Std_Dev']
        # Define the writer here
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Write the current generation's metrics
        writer.writerow({
            'Generation': generation,
            'Avg_Scalar_Fitness': f"{avg_fitness:.4f}",
            'Max_Scalar_Fitness': f"{max_fitness:.4f}",
            'Avg_Defeated_Enemies': f"{avg_defeated_enemies:.4f}",
            'Max_Defeated_Enemies': max_defeated_enemies,
            'Euclidean_Diversity': f"{euclidean_diversity:.4f}",
            'Std_Dev': f"{std_dev:.4f}"
        })

def calculate_population_diversity_euclidean(population: List) -> float:
    """
    Calculate the diversity of the population as the average Euclidean distance between individuals.
    If the population has only one individual, diversity is set to 0.
    """
    population_array = np.array(population)
    if len(population_array) > 1:
        distances = np.linalg.norm(population_array[:, None] - population_array, axis=2)
        distances = distances[np.triu_indices(len(population_array), k=1)]
        avg_distance = np.mean(distances)
    else:
        avg_distance = 0
    return avg_distance

def calculate_population_std(population: List) -> float:
    """
    Calculate the standard deviation of the population.
    If the population has only one individual, std is set to 0.
    """
    population_array = np.array(population)
    if len(population_array) > 1:
        std_dev = np.std(population_array, axis=0)
        avg_std_dev = np.mean(std_dev)
    else:
        avg_std_dev = 0
    return avg_std_dev

# --------------------------
# Evolutionary Algorithm
# --------------------------

def run_evolutionary_algorithm(run_id, toolbox, train_env, enemy_group, logger_instance, experiment_directory):
    """Perform the evolutionary algorithm for one enemy group."""
    group_name = f"run_{run_id}"
    logger_instance.info(f"Starting evolutionary run: {group_name} with Enemy Group: {enemy_group}")

    population = toolbox.population(n=pop_size)

    # Initialize the logbook
    logbook = tools.Logbook()
    logbook.header = ["gen", "fitness", "defeated_enemies", "diversity"]
    logbook.chapters["fitness"].header = ["avg", "max", "std"]
    logbook.chapters["defeated_enemies"].header = ["avg", "max"]
    logbook.chapters["diversity"].header = ["euclidean", "std_dev"]

    # Evaluate initial population
    fitnesses = toolbox.map(lambda ind: toolbox.evaluate(ind, train_env), population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Log initial metrics
    log_generation_metrics(population, 0, logger_instance, experiment_directory, run_id, enemy_group, logbook)  # Generation 0

    # Begin the evolution
    for gen in range(1, n_gen + 1):
        # Select and clone the next generation individuals
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

        # Replace population
        population[:] = offspring

        # Log metrics for this generation
        log_generation_metrics(population, gen, logger_instance, experiment_directory, run_id, enemy_group, logbook)

    # Save the best individual
    best_individual = tools.selBest(population, k=1)[0]
    enemy_group_str = '_'.join(map(str, enemy_group))
    best_filename = os.path.join(experiment_directory, f'best_solution_{group_name}_{enemy_group_str}.txt')
    np.savetxt(best_filename, best_individual)
    logger_instance.info(f"Best solution saved to {best_filename}")

    # Save the logbook
    logbook_filename = os.path.join(experiment_directory, f'logbook_run_{run_id}_group_{enemy_group_str}.csv')
    import pandas as pd
    records = []
    for record in logbook:
        entry = {}
        for key in record:
            if isinstance(record[key], dict):
                for subkey, value in record[key].items():
                    entry[f"{key}_{subkey}"] = value
            else:
                entry[key] = record[key]
        records.append(entry)
    logbook_df = pd.DataFrame(records)
    logbook_df.to_csv(logbook_filename, index=False)
    logger_instance.info(f"Logbook saved to {logbook_filename}")

    return population, best_individual, run_id

# --------------------------
# Main Function
# --------------------------

def main():
    # Load the config
    config = load_config()

    # Retrieve the experiment name from the config
    experiment_name = config['experiment']['name']

    # Create the experiment directory if it doesn't exist
    experiment_directory = create_experiment_directory(experiment_name)

    # Setup logging
    setup_logging(config['logging'])
    logger = logging.getLogger('evolutionary_algorithm')
    logger.info(f"Starting the experiment: {experiment_name}")

    # Batch process to run multiple evolutionary runs
    all_results = []
    for run_id in range(1, multiple_runs + 1):
        for enemy_group in enemy_groups:
            # Set up logger for this run and group
            logger_instance = setup_logging_for_run(run_id, enemy_group, experiment_directory)
            # Assign the provided seed for this run
            unique_seed = random_seed + run_id
            random.seed(unique_seed)
            np.random.seed(unique_seed)
            # Setup DEAP toolbox
            toolbox = setup_deap()
            register_genetic_operators(toolbox)
            # Setup parallel map if enabled
            if parallel_enable:
                from multiprocessing import Pool
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
            population, best_individual, run_id = run_evolutionary_algorithm(
                run_id, toolbox, train_env, enemy_group, logger_instance, experiment_directory
            )
            if parallel_enable:
                pool.close()
                pool.join()
                logger_instance.info("Multiprocessing pool closed.")
            all_results.append((population, best_individual, run_id))

    logger.info("Evolution complete. Results saved in the experiment directory.")

if __name__ == "__main__":
    main()