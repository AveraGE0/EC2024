# --------------------------
# Imports
# --------------------------

import numpy as np
import random
import os
import csv
import pandas as pd
from deap import base, creator, tools
from evoman.environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt

# --------------------------
# Parameters
# --------------------------

EXPERIMENT_NAME = 'experiment_nsga2'
NUM_INPUTS = 20
NUM_OUTPUTS = 5
N_HIDDEN_NEURONS = 10

NUM_WEIGHTS = (
    NUM_INPUTS * N_HIDDEN_NEURONS +  # Input to hidden layer weights
    N_HIDDEN_NEURONS * NUM_OUTPUTS +  # Hidden to output layer weights
    N_HIDDEN_NEURONS +  # Hidden layer biases
    NUM_OUTPUTS  # Output layer biases
)

ENEMY_GROUPS = [[1, 2, 3, 4], [5, 6, 7, 8]]  # Two groups of enemies for training
POP_SIZE = 100
N_GEN = 20  # Number of generations per enemy group
CXPB = 0.7  # Crossover rate
MUTPB = 0.05  # Mutation rate
TOURNAMENT_SIZE = 3
ELITISM_RATE = 0.05  # 5% elitism

# Ensure experiment directory exists
if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)

# --------------------------
# DEAP Setup
# --------------------------

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_float,
    n=NUM_WEIGHTS
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --------------------------
# Environment Setup
# --------------------------

def initialize_environment(experiment_name, enemies, multiplemode, n_hidden_neurons):
    """Initialize the Evoman environment."""
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

# --------------------------
# Evaluation Functions
# --------------------------

def evaluate(individual, env):
    """Evaluate an individual against all enemies in the current group simultaneously."""
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    gain = player_life - enemy_life
    avg_time = time
    return enemy_life, player_life, np.log(avg_time + 1), gain

def evaluate_individual_on_single_enemies(individual, env):
    """Evaluate an individual against each enemy separately."""
    total_gain = 0
    defeated_enemies = 0
    results = {}
    for enemy_id in range(1, 9):
        env.update_parameter('enemies', [enemy_id])
        fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))
        gain = player_life - enemy_life
        total_gain += gain
        if enemy_life <= 0:
            defeated_enemies += 1
        results[f'enemy_{enemy_id}'] = {
            'fitness': fitness,
            'player_life': player_life,
            'enemy_life': enemy_life,
            'time': time,
            'gain': gain
        }
    results['total_gain'] = total_gain
    results['defeated_enemies'] = defeated_enemies
    return results

# --------------------------
# Genetic Operators
# --------------------------

# Register genetic operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=15)
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=-1.0,
    up=1.0,
    eta=15,
    indpb=1.0 / NUM_WEIGHTS
)

def select_with_deterministic_crowding(parents, offspring):
    """Perform deterministic crowding between parents and offspring."""
    new_population = []
    for parent, child in zip(parents, offspring):
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

def log_metrics(gen, population, logbook, csv_logger, diversity_logger, stats_logger, fitness_logger):
    """Log metrics for the current generation."""
    record = {}

    # Calculate fitness statistics
    fitnesses = [ind.fitness.values[1] for ind in population]  # Assuming the second value represents fitness
    min_fitness = np.min(fitnesses)
    max_fitness = np.max(fitnesses)
    avg_fitness = np.mean(fitnesses)

    # Calculate other statistics
    enemy_lives = [ind.fitness.values[0] for ind in population]
    player_lives = [ind.fitness.values[1] for ind in population]
    times = [np.exp(ind.fitness.values[2]) - 1 for ind in population]
    gains = [ind.fitness.values[3] for ind in population]

    # Diversity metrics
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distances.append(euclidean_distance(population[i], population[j]))
    avg_distance = np.mean(distances) if distances else 0

    # Store the key metrics for this generation
    record['gen'] = gen
    record['avg_fitness'] = avg_fitness
    record['min_fitness'] = min_fitness
    record['max_fitness'] = max_fitness
    record['avg_gain'] = np.mean(gains)
    record['avg_distance'] = avg_distance

    # Log to fitness logger (new)
    fitness_logger.writerow({
        'gen': gen,
        'avg_fitness': avg_fitness,
        'min_fitness': min_fitness,
        'max_fitness': max_fitness,
        'avg_gain': np.mean(gains),
        'avg_distance': avg_distance
    })

    # Log to other CSVs (this logs all the additional fields)
    csv_logger.writerow({
        'gen': gen,
        'avg_enemy_life': np.mean(enemy_lives),
        'std_enemy_life': np.std(enemy_lives),
        'min_enemy_life': np.min(enemy_lives),
        'max_enemy_life': np.max(enemy_lives),
        'avg_player_life': np.mean(player_lives),
        'std_player_life': np.std(player_lives),
        'min_player_life': np.min(player_lives),
        'max_player_life': np.max(player_lives),
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'avg_gain': np.mean(gains),
        'std_gain': np.std(gains),
        'min_gain': np.min(gains),
        'max_gain': np.max(gains),
        'avg_distance': avg_distance
    })

    # Log diversity data
    diversity_logger.writerow({'gen': gen, 'avg_distance': avg_distance})

    # Log generation statistics (you can duplicate the fields you want here, similar to csv_logger)
    stats_logger.writerow({
        'gen': gen,
        'avg_enemy_life': np.mean(enemy_lives),
        'std_enemy_life': np.std(enemy_lives),
        'min_enemy_life': np.min(enemy_lives),
        'max_enemy_life': np.max(enemy_lives),
        'avg_player_life': np.mean(player_lives),
        'std_player_life': np.std(player_lives),
        'min_player_life': np.min(player_lives),
        'max_player_life': np.max(player_lives),
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'avg_gain': np.mean(gains),
        'std_gain': np.std(gains),
        'min_gain': np.min(gains),
        'max_gain': np.max(gains),
        'avg_distance': avg_distance
    })

    # Print metrics to console
    print(f"Generation {gen}")
    print(f"Avg Fitness: {record['avg_fitness']:.2f}, Min Fitness: {record['min_fitness']:.2f}, Max Fitness: {record['max_fitness']:.2f}")
    print(f"Avg Enemy Life: {np.mean(enemy_lives):.2f}, Avg Player Life: {np.mean(player_lives):.2f}")
    print(f"Avg Gain: {np.mean(gains):.2f}, Avg Diversity: {avg_distance:.4f}")
    print("------------------------")
# --------------------------
# Analysis Functions
# --------------------------

def analyze_results(all_results):
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

    print(f"Average Gain across all solutions: {np.mean(avg_gains):.2f}")
    print(f"Average Defeated Enemies: {np.mean(defeated_enemies):.2f}")
    print(f"Best Solution Defeated: {max(defeated_enemies)} enemies")
    print(f"Highest Total Gain: {max(total_fitnesses):.2f}")

# --------------------------
# Evolutionary Algorithm
# --------------------------

def evolutionary_algorithm(toolbox, train_env, group_name):
    """Perform the evolutionary algorithm for one enemy group."""
    population = toolbox.population(n=POP_SIZE)
    logbook = tools.Logbook()

    # Prepare CSV logger for per-group metrics
    csv_file = open(os.path.join(EXPERIMENT_NAME, f'metrics_{group_name}.csv'), 'w', newline='')
    csv_logger = csv.DictWriter(csv_file, fieldnames=[
        'gen', 'avg_enemy_life', 'std_enemy_life', 'min_enemy_life', 'max_enemy_life',
        'avg_player_life', 'std_player_life', 'min_player_life', 'max_player_life',
        'avg_time', 'std_time', 'min_time', 'max_time',
        'avg_gain', 'std_gain', 'min_gain', 'max_gain', 'avg_distance'
    ])
    csv_logger.writeheader()

    # Diversity logger
    diversity_file = open(os.path.join(EXPERIMENT_NAME, f'diversity_{group_name}.csv'), 'w', newline='')
    diversity_logger = csv.DictWriter(diversity_file, fieldnames=['gen', 'avg_distance'])
    diversity_logger.writeheader()

    # Create a new CSV file to log generation statistics
    stats_file = open(os.path.join(EXPERIMENT_NAME, 'generation_statistics.csv'), 'w', newline='')
    stats_logger = csv.DictWriter(stats_file, fieldnames=[
        'gen', 'avg_enemy_life', 'std_enemy_life', 'min_enemy_life', 'max_enemy_life',
        'avg_player_life', 'std_player_life', 'min_player_life', 'max_player_life',
        'avg_time', 'std_time', 'min_time', 'max_time',
        'avg_gain', 'std_gain', 'min_gain', 'max_gain', 'avg_distance'
    ])
    stats_logger.writeheader()

    # Fitness logger (new)
    fitness_file = open(os.path.join(EXPERIMENT_NAME, f'fitness_{group_name}.csv'), 'w', newline='')
    fitness_logger = csv.DictWriter(fitness_file, fieldnames=[
        'gen', 'avg_fitness', 'min_fitness', 'max_fitness', 'avg_gain', 'avg_distance'
    ])
    fitness_logger.writeheader()

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind, env=train_env)

    # Log initial metrics
    log_metrics(0, population, logbook, csv_logger, diversity_logger, stats_logger, fitness_logger)

    # Begin the evolution
    for gen in range(1, N_GEN + 1):
        # Selection: Tournament selection
        parents = tools.selTournament(population, len(population), tournsize=TOURNAMENT_SIZE)

        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in parents]

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind, env=train_env)

        # Deterministic Crowding
        population = select_with_deterministic_crowding(parents, offspring)

        # Elitism: keep best individuals from parents and offspring
        elite_size = int(ELITISM_RATE * POP_SIZE)
        combined_population = parents + offspring
        elites = tools.selBest(combined_population, elite_size)

        # Add elites to current population
        population.extend(elites)

        # Ensure population size remains POP_SIZE
        population = tools.selBest(population, POP_SIZE)

        # Log metrics for this generation
        log_metrics(gen, population, logbook, csv_logger, diversity_logger, stats_logger, fitness_logger)

    # Close CSV files
    csv_file.close()
    diversity_file.close()
    stats_file.close()
    fitness_file.close()

    return population, logbook

# --------------------------
# Main Function
# --------------------------

def main():
    random.seed(42)

    eval_env = initialize_environment(
        EXPERIMENT_NAME,
        [1],  # This will be updated during evaluation
        "no",  # Single enemy evaluation
        N_HIDDEN_NEURONS
    )

    all_populations = []
    all_logbooks = []
    group_names = []

    for idx, enemies in enumerate(ENEMY_GROUPS):
        group_name = f"group_{idx+1}"
        group_names.append(group_name)
        print(f"Training on enemy group: {enemies}")

        # Initialize environment for this group
        train_env = initialize_environment(
            EXPERIMENT_NAME,
            enemies,
            "yes",  # Multiple mode activated during training
            N_HIDDEN_NEURONS
        )

        # Run the evolutionary algorithm for this group
        population, logbook = evolutionary_algorithm(toolbox, train_env, group_name)
        all_populations.append(population)
        all_logbooks.append(logbook)

    # Plot metrics for all groups
    from plotting import plot_metrics
    plot_metrics(all_logbooks, group_names)

    # Test best solutions against all enemies
    best_solutions = []
    for idx, population in enumerate(all_populations):
        group_name = group_names[idx]
        # Save the best individual from this group
        best_individual = tools.selBest(population, k=1)[0]
        np.savetxt(os.path.join(EXPERIMENT_NAME, f'best_solution_{group_name}.txt'), best_individual)
        best_solutions.append(best_individual)

    # Re-initialize evaluation environment
    eval_env = initialize_environment(
        EXPERIMENT_NAME,
        [1],  # Enemies will be updated in the function
        "no",  # Single enemy evaluation
        N_HIDDEN_NEURONS
    )

    all_enemy_results = []
    for idx, solution in enumerate(best_solutions):
        enemy_results = evaluate_individual_on_single_enemies(solution, eval_env)
        enemy_results['group'] = group_names[idx]
        all_enemy_results.append(enemy_results)

    # Save results to CSV
    results_list = []
    for result in all_enemy_results:
        row = {'group': result['group'], 'total_gain': result['total_gain'], 'defeated_enemies': result['defeated_enemies']}
        for enemy_id in range(1, 9):
            enemy_key = f'enemy_{enemy_id}'
            row[f'gain_enemy_{enemy_id}'] = result[enemy_key]['gain']
            row[f'player_life_enemy_{enemy_id}'] = result[enemy_key]['player_life']
            row[f'enemy_life_enemy_{enemy_id}'] = result[enemy_key]['enemy_life']
            row[f'time_enemy_{enemy_id}'] = result[enemy_key]['time']
            row[f'fitness_enemy_{enemy_id}'] = result[enemy_key]['fitness']
        results_list.append(row)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(EXPERIMENT_NAME, 'all_enemy_results.csv'), index=False)

    # Analyze and print results
    analyze_results(all_enemy_results)

    # Save best overall individual
    best_overall = max(
        best_solutions,
        key=lambda ind: evaluate_individual_on_single_enemies(ind, eval_env)['defeated_enemies']
    )
    np.savetxt(os.path.join(EXPERIMENT_NAME, 'best_solution_overall.txt'), best_overall)

    print("Evolution complete. Results saved in", EXPERIMENT_NAME)

if __name__ == "__main__":
    main()