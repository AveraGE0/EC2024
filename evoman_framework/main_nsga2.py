import numpy as np
import random
import os
import csv
import pandas as pd
from deap import base, creator, tools
from evoman.environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt

# Constants
EXPERIMENT_NAME = 'experiment_nsga2'
NUM_INPUTS = 20
NUM_OUTPUTS = 5
N_HIDDEN_NEURONS = 10
NUM_WEIGHTS = (NUM_INPUTS * N_HIDDEN_NEURONS) + (N_HIDDEN_NEURONS * NUM_OUTPUTS) + N_HIDDEN_NEURONS + NUM_OUTPUTS
ENEMY_GROUPS = [[2, 4, 7], [4, 7, 8]]  # Two groups of enemies for training
POP_SIZE = 50
N_GEN = 20
CXPB = 0.9
MUTPB = 0.1

# Ensure experiment directory exists
if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)

# DEAP setup
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_WEIGHTS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Environment Setup Module
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


# Evaluation Module
def evaluate(individual, env):
    """Evaluate an individual against all enemies in the current group simultaneously."""
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    gain = player_life - enemy_life
    avg_time = time
    return enemy_life, player_life, np.log(avg_time + 1), gain


def evaluate_individual_on_single_enemies(individual, env):
    """Evaluate an individual against each enemy separately."""
    total_gain = 0
    results = {}
    for enemy_id in range(1, 9):
        env.update_parameter('enemies', [enemy_id])
        fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))
        gain = player_life - enemy_life
        total_gain += gain
        results[f'enemy_{enemy_id}'] = {
            'player_life': player_life,
            'enemy_life': enemy_life,
            'time': time,
            'gain': gain
        }
    results['total_gain'] = total_gain
    return results


# Logging Module
def log_metrics(gen, population, logbook, csv_logger):
    """Log metrics for the current generation."""
    record = {}

    # Calculate various statistics
    fitnesses = [ind.fitness.values for ind in population]
    enemy_lives = [f[0] for f in fitnesses]
    player_lives = [f[1] for f in fitnesses]
    times = [np.exp(f[2]) - 1 for f in fitnesses]
    gains = [f[3] for f in fitnesses]

    record['gen'] = gen
    record['avg_enemy_life'] = np.mean(enemy_lives)
    record['std_enemy_life'] = np.std(enemy_lives)
    record['min_enemy_life'] = np.min(enemy_lives)
    record['max_enemy_life'] = np.max(enemy_lives)
    record['avg_player_life'] = np.mean(player_lives)
    record['std_player_life'] = np.std(player_lives)
    record['min_player_life'] = np.min(player_lives)
    record['max_player_life'] = np.max(player_lives)
    record['avg_time'] = np.mean(times)
    record['std_time'] = np.std(times)
    record['min_time'] = np.min(times)
    record['max_time'] = np.max(times)
    record['avg_gain'] = np.mean(gains)
    record['std_gain'] = np.std(gains)
    record['min_gain'] = np.min(gains)
    record['max_gain'] = np.max(gains)

    logbook.record(**record)
    csv_logger.writerow(record)

    print(f"Generation {gen}")
    print(f"Avg Enemy Life: {record['avg_enemy_life']:.2f}")
    print(f"Avg Player Life: {record['avg_player_life']:.2f}")
    print(f"Avg Gain: {record['avg_gain']:.2f}")
    print(f"Avg Time: {record['avg_time']:.2f}")
    print("------------------------")


# Plotting Module
def plot_metrics(logbook):
    """Plot metrics over generations."""
    gen = logbook.select('gen')
    avg_enemy_life = logbook.select('avg_enemy_life')
    avg_player_life = logbook.select('avg_player_life')
    avg_gain = logbook.select('avg_gain')
    avg_time = logbook.select('avg_time')

    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    axs[0].plot(gen, avg_enemy_life, label='Avg Enemy Life')
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Average Enemy Life')
    axs[0].legend()

    axs[1].plot(gen, avg_player_life, label='Avg Player Life')
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Average Player Life')
    axs[1].legend()

    axs[2].plot(gen, avg_gain, label='Avg Gain')
    axs[2].set_xlabel('Generation')
    axs[2].set_ylabel('Average Gain')
    axs[2].legend()

    axs[3].plot(gen, avg_time, label='Avg Time')
    axs[3].set_xlabel('Generation')
    axs[3].set_ylabel('Average Time')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_NAME, 'metrics_plot.png'))


# Analysis Module
def analyze_results(all_results):
    avg_gains = []
    defeated_enemies = []

    for result in all_results:
        gains = [r['gain'] for r in result.values() if isinstance(r, dict)]
        avg_gains.append(sum(gains) / len(gains))
        defeated = sum(1 for r in result.values() if isinstance(r, dict) and r['enemy_life'] <= 0)
        defeated_enemies.append(defeated)

    print(f"Average Gain across all solutions: {sum(avg_gains) / len(avg_gains):.2f}")
    print(f"Average Defeated Enemies: {sum(defeated_enemies) / len(defeated_enemies):.2f}")
    print(f"Best Solution Defeated: {max(defeated_enemies)} enemies")


# Evolutionary Algorithm Module
def evolutionary_algorithm(toolbox, train_env):
    """Perform the evolutionary algorithm."""
    population = toolbox.population(n=POP_SIZE)
    logbook = tools.Logbook()
    pareto = tools.ParetoFront()

    # Prepare CSV logger
    csv_file = open(os.path.join(EXPERIMENT_NAME, 'metrics.csv'), 'w', newline='')
    csv_logger = csv.DictWriter(csv_file, fieldnames=['gen', 'avg_enemy_life', 'std_enemy_life', 'min_enemy_life',
                                                      'max_enemy_life',
                                                      'avg_player_life', 'std_player_life', 'min_player_life',
                                                      'max_player_life',
                                                      'avg_time', 'std_time', 'min_time', 'max_time',
                                                      'avg_gain', 'std_gain', 'min_gain', 'max_gain'])
    csv_logger.writeheader()

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind, env=train_env)

    # Log initial metrics
    log_metrics(0, population, logbook, csv_logger)

    # Begin the evolution
    for gen in range(1, N_GEN + 1):
        # Select and clone offspring
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

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

        # Update population and Pareto front
        population[:] = offspring
        pareto.update(population)

        # Log metrics
        log_metrics(gen, population, logbook, csv_logger)

        # Switch enemy group every 25 generations
        if gen % 25 == 0:
            new_group = ENEMY_GROUPS[(gen // 25) % len(ENEMY_GROUPS)]
            train_env.update_parameter('enemies', new_group)
            print(f"Switched to enemy group: {new_group}")

    # Close CSV file
    csv_file.close()

    return population, logbook


# Main Function
def main():
    random.seed(42)

    # Initialize environments
    train_env = initialize_environment(
        EXPERIMENT_NAME,
        ENEMY_GROUPS[0],
        "yes",  # Multiple mode activated during training
        N_HIDDEN_NEURONS
    )

    eval_env = initialize_environment(
        EXPERIMENT_NAME,
        [1],  # This will be updated during evaluation
        "no",  # Single enemy evaluation
        N_HIDDEN_NEURONS
    )

    # Register evaluation function with the training environment
    toolbox.register("evaluate", evaluate, env=train_env)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=15)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=-1.0, up=1.0, eta=15, indpb=1.0 / NUM_WEIGHTS)
    toolbox.register("select", tools.selNSGA2)

    # Run the evolutionary algorithm
    population, logbook = evolutionary_algorithm(toolbox, train_env)

    # Plot metrics
    plot_metrics(logbook)

    # Test best solutions against all enemies
    best_solutions = tools.selBest(population, k=10)  # Select top 10 solutions
    all_enemy_results = []

    for solution in best_solutions:
        enemy_results = evaluate_individual_on_single_enemies(solution, eval_env)
        all_enemy_results.append(enemy_results)

    # Save results to CSV
    results_df = pd.DataFrame(all_enemy_results)
    results_df.to_csv(os.path.join(EXPERIMENT_NAME, 'all_enemy_results.csv'), index=False)

    # Analyze and print results
    analyze_results(all_enemy_results)

    # Save best overall individual
    best_overall = max(best_solutions, key=lambda ind: evaluate_individual_on_single_enemies(ind, eval_env)['total_gain'])
    np.savetxt(os.path.join(EXPERIMENT_NAME, 'best_solution.txt'), best_overall)

    print("Evolution complete. Results saved in", EXPERIMENT_NAME)


if __name__ == "__main__":
    main()