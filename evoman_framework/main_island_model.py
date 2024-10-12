import numpy as np
import random
import os
import csv
import matplotlib.pyplot as plt
from evoman.environment import Environment
from demo_controller import player_controller

# --------------------------
# Parameters
# --------------------------

NUM_ISLANDS = 4
ISLAND_POPULATION = 50
NUM_GENERATIONS = 20
MIGRATION_INTERVAL = 5
MIGRATION_RATE = 0.1
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
MUTATION_SCALE = 0.1
ETA = 15  # For SBX crossover

# Neural network parameters
NUM_INPUTS = 20
NUM_OUTPUTS = 5
N_HIDDEN_NEURONS = 10

# Calculate the number of weights in the neural network
NUM_WEIGHTS = (
    (NUM_INPUTS * N_HIDDEN_NEURONS) +  # Input to hidden layer weights
    (N_HIDDEN_NEURONS * NUM_OUTPUTS) +  # Hidden to output layer weights
    N_HIDDEN_NEURONS +  # Hidden layer biases
    NUM_OUTPUTS  # Output layer biases
)

# Experiment name and directory
EXPERIMENT_NAME = '/Users/mert/Library/CloudStorage/OneDrive-Personal/Documents/Persoonlijk/Study/VU/Year 2/Period 1/Evolutionary Computing/Assignments/Assignment Evoman Task 2/EC2024/evoman_framework/experiment_island_model'

# Create the experiment directory if it doesn't exist
if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)

# --------------------------
# Environment Setup
# --------------------------

def initialize_environment(enemies, multiplemode):
    """Initialize the Evoman environment."""
    return Environment(
        experiment_name=EXPERIMENT_NAME,
        enemies=enemies,
        multiplemode=multiplemode,
        playermode="ai",
        player_controller=player_controller(N_HIDDEN_NEURONS),
        enemymode="static",
        level=2,
        contacthurt='player',
        speed="fastest"
    )

# Initialize the environment for training (multiple enemies)
env = initialize_environment(enemies=[1, 2, 3], multiplemode="yes")

# --------------------------
# Evolutionary Algorithm Functions
# --------------------------

def initialize_population(size, num_weights):
    """Initializes a population with random weights."""
    return [np.random.uniform(-1, 1, num_weights) for _ in range(size)]

def evaluate_population(population):
    """Evaluates the fitness of each individual in the population."""
    fitness_scores = []
    for individual in population:
        fitness = evaluate_individual(individual)
        fitness_scores.append(fitness)
    return fitness_scores

def evaluate_individual(individual):
    """Evaluates an individual by playing the game and returning the fitness."""
    fitness, player_life, enemy_life, time = env.play(pcont=individual)
    return fitness

def selection(population, fitness_scores):
    """Performs tournament selection to choose parents for reproduction."""
    selected_parents = []
    pop_fitness = list(zip(population, fitness_scores))
    for _ in range(len(population)):
        # Randomly select individuals for the tournament
        participants = random.sample(pop_fitness, k=TOURNAMENT_SIZE)
        # Choose the best among them
        winner = max(participants, key=lambda x: x[1])
        selected_parents.append(winner[0])
    return selected_parents

def crossover(parents):
    """Applies simulated binary crossover (SBX) to generate offspring."""
    offspring = []
    pop_size = len(parents)
    for i in range(0, pop_size, 2):
        parent1 = parents[i]
        parent2 = parents[(i + 1) % pop_size]
        if random.random() <= CROSSOVER_RATE:
            child1, child2 = sbx_operator(parent1, parent2, ETA)
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1.copy(), parent2.copy()])
    return offspring

def sbx_operator(parent1, parent2, eta):
    """Performs the Simulated Binary Crossover (SBX) operation."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(len(parent1)):
        u = random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
        child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
    return child1, child2

def mutation(offspring):
    """Applies Gaussian mutation to the offspring."""
    for individual in offspring:
        for i in range(len(individual)):
            if random.random() <= MUTATION_RATE:
                individual[i] += np.random.normal(0, MUTATION_SCALE)
    return offspring

def migrate(islands):
    """Performs migration of individuals between islands."""
    num_migrants = int(MIGRATION_RATE * len(islands[0]['population']))
    for i in range(len(islands)):
        source_island = islands[i]
        target_island = islands[(i + 1) % len(islands)]
        # Select migrants (e.g., top performers)
        migrants, migrant_fitnesses = select_migrants(
            source_island['population'], source_island['fitness'], num_migrants)
        # Replace worst individuals in target island
        replace_individuals(
            target_island['population'], target_island['fitness'], migrants, migrant_fitnesses)

def select_migrants(population, fitness_scores, num_migrants):
    """Selects the top-performing individuals to migrate."""
    # Pair population with fitness
    paired = list(zip(population, fitness_scores))
    # Sort by fitness (descending order; best individuals first)
    sorted_paired = sorted(
        paired, key=lambda x: x[1] if x[1] is not None else float('-inf'), reverse=True)
    # Select top individuals and their fitnesses
    migrants = [individual for individual, fitness in sorted_paired[:num_migrants]]
    migrant_fitnesses = [fitness for individual, fitness in sorted_paired[:num_migrants]]
    return migrants, migrant_fitnesses

def replace_individuals(population, fitness_scores, migrants, migrant_fitnesses):
    """Replaces the worst-performing individuals with migrants."""
    # Pair population with fitness
    paired = list(zip(population, fitness_scores))
    # Sort by fitness (ascending order; worst individuals first)
    sorted_paired = sorted(
        paired, key=lambda x: x[1] if x[1] is not None else float('inf'))
    # Replace worst individuals
    for i in range(len(migrants)):
        sorted_paired[i] = (migrants[i], migrant_fitnesses[i])
    # Unzip the list back to population and fitness_scores
    population[:], fitness_scores[:] = zip(*sorted_paired)

def test_against_all_enemies(individual):
    """Tests an individual against all enemies and calculates the total gain."""
    total_gain = 0
    for enemy_id in range(1, 9):
        # Update the enemy in the environment
        test_env = initialize_environment(enemies=[enemy_id], multiplemode="no")
        fitness, player_life, enemy_life, time = test_env.play(pcont=individual)
        gain = player_life - enemy_life
        total_gain += gain
    return total_gain

# --------------------------
# Logging and Statistics
# --------------------------

def log_statistics(islands, generation, logbook):
    """Logs statistics for each generation and prints them."""
    record = {'generation': generation}
    for idx, island in enumerate(islands):
        mean_fitness = np.mean(island['fitness'])
        max_fitness = np.max(island['fitness'])
        min_fitness = np.min(island['fitness'])
        std_fitness = np.std(island['fitness'])

        record[f'island_{idx}_mean'] = mean_fitness
        record[f'island_{idx}_max'] = max_fitness
        record[f'island_{idx}_min'] = min_fitness
        record[f'island_{idx}_std'] = std_fitness

        # Print the statistics to the console
        print(f"Generation {generation}, Island {idx}: Mean Fitness = {mean_fitness:.4f}, "
              f"Max Fitness = {max_fitness:.4f}, Min Fitness = {min_fitness:.4f}, "
              f"Std Fitness = {std_fitness:.4f}")

    logbook.append(record)

def save_logbook(logbook):
    """Saves the logbook to a CSV file."""
    keys = logbook[0].keys()
    with open(os.path.join(EXPERIMENT_NAME, 'island_model_metrics.csv'), 'w', newline='') as csvfile:
        dict_writer = csv.DictWriter(csvfile, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(logbook)

def plot_metrics(logbook):
    """Plots the metrics over generations."""
    generations = [entry['generation'] for entry in logbook]
    num_islands = (len(logbook[0]) - 1) // 4  # Exclude 'generation' key

    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    for idx in range(num_islands):
        mean_fitness = [entry[f'island_{idx}_mean'] for entry in logbook]
        max_fitness = [entry[f'island_{idx}_max'] for entry in logbook]
        min_fitness = [entry[f'island_{idx}_min'] for entry in logbook]
        std_fitness = [entry[f'island_{idx}_std'] for entry in logbook]

        axs[0].plot(generations, mean_fitness, label=f'Island {idx}')
        axs[1].plot(generations, max_fitness, label=f'Island {idx}')
        axs[2].plot(generations, min_fitness, label=f'Island {idx}')
        axs[3].plot(generations, std_fitness, label=f'Island {idx}')

    axs[0].set_title('Mean Fitness over Generations')
    axs[1].set_title('Max Fitness over Generations')
    axs[2].set_title('Min Fitness over Generations')
    axs[3].set_title('Std Fitness over Generations')

    for ax in axs:
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_NAME, 'island_model_metrics.png'))
    plt.close()

# --------------------------
# Main Evolutionary Loop
# --------------------------

def main():
    # Initialize islands
    islands = []
    for island_idx in range(NUM_ISLANDS):
        island = {
            'population': initialize_population(ISLAND_POPULATION, NUM_WEIGHTS),
            'fitness': None
        }
        islands.append(island)

    # Initialize logbook
    logbook = []

    # Evolutionary loop
    for generation in range(NUM_GENERATIONS):
        for island in islands:
            # Evaluate Fitness
            island['fitness'] = evaluate_population(island['population'])

            # Selection
            parents = selection(island['population'], island['fitness'])

            # Crossover
            offspring = crossover(parents)

            # Mutation
            offspring = mutation(offspring)

            # Create New Population
            island['population'] = offspring

        # Migration Step
        if generation % MIGRATION_INTERVAL == 0 and generation != 0:
            migrate(islands)

        # Logging and Statistics
        log_statistics(islands, generation, logbook)

    # Save logbook to CSV
    save_logbook(logbook)

    # Plot metrics
    plot_metrics(logbook)

    # After evolution, evaluate best individuals from each island
    best_individuals = []
    for island in islands:
        best_idx = np.argmax(island['fitness'])
        best_individuals.append(island['population'][best_idx])

    # Test the best individuals against all enemies
    gains = []
    for individual in best_individuals:
        total_gain = test_against_all_enemies(individual)
        gains.append(total_gain)

    # Find the best overall individual
    best_overall = best_individuals[np.argmax(gains)]

    # Save the best overall individual to the experiment directory
    np.savetxt(os.path.join(EXPERIMENT_NAME, 'best_overall_solution.txt'), best_overall)

    print("Evolution complete. Results saved in", EXPERIMENT_NAME)

if __name__ == "__main__":
    main()