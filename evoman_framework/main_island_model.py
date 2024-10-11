import numpy as np
import random
import os
from evoman.environment import Environment
from demo_controller import player_controller

# --------------------------
# Parameters
# --------------------------

num_islands = 4
island_population = 50
num_generations = 20
migration_interval = 5
migration_rate = 0.1
tournament_size = 3
crossover_rate = 0.9
mutation_rate = 0.1
mutation_scale = 0.1

# Define the number of sensors (inputs) and actions (outputs)
num_inputs = 20  # Number of inputs (depends on the game)
num_outputs = 5  # Number of outputs (actions)
n_hidden_neurons = 10  # Adjust as needed

# Calculate the number of weights in the neural network
num_weights = (
    (num_inputs * n_hidden_neurons) +  # Input to hidden layer weights
    (n_hidden_neurons * num_outputs) +  # Hidden to output layer weights
    n_hidden_neurons +  # Hidden layer biases
    num_outputs  # Output layer biases
)

# Define the experiment name as the full path
experiment_name = '/Users/mert/Library/CloudStorage/OneDrive-Personal/Documents/Persoonlijk/Study/VU/Year 2/Period 1/Evolutionary Computing/Assignments/Assignment Evoman Task 2/EC2024/evoman_framework/experiment_island_model'

# Create the experiment directory if it doesn't exist
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# --------------------------
# Environment Setup
# --------------------------

# Set up the environment with desired parameters
env = Environment(
    experiment_name=experiment_name,
    enemies=[1, 2, 3],  # Replace with your chosen enemies
    multiplemode="yes",  # Enable multiple enemy mode for a generalist agent
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    contacthurt='player',
    speed="fastest"
)

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
    for _ in range(len(population)):
        # Randomly select individuals for the tournament
        participants = random.sample(list(zip(population, fitness_scores)), k=tournament_size)
        # Choose the best among them
        winner = max(participants, key=lambda x: x[1])
        selected_parents.append(winner[0])
    return selected_parents

def crossover(parents, crossover_rate=0.9, eta=15):
    """Applies simulated binary crossover (SBX) to generate offspring."""
    offspring = []
    pop_size = len(parents)
    for i in range(0, pop_size, 2):
        parent1 = parents[i]
        parent2 = parents[(i + 1) % pop_size]
        if random.random() <= crossover_rate:
            child1, child2 = sbx_operator(parent1, parent2, eta)
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

def mutation(offspring, mutation_rate=0.1, sigma=0.1):
    """Applies Gaussian mutation to the offspring."""
    for individual in offspring:
        for i in range(len(individual)):
            if random.random() <= mutation_rate:
                individual[i] += np.random.normal(0, sigma)
    return offspring

def migrate(islands, migration_rate):
    """Performs migration of individuals between islands."""
    num_migrants = int(migration_rate * len(islands[0]['population']))
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

def log_statistics(islands, generation, filename):
    """Logs statistics for each generation to a file and prints them."""
    # Use the full path for the filename
    filepath = os.path.join(experiment_name, filename)

    # Check if it's the first generation to write the header
    if generation == 0:
        with open(filepath, 'w') as f:
            # Write the header
            header = 'Generation'
            for idx in range(len(islands)):
                header += (f'\tIsland_{idx}_Mean_Fitness\tIsland_{idx}_Max_Fitness'
                           f'\tIsland_{idx}_Min_Fitness\tIsland_{idx}_Std_Fitness')
            f.write(header + '\n')

    # Prepare the data for the current generation
    data_line = f'{generation}'
    for idx, island in enumerate(islands):
        mean_fitness = np.mean(island['fitness'])
        max_fitness = np.max(island['fitness'])
        min_fitness = np.min(island['fitness'])
        std_fitness = np.std(island['fitness'])

        data_line += f'\t{mean_fitness:.4f}\t{max_fitness:.4f}'
        data_line += f'\t{min_fitness:.4f}\t{std_fitness:.4f}'

        # Print the statistics to the console
        print(f"Generation {generation}, Island {idx}: Mean Fitness = {mean_fitness:.4f}, "
              f"Max Fitness = {max_fitness:.4f}, Min Fitness = {min_fitness:.4f}, "
              f"Std Fitness = {std_fitness:.4f}")

    # Append the data to the file
    with open(filepath, 'a') as f:
        f.write(data_line + '\n')

def test_against_all_enemies(individual):
    """Tests an individual against all enemies and calculates the total gain."""
    total_gain = 0
    for enemy_id in range(1, 9):
        # Update the enemy in the environment
        test_env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy_id],
            multiplemode="no",
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            contacthurt='player',
            speed="fastest"
        )
        fitness, player_life, enemy_life, time = test_env.play(pcont=individual)
        gain = player_life - enemy_life
        total_gain += gain
    return total_gain

# --------------------------
# Main Evolutionary Loop
# --------------------------

# Initialize islands
islands = []
for _ in range(num_islands):
    island = {
        'population': initialize_population(island_population, num_weights),
        'fitness': None
    }
    islands.append(island)

# Define the filename for logging
filename = 'island_model_metrics.txt'  # Filename without path

# Evolutionary loop
for generation in range(num_generations):
    for island in islands:
        # Evaluate Fitness
        island['fitness'] = evaluate_population(island['population'])

        # Selection
        parents = selection(island['population'], island['fitness'])

        # Crossover
        offspring = crossover(parents, crossover_rate, eta=15)

        # Mutation
        offspring = mutation(offspring, mutation_rate, mutation_scale)

        # Create New Population
        island['population'] = offspring

    # Migration Step
    if generation % migration_interval == 0 and generation != 0:
        migrate(islands, migration_rate)

    # Logging and Statistics
    log_statistics(islands, generation, filename)

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
best_overall = max(best_individuals, key=lambda ind: evaluate_individual(ind))

# Save the best overall individual to the experiment directory
np.savetxt(os.path.join(experiment_name, 'groupnumber.txt'), best_overall)