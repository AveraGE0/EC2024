"""
This code is designed to implement multi-objective evolutionary optimization using DEAP,
specifically tracking fitness metrics over multiple generations and multiple runs of the
evolutionary algorithm. It provides the functionality to calculate and log both per-generation
and per-run fitness metrics for easier performance comparison of different models.

### Key Components:
1. **log_generation_fitness()**:
    - Logs and calculates the average and max fitness for each generation within a single run.
    - This function is called at the end of each generation after evaluating all individuals in the population.

2. **run_evolution_multiple_times()**:
    - Runs the evolutionary algorithm multiple times (e.g., 10 times) and stores the fitness values for each run.
    - Fitness values are stored as lists, tracking the average and max fitness for each generation during each run.

3. **aggregate_fitness_data()**:
    - Aggregates fitness data over multiple runs, computing the average of the average fitness and the average of the max fitness for each generation across all runs.
    - This allows for easier performance comparison across multiple experiments.

4. **log_aggregated_results()**:
    - Logs the aggregated fitness metrics for each generation after running the evolutionary algorithm multiple times.
    - Provides a clear report on the "average of the average fitness" and the "average of the max fitness" over all runs.

### How It Works:
- **Per Generation**:
    - After evaluating the population for each generation, the fitness values (e.g., `avg_fitness` and `max_fitness`) are calculated and logged.
    - These metrics are tracked to assess how the population performs across generations.

- **Per Run**:
    - The evolutionary process can be run multiple times (e.g., 10 runs).
    - The fitness data for each generation within each run is stored and aggregated later.

- **Aggregation Across Runs**:
    - After all runs are complete, the fitness values are aggregated to calculate the overall average performance of the model.
    - This helps to understand how consistent the model's performance is across different runs and random initial conditions.

### How to Look Up or Modify This Code:
1. **DEAP Documentation**:
    - To revisit how DEAP handles multi-objective optimization, you can refer to DEAP’s official documentation:
      - https://deap.readthedocs.io/en/master/
    - Particularly, look into sections on:
      - `creator.create()` for defining fitness weights.
      - `toolbox.register()` for registering functions like `evaluate()` and mutation/selection operators.

2. **Logging and Fitness Calculation**:
    - If you want to adjust how fitness is calculated (e.g., using a different metric), you can modify the `log_generation_fitness()` function to reflect those changes.
    - The `custom_fitness` function can be adjusted to reflect new fitness formulas, ensuring that you log the appropriate fitness values for comparison.

3. **Saving and Loading Results**:
    - If you want to save the fitness results to a file (e.g., CSV), you can extend the `log_aggregated_results()` function to write the fitness data to a file.
    - Similarly, you can modify the function to load results from a previous run if needed.

4. **Aggregation of Multiple Runs**:
    - If you wish to modify the number of runs or how data is aggregated, the `run_evolution_multiple_times()` and `aggregate_fitness_data()` functions are the key places to make adjustments.
    - You can adjust the number of runs or change the aggregation logic to include additional metrics or other forms of aggregation (e.g., median instead of mean).

### Next Steps:
- Ensure that the population size, number of generations, and the number of runs match your experimental setup.
- If running on large datasets or multiple generations, consider storing intermediate results in a file to avoid losing data due to unexpected interruptions.
- After running multiple experiments, use the `log_aggregated_results()` function to review the overall performance of your models.
"""

import numpy as np
from typing import List, Dict, Tuple




def log_generation_fitness(population: List, generation: int, logger_instance=None) -> Tuple[float, float]:
    """Log the average and max fitness for the current generation."""

    # Extract the fitness values of each individual
    fitness_values = [ind.fitness.values for ind in population]

    # Calculate average and max fitness
    avg_fitness = np.mean(fitness_values)
    max_fitness = np.max(fitness_values)

    # Log the results for this generation
    if logger_instance:
        logger_instance.info(f"Generation {generation}: Avg Fitness = {avg_fitness}, Max Fitness = {max_fitness}")

    # Optionally return these values for further aggregation later
    return avg_fitness, max_fitness


def run_evolution_multiple_times(num_runs: int, toolbox, generations: int, pop_size: int):
    """Run the evolutionary process multiple times and aggregate fitness data."""
    all_run_data = {
        'avg_fitness_per_gen': [],
        'max_fitness_per_gen': []
    }

    for run in range(num_runs):
        print(f"Starting Run {run + 1}")

        # Initialize the population for the run
        population = toolbox.population(n=pop_size)

        avg_fitness_per_gen = []
        max_fitness_per_gen = []

        # Run the evolution for the specified number of generations
        for gen in range(generations):
            # Evaluate the population and perform selection, crossover, mutation, etc.
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Calculate and store average and max fitness for this generation
            avg_fitness, max_fitness = log_generation_fitness(population, gen, logger_instance=None)
            avg_fitness_per_gen.append(avg_fitness)
            max_fitness_per_gen.append(max_fitness)

        # Append this run's data to the master list
        all_run_data['avg_fitness_per_gen'].append(avg_fitness_per_gen)
        all_run_data['max_fitness_per_gen'].append(max_fitness_per_gen)

    return all_run_data


def aggregate_fitness_data(all_run_data: Dict[str, List[List[float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate the fitness data across multiple runs."""

    # Convert lists to numpy arrays for easier aggregation
    avg_fitness_array = np.array(all_run_data['avg_fitness_per_gen'])
    max_fitness_array = np.array(all_run_data['max_fitness_per_gen'])

    # Calculate the average of the average fitness across runs, per generation
    avg_of_avg_fitness_per_gen = np.mean(avg_fitness_array, axis=0)  # Average across runs for each generation

    # Calculate the average of the max fitness across runs, per generation
    avg_of_max_fitness_per_gen = np.mean(max_fitness_array, axis=0)

    return avg_of_avg_fitness_per_gen, avg_of_max_fitness_per_gen


def log_aggregated_results(avg_of_avg_fitness_per_gen: np.ndarray, avg_of_max_fitness_per_gen: np.ndarray):
    """Log the aggregated results of fitness over multiple runs."""
    for gen, (avg_fitness, max_fitness) in enumerate(zip(avg_of_avg_fitness_per_gen, avg_of_max_fitness_per_gen)):
        print(f"Generation {gen + 1}: Avg of Avg Fitness = {avg_fitness}, Avg of Max Fitness = {max_fitness}")


