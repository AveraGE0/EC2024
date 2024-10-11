import pandas as pd
import matplotlib.pyplot as plt


def analyze_results(csv_file):
    # Load the CSV data
    df = pd.read_csv(csv_file)

    # 1. Plot average fitness metrics over generations
    plt.figure(figsize=(10, 6))
    plt.plot(df['Generation'], df['Avg_Enemy_Health'], label='Avg Enemy Health')
    plt.plot(df['Generation'], df['Avg_Player_Health'], label='Avg Player Health')
    plt.plot(df['Generation'], df['Avg_Time'], label='Avg Time')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Average Fitness Over Generations')
    plt.legend()
    plt.savefig('fitness_over_generations.png')
    plt.show()

    # 2. Plot Pareto front for player health vs enemy health
    plt.scatter(df['Avg_Player_Health'], df['Avg_Enemy_Health'], c=df['Generation'], cmap='viridis')
    plt.colorbar(label='Generation')
    plt.xlabel('Player Health')
    plt.ylabel('Enemy Health')
    plt.title('Pareto Front Across Generations')
    plt.savefig('pareto_front.png')
    plt.show()

    # 3. Generate Summary Stats
    summary = df[['Avg_Enemy_Health', 'Avg_Player_Health', 'Avg_Time']].describe()
    print(summary)

    # Save summary to file
    summary.to_csv('summary_statistics.csv')


# Example usage
analyze_results('experiment_nsga2/generation_statistics.csv')