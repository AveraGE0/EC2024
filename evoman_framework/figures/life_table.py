import os
import pickle
import pandas as pd
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
from tabulate import tabulate


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # path to experiment with best solution
    experiment_path = "../experiments/config_island_final_0"
    with open(os.path.join(experiment_path, "best_individual_multi_gain.pkl"), mode="rb") as b_file:
        individual = np.array(pickle.load(b_file))
    
    df_life_table = pd.DataFrame({"Enemy": [], "Player Health": [], "Enemy Health": []})

    for enemy in range(1, 9):
        env = Environment(
                experiment_name="test",
                enemies=[enemy],
                playermode="ai",
                player_controller=player_controller(10),
                enemymode="static",
                level=2,
                speed="fastest")
        #print(individual)
        f, p, e, t = env.play(pcont=individual)
        df_life_table = pd.concat([
            df_life_table,
            pd.DataFrame({"Enemy": [int(enemy)], "Player Health": [p], "Enemy Health": [e]})
        ])
    df_life_table["Enemy"] = df_life_table["Enemy"].astype(int)
    df_life_table = df_life_table.map(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    df_life_table = df_life_table.astype(str)

    df_life_table = df_life_table.T

    # Use tabulate to generate a LaTeX table
    latex_table = tabulate(df_life_table, tablefmt="latex")
    # Add full grid: vertical (|l|c|c|c|) and horizontal lines (\hline)
    latex_table = latex_table.replace("\\hline", "")  # Avoid duplicate hlines from tabulate
    latex_table = latex_table.replace("\\begin{tabular}{lrrrrrrrr}\n", "\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\n\\hline")  # Add vertical lines

    # Insert \hline after each row
    lines = latex_table.splitlines()
    for i in range(2, len(lines) - 1):
        if lines[i].endswith("\\\\"):
            lines[i] += "\n\\hline"
    latex_table = "\n".join(lines)


    latex_table_template = f"""
\\begin{{table}}[h!]
\\centering
{latex_table}
\\caption{{Enemy and player lives for the overall best solution for all enemies.}}
\\label{{tab:lives_table}}
\\end{{table}}
    """
    latex_table_template = latex_table_template\
        .replace("Enemy", "\\textbf{Enemy}", 1)\
        .replace("Player Health", "\\textbf{Player Health}")\
        .replace("Enemy Health", "\\textbf{Enemy Health}")
    # Save to file (optional)
    with open("../a2_plots/island_all_enemies_life_table.tex", mode='w', encoding="utf-8") as f:
        f.write(latex_table_template)