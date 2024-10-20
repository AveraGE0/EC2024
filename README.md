# EC2024

The repo already contains a version of the EVOMAN frame work (17.09.2024).

## Installation

pip install -r requirements.txt

## Testing if it runs

Go into the ***evoman_framework/*** directory (`cd evoman_framework`) $\rightarrow$ 

```bash 
python3 evolution_specialist.py
```

Plots can only be generated with the data of runs which is not included in the github project!

## Task 2 run

Go into evoman_framework and run 

- python evolution_generalist_island.py
- python evolution_multi_objective.py

## Creating plots

To create the plots for assignment 1, either run the evolution_specialist and adjust the names
in the evoman_framework/figures/a1_figures.py script or get a copy of the original data.
```bash
python -m figures.a1_figures
```