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


## Creating plots

To create the plots for assignment 1, either run the evolution_specialist and adjust the names
in the evoman_framework/figures/a1_figures.py script or get a copy of the original data.
```bash
python -m figures.a1_figures
```

## Notes

6 defeated with:
and 

FitnessChangeWeighting
relative_increase: float = 0.2
### general
name: "competition_test"
population_size: 400
generations: 60

### Neural Network
n_inputs: 20
hidden_size: 10
n_outputs: 5
init_low: -10.0
init_up: 10.0

### crossover
p_crossover: 0.8
SBX_eta: 5.0

### mutation
p_mutation: 0.3

polynomial_eta: 10.0
polynomial_low: -10.0
polynomial_up: 10.0
polynomial_indpb: 0.7

### offspring selection
sel_tournament_size: 3

### replacement selection
rep_tournament_size: 2
elitism_size: 2  # 0 to turn off
sigma: 8

### island parameters
islands: 5
migration_rate: 0.1
migration_interval: 5
migration_tournament_size: 2

### environment specific
multiplemode: "yes"
level: 2
train_enemy: [1, 2, 3, 4, 5, 6, 7, 8]
test_enemies: [2, 5, 7]
repeat: 10


Also 6 with same and
DefeatedProportionalWeighter(step_size: int = 0.2)
Fitness sharing, fitness_sharing,
        #distance_func=same_loss,
        #distance_property="defeated",
        #tournsize=config["rep_tournament_size"],
        #sigma=config["sigma"]

(possible hardness)
[0.247 0.943 0.897 0.985 1.    0.048 0.7   1.   ]