
# EC2024 - Task 2 General Agent

This project contains the `main_nsga2.py` model, an implementation for running evolutionary algorithms in batch and single-run modes.

## Features

- Configurable through `config_nsga2.yaml`
- Parallel batch processing
- Logging and analysis tools
- Optional deterministic crowding for diversity

## Getting Started

1. Clone the Repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to Your Branch:
   ```bash
   git checkout task2-general-agent
   ```

3. Install Dependencies:
   Ensure you have all required Python packages installed.

## Configuration

Edit `config_nsga2.yaml` to set your experiment’s parameters, such as:

- Neural network settings
- Population size
- Number of generations
- Evolutionary algorithm options

## Running the Model

Run the following command to execute the model:

```bash
python main_nsga2.py
```

## Output

Results are saved in a directory named after your experiment, as specified in the config file.
## Plotting Results

Generate plots using the following command:

```bash
python plot_ea_results.py
```
