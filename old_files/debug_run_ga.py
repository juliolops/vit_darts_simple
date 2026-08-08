#!/usr/bin/env python3
import os
from run_ga_evolution import main  # assumes run_evolution.py is on your PYTHONPATH

# — optional: lock to a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# — experiment parameters
params = {
    # Core run_evolution args:
    "experiment_path":      "experiment_cifar10_gaX/exp1_repeat_1",
    "data_path":            "cifar10_data",
    "dataset":              "cifar10",
    "config_file":          "config_files_cifar/config0.txt",
    "continue_path":        "",                # or a path to resume
    "log_level":            "DEBUG",
    "optimizer":            "AdamW",
    "fitness_metric":       "best_accuracy",
    "data_augmentation":    False,
    "early_stopping":       True,
    "en_pop_crossover":     True,
    "save_checkpoints_epochs": 5,
    "limit_data_value":        10000,
    "backbone_name":           "resnet18",
    "network_config":          "default",

    # GA-specific args:
    "population_size":      4,                # e.g. 50 individuals
    "num_generations":      50,               # e.g. 100 generations
    "max_num_nodes":        5,                # chromosome length
    "crossover_rate":       0.5,
    "mutation_rate":        0.2,
    "elitism":              True,
    "patience":             60,                # early‑stop patience in gens
}

if __name__ == "__main__":
    main(**params)
