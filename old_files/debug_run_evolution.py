import os
from run_evolution2 import main  # Import the main function from your run_evolution.py

# If necessary, set environment variables (simulate what your shell script does)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define parameters similar to those defined in your .sh script
parameters = {
    "experiment_path": "experiment_cifar10_qnasXX/exp1_repeat_2",
    "data_path": "cifar10_data",
    "dataset": "cifar10",
    "config_file": "config_files_cifar/config0_0.txt",
    "continue_path": "",
    "log_level": "INFO",  # Using DEBUG for detailed output during development
    "optimizer": "AdamW",
    "fitness_metric": "best_accuracy",
    "data_augmentation": False,  # Set as needed for your experiment
    "early_stopping": True,
    "en_pop_crossover": True,
    "save_checkpoints_epochs": 5,
    "limit_data_value": 10000,
    "backbone_name": "resnet18",
    "network_config": "default",
}

if __name__ == "__main__":
    # Call the main function with the specified parameters.
    main(**parameters)
