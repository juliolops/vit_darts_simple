"""Plotting and result-aggregation helpers for MoQ-NAS.

Confusion-matrix/training-history plots, hypervolume statistics and
Pareto-front visualizations, extracted verbatim from ``utils/helpers.py``
(Block C of the refactor roadmap). ``helpers.py`` re-exports these names
until it becomes a facade (stage C.6), so both import paths are
equivalent.
"""
import os
import json
import pickle as pkl
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import GPUtil
from pymoo.indicators.hv import Hypervolume

from utils.io import load_pkl, load_retrain_results

def plot_confusion_matrix(confusion_matrix, labels):
    confusion_matrix= np.array(confusion_matrix)

    df_cm = pd.DataFrame(confusion_matrix, index = labels, columns = labels)
    plt.figure(figsize = (7,6))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', cbar=False, fmt='g')
    plt.title('Confusion matrix - Retrained model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def test_acc_mean_std(experiment_path, retrain_file_name):
    retrain_data = load_retrain_results(experiment_path, retrain_file_name)
    test_acc_mean = np.mean([retrain_data[key]['test_accuracy'] for key in retrain_data.keys()])
    test_acc_std = np.std([retrain_data[key]['test_accuracy'] for key in retrain_data.keys()])
    
    return test_acc_mean, test_acc_std

def agg_results(results_dict):
    # Create an empty dictionary to store the mean and std for each variable
    
    agg_results_dict = {
        "training_losses": [],
        "validation_losses": [],
        "training_accuracies": [],
        "validation_accuracies": [],
        # Add other variables as needed
    }
    # Loop through each dictionary and aggregate the results
    for key in results_dict.keys():
        current_dict = results_dict[key]  # Replace 'results_dicts' with the actual list of dictionaries
        agg_results_dict["training_losses"].append(current_dict["training_losses"])
        agg_results_dict["validation_losses"].append(current_dict["validation_losses"])
        agg_results_dict["training_accuracies"].append(current_dict["training_accuracies"])
        agg_results_dict["validation_accuracies"].append(current_dict["validation_accuracies"])
    
    # Convert the lists to NumPy arrays
    agg_results_dict["training_losses"] = np.array(agg_results_dict["training_losses"])
    agg_results_dict["validation_losses"] = np.array(agg_results_dict["validation_losses"])
    agg_results_dict["training_accuracies"] = np.array(agg_results_dict["training_accuracies"])
    agg_results_dict["validation_accuracies"] = np.array(agg_results_dict["validation_accuracies"])

    # Calculate the mean and std across the first axis (axis=0)
    agg_results_dict["mean_training_losses"] = np.mean(agg_results_dict["training_losses"], axis=0)
    agg_results_dict["std_training_losses"] = np.std(agg_results_dict["training_losses"], axis=0)
    agg_results_dict["mean_validation_losses"] = np.mean(agg_results_dict["validation_losses"], axis=0)
    agg_results_dict["std_validation_losses"] = np.std(agg_results_dict["validation_losses"], axis=0)
    agg_results_dict["mean_training_accuracies"] = np.mean(agg_results_dict["training_accuracies"], axis=0)
    agg_results_dict["std_training_accuracies"] = np.std(agg_results_dict["training_accuracies"], axis=0)
    agg_results_dict["mean_validation_accuracies"] = np.mean(agg_results_dict["validation_accuracies"], axis=0)
    agg_results_dict["std_validation_accuracies"] = np.std(agg_results_dict["validation_accuracies"], axis=0)
    
    return agg_results_dict

def plot_training_history(results_dict:dict, params:dict=None, retrain:bool=False, title:str=''):
    """ Plot the training history of a model.
    
    Args:
        results_dict: (dict) dictionary with the training history.
    """
    num_keys = len(results_dict.keys())
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    if retrain:
        if num_keys > 1:
            keys = list(results_dict.keys())
            total_epochs = len(results_dict[keys[0]]['training_losses'])
            epochs = range(1, total_epochs + 1)
            test_acc_mean = np.mean([results_dict[key]['test_accuracy'] for key in results_dict.keys()])
            test_acc_std = np.std([results_dict[key]['test_accuracy'] for key in results_dict.keys()])
            agg_results_dict = agg_results(results_dict)
            ax[0].plot(epochs, agg_results_dict["mean_training_losses"], label='Training', color='blue')
            ax[0].fill_between(epochs, 
                                agg_results_dict["mean_training_losses"] - agg_results_dict["std_training_losses"], 
                                agg_results_dict["mean_training_losses"] + agg_results_dict["std_training_losses"], 
                                color='blue', alpha=0.2)
            ax[0].plot(epochs, agg_results_dict["mean_validation_losses"], label='Validation', color='red')
            ax[0].fill_between(epochs, 
                                agg_results_dict["mean_validation_losses"] - agg_results_dict["std_validation_losses"], 
                                agg_results_dict["mean_validation_losses"] + agg_results_dict["std_validation_losses"], 
                                color='red', alpha=0.2)
            ax[0].set_title('Loss')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Loss')
            ax[0].legend(fontsize=12)
            ax[0].grid(True)
            ax[0].set_xlim([1, total_epochs])
            ax[0].set_ylim([0, 1.5])
            
            ax[1].plot(epochs, agg_results_dict["mean_training_accuracies"], label='Training', color='blue')
            ax[1].fill_between(epochs, 
                                agg_results_dict["mean_training_accuracies"] - agg_results_dict["std_training_accuracies"], 
                                agg_results_dict["mean_training_accuracies"] + agg_results_dict["std_training_accuracies"], 
                                color='blue', alpha=0.2)
            ax[1].plot(epochs, agg_results_dict["mean_validation_accuracies"], label='Validation', color='red')
            ax[1].fill_between(epochs, 
                                agg_results_dict["mean_validation_accuracies"] - agg_results_dict["std_validation_accuracies"], 
                                agg_results_dict["mean_validation_accuracies"] + agg_results_dict["std_validation_accuracies"], 
                                color='red', alpha=0.2)
            
            ax[1].axhline(y=test_acc_mean, color='green', linestyle='--', label='Test Accuracy')
            ax[1].text(epochs[-2], test_acc_mean+1, f'{test_acc_mean:.2f} ± {test_acc_std:.2f}', ha='right', va='center', color='black', fontsize=14)
            
            ax[1].set_title('Accuracy')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend(loc='lower right', fontsize=14)
            ax[1].grid(True)
            ax[1].set_xlim([1, total_epochs])
            # add plt title
            plt.suptitle(f'Training History: {title}', fontsize=16)
            plt.show()
        else:
            results_dict = results_dict[list(results_dict.keys())[0]]
            epochs = range(1, len(results_dict['training_losses']) + 1)
            ax[0].plot(epochs, results_dict["training_losses"], 'b', label='Training loss')
            ax[0].plot(epochs, results_dict["validation_losses"], 'r', label='Validation loss')
            ax[0].set_title('Loss')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].legend()
            ax[0].grid(True)
            
            ax[1].plot(epochs, results_dict["training_accuracies"], 'b', label='Training Acc')
            ax[1].plot(epochs, results_dict["validation_accuracies"], 'r', label='Validation Acc')
            max_acc, index = max(results_dict["validation_accuracies"]), results_dict["validation_accuracies"].index(max(results_dict["validation_accuracies"]))
            ax[1].plot(index+1, max_acc, 'go', label='Max Acc')
            ax[1].text(index+1, max_acc+0.1, f'{max_acc:.2f}', fontsize=12)
            ax[1].set_title('Accuracy')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend()
            ax[1].grid(True)
    else:
        epochs = range(1, len(results_dict['training_losses']) + 1)
        eval_starts = params["max_epochs"] - params["epochs_to_eval"]
        epochs_val = range(eval_starts+1, max(epochs)+1)
    
        ax[0].plot(epochs, results_dict["training_losses"], 'b', label='Training loss')
        ax[0].plot(epochs_val, results_dict["validation_losses"], 'r', label='Validation loss')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].grid(True)
        

        ax[1].plot(epochs, results_dict["training_accuracies"], 'b', label='Training Acc')
        ax[1].plot(epochs_val, results_dict["validation_accuracies"], 'r', label='Validation Acc')
        max_acc, index = max(results_dict["validation_accuracies"]), results_dict["validation_accuracies"].index(max(results_dict["validation_accuracies"]))
        ax[1].plot(index+1, max_acc, 'go', label='Max Acc')
        ax[1].text(index+1, max_acc+0.1, f'{max_acc:.2f}', fontsize=12)
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].grid(True)
    
    plt.show()

def get_gpu_memory():
    """
    Retrieve GPU memory usage using GPUtil.
    
    Returns:
    - Used memory in MB.
    """
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed  # Assuming single-GPU use; modify if using multiple GPUs
    return None

# NON-CANONICAL: hardcodes col0; replaced by algorithms.pareto.hypervolume in D-cleanup
def compute_hypervolume_mixed(front_raw: np.ndarray, ref_point=None) -> float:
    """
    Compute hypervolume for a 3-objective Pareto front where:
        - front_raw[:, 0] = accuracy (to be maximized)
        - front_raw[:, 1] = num_parameters (to be minimized)
        - front_raw[:, 2] = inference_time (to be minimized)
    We first convert everything into minimization form by flipping accuracy → -accuracy,
    then build a reference point slightly above the “worst” in each dimension,
    and finally call pymoo’s Hypervolume on that minimization front.

    .. deprecated::
        Non-canonical variant: it hardcodes the flip of column 0 and does
        NOT flip a user-supplied ``ref_point``, unlike the identical class
        methods in nsga2/moqnas (the canonical version). Kept only for the
        plotting helpers in this module; replaced by
        ``algorithms.pareto.hypervolume`` in the D-block cleanup. Do not
        use as a parity reference.

    Args:
        front_raw (np.ndarray): shape=(N, 3) with columns [acc, params, time].
        ref_point (np.ndarray): shape=(3,) with the reference point for hypervolume calculation.
    Returns:
        float: the hypervolume (in the original mixed‐obj space).
    """
    if front_raw is None or len(front_raw) == 0:
        return 0.0
    f = np.array(front_raw, dtype=float, copy=True)
    f[:, 0] = -f[:, 0]  # flip accuracy to minimization
    # Choose a safe reference point (must be worse than all points for minimization)
    if ref_point is None:
        rp = np.max(f, axis=0) + 1e-6
    else:
        rp = np.asarray(ref_point, dtype=float)
    return float(Hypervolume(ref_point=rp).do(f))

def plot_hypervolume_over_epochs(main_path, experiment_pattern="exp1_repeat"):

    folder_list = [f for f in os.listdir(main_path) if experiment_pattern in f]
    print(f"Found {len(folder_list)} folders matching the pattern '{experiment_pattern}'.")
    plt.figure(figsize=(10, 6))
    num_plots = 0

    for folder in folder_list:
        history_path = os.path.join(main_path, folder, "pareto_history.pkl")
        if not os.path.exists(history_path):
            continue

        with open(history_path, "rb") as pf:
            history = pkl.load(pf)

        generations = sorted(history.keys())
        hypervolumes = [history[gen].get("hypervolume", 0.0) for gen in generations]

        if generations and any(hypervolumes):
            plt.plot(generations, hypervolumes, marker='o', label=folder)
            num_plots += 1

    plt.title("Hypervolume over Generations for Each Experiment")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    if num_plots > 0:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _get_hypervolume_stats(path_or_pattern):
    """
    Helper function to load hypervolume data. It supports three modes:
    1. Path to a single run folder.
    2. Path to a main directory containing multiple run sub-folders.
    3. Path with a prefix to match specific run folders (e.g., "results/exp_").
    """
    run_paths_to_check = []

    # Mode 3: Path is a prefix pattern
    # This is likely if the path itself doesn't exist but its parent directory does.
    if not os.path.exists(path_or_pattern) and os.path.isdir(os.path.dirname(path_or_pattern)):
        parent_dir, prefix = os.path.split(path_or_pattern)
        # Handle case where parent_dir is empty (relative path)
        if not parent_dir:
            parent_dir = '.'
        print(f"Interpreting '{path_or_pattern}' as a prefix pattern.")
        run_paths_to_check = [
            f.path for f in os.scandir(parent_dir)
            if f.is_dir() and f.name.startswith(prefix)
        ]
    # If the path exists, handle Modes 1 and 2
    elif os.path.exists(path_or_pattern):
        # Mode 1: Path is a single run folder
        single_run_pkl = os.path.join(path_or_pattern, "pareto_history.pkl")
        if os.path.isfile(single_run_pkl):
            print(f"Interpreting '{path_or_pattern}' as a single run.")
            run_paths_to_check.append(path_or_pattern)
        # Mode 2: Path is a main directory containing run folders
        elif os.path.isdir(path_or_pattern):
            subdirs = [f.path for f in os.scandir(path_or_pattern) if f.is_dir()]
            if subdirs:
                print(f"Interpreting '{path_or_pattern}' as a main directory containing {len(subdirs)} run(s).")
                run_paths_to_check = subdirs

    if not run_paths_to_check:
        print(f"Warning: No runs found for path/pattern '{path_or_pattern}'")
        return np.array([]), np.array([]), np.array([])

    hvs_by_gen = defaultdict(list)
    # Process all identified runs from any of the modes
    for run_path in run_paths_to_check:
        history_path = os.path.join(run_path, "pareto_history.pkl")
        if not os.path.exists(history_path):
            continue
        with open(history_path, "rb") as pf:
            history = pkl.load(pf)
        for gen, data in history.items():
            if 'hypervolume' in data and isinstance(data['hypervolume'], (int, float)):
                hvs_by_gen[gen].append(data['hypervolume'])

    if not hvs_by_gen:
        print(f"Warning: No valid hypervolume data could be loaded. Please check your 'pareto_history.pkl' files.")
        return np.array([]), np.array([]), np.array([])

    generations = np.array(sorted(hvs_by_gen.keys()))
    mean_hvs = np.array([np.mean(hvs_by_gen[gen]) for gen in generations])
    std_hvs = np.array([np.std(hvs_by_gen[gen]) for gen in generations])
    return generations, mean_hvs, std_hvs

def plot_hypervolume_comparison(path_exp1, path_exp2, label_exp1="Method 1", label_exp2="Method 2"):
    """
    Compares the hypervolume evolution of two experiments by plotting their
    mean performance and standard deviation across multiple runs.

    Args:
        path_exp1 (str): Directory path for the first experiment. This directory
                        should contain a subfolder for each independent run.
        path_exp2 (str): Directory path for the second experiment.
        label_exp1 (str): Plot label for the first experiment.
        label_exp2 (str): Plot label for the second experiment.
    """
    # Get statistics for the first experiment
    print(f"--- Processing Experiment 1: {label_exp1} ---")
    gens1, means1, stds1 = _get_hypervolume_stats(path_exp1)

    # Get statistics for the second experiment
    print(f"\n--- Processing Experiment 2: {label_exp2} ---")
    gens2, means2, stds2 = _get_hypervolume_stats(path_exp2)

    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot for Experiment 1
    if len(gens1) > 0:
        ax.plot(gens1, means1, label=label_exp1, lw=2.5)
        ax.fill_between(gens1, means1 - stds1, means1 + stds1, alpha=0.2, label=f'{label_exp1} (Std. Dev.)')
    else:
        print(f"Could not plot '{label_exp1}' due to lack of data.")

    # Plot for Experiment 2
    if len(gens2) > 0:
        ax.plot(gens2, means2, label=label_exp2, lw=2.5)
        ax.fill_between(gens2, means2 - stds2, means2 + stds2, alpha=0.2, label=f'{label_exp2} (Std. Dev.)')
    else:
        print(f"Could not plot '{label_exp2}' due to lack of data.")
        
    # Final plot styling
    ax.set_title("Hypervolume Comparison", fontsize=16, weight='bold')
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    
    if len(gens1) > 0 or len(gens2) > 0:
        ax.legend(fontsize=11)
        
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()

# Pretty axis labels for known objective names; anything else falls back to
# its raw name, so arbitrary objective sets (Area 3) plot without code edits.
_OBJECTIVE_LABELS = {
    "accuracy": "Accuracy (%)",
    "best_accuracy": "Accuracy (%)",
    "params": "Params (M)",
    "total_params": "Params (count)",
    "inference_time": "Inference Time (µs)",
    "cuda_inference_time": "Inference Time (µs)",
    "total_flops": "FLOPs (count)",
}


def plot_pareto_evolution(history, dims="3d",
                            x=None, y=None, z=None,
                            width=1200, height=800, y_range=None):
    """
    Plot the evolution of Pareto fronts over generations en 2D o 3D con tooltips
    formateados a 2 decimales.

    The records' objective keys are taken as-is (whatever objective set the
    run used, e.g. best_accuracy/total_params/total_flops in
    pareto_history.pkl, or the legacy accuracy/params/inference_time from
    load_data_for_pareto). When x/y/z are not given they default to the
    record's objective keys in order (x=2nd, y=3rd, z=1st for 3D, matching
    the historical params/time/accuracy layout).
    """
    # 1) Flatten the history into a DataFrame, skipping the "hypervolume" key
    rows = []
    objective_keys = None
    for gen, fronts in history.items():
        for level, recs in fronts.items():
            # Skip any non-integer key (e.g. "hypervolume")
            if not isinstance(level, int):
                continue

            for rec in recs:
                values = {k: v for k, v in rec.items() if k != "id"}
                if objective_keys is None:
                    objective_keys = list(values)
                rows.append({
                    "generation": gen,
                    "front_level": level,
                    **values,
                })

    df = pd.DataFrame(rows)

    # 2b) Default axes from the objective keys present in the data
    if objective_keys:
        if x is None:
            x = objective_keys[1] if len(objective_keys) > 1 else objective_keys[0]
        if y is None:
            y = objective_keys[2] if len(objective_keys) > 2 else objective_keys[0]
        if z is None:
            z = objective_keys[0]
    axis_label = lambda k: _OBJECTIVE_LABELS.get(k, k)

    # 2) Build the plot in 3D or 2D
    if dims == "3d":
        fig = px.scatter_3d(
            df,
            x=x, y=y, z=z,
            color="front_level",
            animation_frame="generation",
            width=width, height=height,
            title="Pareto Front Evolution (3D)",
            labels={
                x: axis_label(x),
                y: axis_label(y),
                z: axis_label(z),
                "front_level": "Front Level",
                "generation": "Generation"
            },
            custom_data=["front_level", "generation"]
        )
        fig.update_traces(
            marker=dict(size=4),
            hovertemplate=(
                f"{axis_label(x)}: %{{x:.2f}}<br>"
                f"{axis_label(y)}: %{{y:.2f}}<br>"
                f"{axis_label(z)}: %{{z:.2f}}<br>"
                "Front Level: %{customdata[0]}<br>"
                "Generation: %{customdata[1]}<extra></extra>"
            )
        )
        fig.update_layout(
            scene=dict(
                yaxis=dict(range=[df[y].min() * 0.9, df[y].max() * 1.5],),
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )

    else:
        # 2D scatter
        fig = px.scatter(
            df,
            x=x, y=y,
            color="front_level",
            animation_frame="generation",
            width=width, height=height,
            title="Pareto Front Evolution (2D)",
            labels={
                x: axis_label(x),
                y: axis_label(y),
                "front_level": "Front Level",
                "generation": "Generation"
            },
            custom_data=["front_level", "generation"]
        )
        fig.update_traces(
            marker=dict(size=6),
            hovertemplate=(
                f"{axis_label(x)}: %{{x:.2f}}<br>"
                f"{axis_label(y)}: %{{y:.2f}}<br>"
                "Front Level: %{customdata[0]}<br>"
                "Generation: %{customdata[1]}<extra></extra>"
            )
        )
        if y_range is not None:
            fig.update_layout(yaxis=dict(range=y_range))
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))

    fig.show()

def load_data_for_pareto(file_path):
    """
    Loads and processes data from a single large JSON results file and 
    formats it for the plot_pareto_evolution function.

    This version correctly reads the entire file as a single JSON object.

    Args:
        file_path (str): The path to the input JSON file.

    Returns:
        dict: A dictionary formatted for the plot_pareto_evolution function.
    """
    # Open and load the entire file as a single JSON object
    with open(file_path, 'r') as f:
        full_data = json.load(f)

    # Since 'generation' and 'front_level' are not in the source data,
    # we place all results into a single generation (0) and front (0) for plotting.
    pareto_front_records = []

    # Iterate through each main key (e.g., "0_17", "0_2") in the JSON data
    for key, retrain_data in full_data.items():
        # Lists to store metrics from each retrain trial for averaging
        accuracies = []
        params_list = []
        inference_times = []

        # Iterate through each retrain instance (e.g., "retrain_1", "retrain_2")
        for retrain_key, metrics in retrain_data.items():
            # Ensure the retrain entry has the necessary metrics
            if "test_accuracy" in metrics and "total_params" in metrics and "cuda_inference_time" in metrics:
                accuracies.append(metrics['test_accuracy'])
                params_list.append(metrics['total_params'])
                inference_times.append(metrics['cuda_inference_time'])

        # If we found any valid retrain data, calculate the means
        if accuracies:
            mean_accuracy = statistics.mean(accuracies)
            mean_params = statistics.mean(params_list)
            mean_inference_time = statistics.mean(inference_times)
            
            pareto_front_records.append({
                "accuracy": mean_accuracy,
                # Convert params to millions (M) for a more readable plot scale
                "params": mean_params / 1_000_000, 
                "inference_time": mean_inference_time
            })

    # Structure the data into the nested dictionary format expected by the plotting function
    history = {
        0: {  # Generation 0
            0: pareto_front_records  # Front Level 0
        }
    }
    return history
