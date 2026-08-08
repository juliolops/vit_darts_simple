"""File I/O helpers for MoQ-NAS.

YAML/JSON/pickle loading and saving, atomic writes, evaluation-cache
backups and experiment-artifact readers, extracted verbatim from
``utils/helpers.py`` (Block C of the refactor roadmap). ``helpers.py``
re-exports these names until it becomes a facade (stage C.6), so both
import paths are equivalent.
"""
import os
import json
import tempfile
import pickle as pkl
from pickle import dump, load, HIGHEST_PROTOCOL
from typing import Dict

import yaml


def load_yaml(file_path):
    """ Wrapper to load a yaml file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        dict with loaded parameters.
    """

    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def _deep_merge(dst: dict, src: dict) -> dict:
    """Deep-merge ``src`` into ``dst`` (in place).

    Parameters
    ----------
    dst : dict
        Destination mapping, modified in place.
    src : dict
        Source mapping; nested dicts are merged recursively, any other
        value overwrites the one in ``dst``.

    Returns
    -------
    dict
        The mutated ``dst``.
    """
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def _atomic_write_yaml(file_path: str, data: dict):
    """Write YAML atomically (temp file + ``os.replace``).

    Parameters
    ----------
    file_path : str
        Destination path; parent directories are created if missing.
    data : dict
        Mapping to serialize with ``yaml.safe_dump``.
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".params.", suffix=".tmp", dir=os.path.dirname(file_path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,         # keep human-friendly order if present
                default_flow_style=False,
                allow_unicode=True
            )
        os.replace(tmp_path, file_path)  # atomic on POSIX
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def update_yaml_file(file_path: str, patch: dict):
    """
    Read existing YAML (or {}), deep-merge `patch`, then atomically write it back.
    """
    current = load_yaml(file_path)
    if not isinstance(current, dict):
        # If the root isn't a mapping, upgrade to a mapping
        current = {"_value": current}
    merged = _deep_merge(current, patch or {})
    _atomic_write_yaml(file_path, merged)

def load_pkl(file_path):
    """ Load a pickle file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        loaded data.
    """

    with open(file_path, 'rb') as f:
        file = pkl.load(f)

    return file

def save_pkl(file_path, obj):
    """Pickle ``obj`` to ``file_path``, creating parent directories.

    Parameters
    ----------
    file_path : str
        Destination path for the pickle file.
    obj : Any
        Object to serialize with ``pickle.HIGHEST_PROTOCOL``.
    """
    from os import makedirs
    from os.path import dirname
    from pickle import dump, HIGHEST_PROTOCOL

    parent = dirname(file_path)
    if parent:
        makedirs(parent, exist_ok=True)

    with open(file_path, "wb") as f:
        dump(obj, f, protocol=HIGHEST_PROTOCOL)

def create_info_file(out_path, info_dict, file_name='data_info.txt'):
    """ Saves info in *info_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save info file.
        info_dict: dict with all relevant info the user wants to save in the info file.
    """

    with open(os.path.join(out_path, file_name), 'w') as f:
        yaml.dump(info_dict, f)

def save_results_file(out_path, results_dict, file_name='retrain_results.txt'):
    """ Saves results in *results_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save results file.
        results_dict: dict with all relevant results the user wants to save in the results file.
    """

    with open(os.path.join(out_path, file_name), 'w') as f:
        json.dump(results_dict, f, indent=4)

def load_retrain_results(experiment_path, retrain_file_name='retrain_results_F13_multistep.txt'):
    """
    Load and identify the best retrained model from a JSON results file, then
    return the directory path, best model path, and its network definition.

    Args:
        experiment_path (str):
            The path to the experiment folder containing retraining results.
        retrain_file_name (str, optional):
            The name of the JSON file that stores retrain results
            (keys map to experiment runs, values include test metrics).
            Defaults to 'retrain_results_F13_multistep.txt'.

    Returns:
        dict:
            A dictionary with:
                - 'net': (list) the network layer definitions from the best run.
                - 'retrain_path': (str) the folder where the best retraining logs
                and files are stored.
                - 'best_model_path': (str) the file path to the best model checkpoint
                (`best_model.pth`) in the best retraining folder.

    Raises:
        FileNotFoundError:
            If the determined best retrain folder does not exist, or if the JSON results file
            or `retraining_params.txt` file are missing or unreadable.
    """
    file_path = os.path.join(experiment_path, retrain_file_name)
    with open(file_path, 'r') as f:
        retrain_data = json.load(f)

    # Determine the key with the highest test accuracy
    best_key = max(retrain_data, key=lambda x: retrain_data[x]['test_accuracy'])

    # Convert key naming (e.g., "multistep_F13_retrain_1" -> "retrain_F13_1")
    parts = best_key.split("_")
    best_key = f"{parts[2]}_{parts[1]}_{parts[3]}"

    # Construct path to the folder for the best retraining run
    retrain_path = os.path.join(experiment_path, best_key)
    if not os.path.exists(retrain_path):
        raise FileNotFoundError(f"Could not find the retrain folder at {retrain_path}")

    # Load retraining params (YAML) within the best retraining folder
    with open(os.path.join(retrain_path, 'retraining_params.txt'), 'r') as file:
        best_retrain_info = yaml.safe_load(file)

    net_list = best_retrain_info.get('net_list', [])

    # Build the path to the best model file
    best_model_path = os.path.join(retrain_path, 'best_model.pth')

    return {'net': net_list, 'retrain_path': retrain_path, 'best_model_path': best_model_path}


def load_log_params_evolution(experiment_path: str):
    """
    Loads the log parameters for the evolution process from the specified experiment path.

    Parameters:
    - experiment_path (str): The path to the experiment folder containing evolved data.

    Returns:
    dict: A dictionary containing the log parameters for the evolution process.

    This method reads the log parameters for the evolution process from the
    'log_params_evolution.txt' file. These typically include:
        - train_spec  (dict)
        - QNAS_spec   (dict)
        - fn_dict     (dict)
    among other possible keys like population size, generations, mutation rate, etc.
    """

    log_file = os.path.join(experiment_path, 'log_params_evolution.txt')
    if not os.path.isfile(log_file):
        raise FileNotFoundError(f"Could not find log_params_evolution.txt at {log_file}")

    with open(log_file, 'r') as file:
        log_params = yaml.safe_load(file)

    # Extract the subsets you need: train, QNAS, fn_dict
    train_spec = dict(log_params['train'])
    QNAS_spec = dict(log_params['QNAS'])
    fn_dict = log_params['fn_dict']

    # Return them together in a dictionary (you can rename or restructure as you prefer):
    return {
        'train_spec': train_spec,
        'QNAS_spec': QNAS_spec,
        'fn_dict': fn_dict
    }

def backup_cache(data, file_path: str = None) -> None:
    """
    Backup (update) the cache of evaluated individuals to a file.

    If the backup file exists, load its contents, update them with `data`,
    then write the merged dictionary back to the file. Otherwise, simply
    write `data` to the file.

    Args:
        data: dictionary containing evaluated individuals (e.g. self.evaluated).
        file_path: The path to the directory where the backup file is stored.
                If None, you can set a default path.
    """
    if file_path is None:
        file_path = os.getcwd()  # or some default directory
    file_name = os.path.join(file_path, "cache_backup.pkl")

    # Load existing cache if it exists
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            existing_data = load(f)
        # If both existing_data and data are dictionaries, update the existing data
        if isinstance(existing_data, dict) and isinstance(data, dict):
            existing_data.update(data)
            combined_data = existing_data
        else:
            combined_data = data
    else:
        combined_data = data

    with open(file_name, "wb") as f:
        dump(combined_data, f, protocol=HIGHEST_PROTOCOL)

def load_cache(file_path: str) -> Dict:
    """
    Load a cache backup from file into self.evaluated.

    Args:
        file_path: The path to the backup file.
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = load(f)
    else:
        print(f"Cache backup file {file_path} not found. Starting with empty cache.")
        data = {}
    return data

def load_pareto_history(filepath="pareto_history.pkl"):
    """
    Carga el archivo pickle que contiene fronts_history.
    Devuelve un dict: {generacion: {nivel_frente: [registros...]}}
    """
    with open(filepath, "rb") as f:
        history = pkl.load(f)
    return history

def load_history_from_json(file_path: str) -> dict:
    """Loads a history database from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            history_from_json = json.load(f)
        history_db = {tuple(map(int, k.split(','))): v for k, v in history_from_json.items()}
        print(f"Successfully loaded {len(history_db)} architectures from {file_path}")
        return history_db
    except FileNotFoundError:
        print(f"History file not found at {file_path}. Starting with an empty database.")
        return {}

def save_history_to_json(history: dict, file_path: str):
    """
    Saves the current state of the history database to a JSON file.

        history: dict: The history database to save.
        file_path: str: The path to the JSON file where the history will be saved.
    """
    # Convert tuple keys to comma-separated strings to make them JSON-compatible.
    history_for_json = {",".join(map(str, k)): v for k, v in history.items()}

    try:
        # Write to a temporary file first to prevent data corruption if the script crashes mid-write
        temp_file_path = file_path + ".tmp"
        with open(temp_file_path, 'w') as f:
            json.dump(history_for_json, f, indent=4)
        # If write is successful, rename the temporary file to the final name
        os.replace(temp_file_path, file_path)
    except IOError as e:
        print("Failed to save history file: %s", e)
