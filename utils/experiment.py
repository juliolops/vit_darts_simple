"""Experiment-folder management helpers for MoQ-NAS.

Natural sorting, experiment-path validation, old-run cleanup and
evolution-time estimation, extracted verbatim from ``utils/helpers.py``
(Block C of the refactor roadmap). ``helpers.py`` re-exports these names
until it becomes a facade (stage C.6), so both import paths are
equivalent.
"""
import os
import re
import shutil
import logging
from shutil import rmtree
from typing import List, Optional


def natural_key(string):
    """Sort key that orders strings by their embedded integers.

    Splits the string on digit runs and converts them to ``int``, so
    ``sort()`` compares numeric parts numerically instead of
    lexicographically.

    Parameters
    ----------
    string : str
        The string to derive the sort key from (e.g. a folder name like
        ``"1_10"``).

    Returns
    -------
    list
        Alternating non-digit substrings and integers.

    Examples
    --------
    >>> sorted(['1_10', '1_2', '1_1'], key=natural_key)
    ['1_1', '1_2', '1_10']
    """

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def check_file_exists(file_path):
    """ Check if a file exists.

    Args:
        file_path: (str) path to the file to check.

    Returns:
        True if the file exists, False otherwise.
    """
    if os.path.exists(file_path):
        return True
    else:
        return False

def delete_old_dirs(path, keep_best=False, best_id=''):
    """ Delete directories with old training files (models, checkpoints...). Assumes the
        directories' names start with digits.

    Args:
        path: (str) path to the experiment folder.
        keep_best: (bool) True if user wants to keep files from the best individual.
        best_id: (str) id of the best individual.
    """

    folders = [os.path.join(path, d) for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and d[0].isdigit()]
    folders.sort(key=natural_key)

    if keep_best and best_id:
        folders = [d for d in folders if os.path.basename(d) != best_id]

    for f in folders:
        rmtree(f)

def check_files(exp_path):
    """ Check if exp_path exists and if it does, check if log_file is valid.

    Args:
        exp_path: (str) path to the experiment folder.
    """
    if not os.path.exists(exp_path):
        raise OSError('User must provide a valid "--experiment_path" to continue '
                        'evolution or to retrain a model.')

    # 1. If there’s a symlink named "best_so_far", use its target
    best_link = os.path.join(exp_path, 'best_so_far')
    if os.path.islink(best_link):
        target = os.readlink(best_link)
        if os.path.isdir(target):
            best_result_folder = target
        else:
            raise ValueError(f'"best_so_far" symlink does not point to a directory: {target}')
    else:
        # 2. Otherwise, find subdirectories whose names start with a digit
        experiment_folders = [f.name for f in os.scandir(exp_path) if f.is_dir()]
        digit_folders = [name for name in experiment_folders if name and name[0].isdigit()]
        if not digit_folders:
            raise ValueError(f'No experiment folders starting with a digit found in: {exp_path}')

        # 3. Define numeric sort key (split on '_' and convert digit parts)
        def numeric_key(s):
            parts = s.split('_')
            return tuple(int(p) for p in parts if p.isdigit())

        best_name = min(digit_folders, key=numeric_key)
        best_result_folder = os.path.join(exp_path, best_name)

    # 4. Validate training_params.txt inside the chosen folder
    params_file = os.path.join(best_result_folder, 'training_params.txt')
    if not os.path.exists(params_file):
        raise OSError('training_params.txt not found!')
    if os.stat(params_file).st_size == 0:
        raise OSError('User must provide an "--experiment_path" with a valid data file to '
                        'continue evolution or to retrain a model.')

    # 5. Validate log_params_evolution.txt at the root of exp_path
    log_file = os.path.join(exp_path, 'log_params_evolution.txt')
    if not os.path.exists(log_file):
        raise OSError('log_params_evolution.txt not found!')
    if os.stat(log_file).st_size == 0:
        raise OSError('User must provide an "--experiment_path" with a valid config_file '
                        'to continue evolution or to retrain a model.')

    return best_result_folder

def calculate_time(start_time, elapse_time,current_gen:int=0, max_generations:int=300, end_evol = True):
    """
    Calculate the elapsed time and the estimated remaining time in the evolution process.

    Parameters:
    start_time (int): The start time of the evolution process.
    elapse_time (int): The current time in the evolution process.
    current_gen (int): The current generation number. Default is 0.
    max_generations (int): The maximum number of generations. Default is 300.
    end_evol (bool): If True, only calculate the elapsed time. If False, also calculate the estimated remaining time. Default is True.

    Returns:
    tuple: If end_evol is True, returns a tuple (hours, minutes) representing the elapsed time.
        If end_evol is False, returns a tuple (hours, minutes, remaining_total_hours, remaining_total_minutes) representing the elapsed time and the estimated remaining time.
    """

    total_time = elapse_time - start_time
    hours = int(total_time / 3600)
    minutes = int((total_time - hours * 3600) / 60)

    if end_evol:
        return hours, minutes
    else:
        avg_time_per_gen = total_time / current_gen if current_gen != 0 else 0
        remaining_total_time = avg_time_per_gen * (max_generations - current_gen)
        remaining_total_hours = int(remaining_total_time / 3600)
        remaining_total_minutes = int((remaining_total_time - remaining_total_hours * 3600) / 60)

        return hours, minutes, remaining_total_hours, remaining_total_minutes

def delete_old_dirs_v2(experiment_path: str,generation: int,keep_ids: List[str],is_snapshot_gen: bool = False,
                    results_subdir: str = "results",archive_subdir: str = "archive",snapshots_subdir: str = "snapshots",
                    link_name: str = "best_so_far",logger: Optional[logging.Logger] = None) -> None:
    """
    Manages experiment artifacts by moving results, taking snapshots,
    pruning the archive, and updating a symlink.

    If `is_snapshot_gen` is True, this function will also copy the current
    set of `keep_ids` to a permanent snapshot directory for that generation.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    base = experiment_path
    results_dir = os.path.join(base, results_subdir, f"gen_{generation}")
    archive_dir = os.path.join(base, archive_subdir)
    os.makedirs(archive_dir, exist_ok=True)

    # 1) Move any new results from the temporary 'results' dir to the 'archive' dir
    if os.path.isdir(results_dir):
        for kid in keep_ids:
            src = os.path.join(results_dir, kid)
            dst = os.path.join(archive_dir, kid)
            if os.path.isdir(src) and not os.path.isdir(dst):
                try:
                    shutil.move(src, dst)
                    logger.debug(f"Archived {src} -> {dst}")
                except Exception as e:
                    logger.warning(f"Error moving {src} -> {dst}: {e}")
        try:
            shutil.rmtree(results_dir)
            logger.debug(f"Removed temp dir {results_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temp dir {results_dir}: {e}")

    # --- NEW: Take a snapshot if this is a snapshot generation ---
    if is_snapshot_gen:
        snapshot_gen_dir = os.path.join(base, snapshots_subdir, f"gen_{generation}")
        os.makedirs(snapshot_gen_dir, exist_ok=True)
        logger.info(f"Taking snapshot for generation {generation} -> {snapshot_gen_dir}")

        for kid in keep_ids:
            src = os.path.join(archive_dir, kid)
            dst = os.path.join(snapshot_gen_dir, kid)
            if os.path.isdir(src) and not os.path.isdir(dst):
                try:
                    shutil.copytree(src, dst)
                    logger.debug(f"Copied {src} to snapshot {dst}")
                except Exception as e:
                    logger.warning(f"Error copying snapshot for {kid}: {e}")

    # 2) PRUNING: Always prune the main archive to keep it clean.
    #    This removes any models from 'archive/' that are no longer in the current Pareto front.
    for folder in os.listdir(archive_dir):
        if folder not in keep_ids:
            path = os.path.join(archive_dir, folder)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    logger.debug(f"Pruned old model from archive: {path}")
                except Exception as e:
                    logger.warning(f"Error pruning {path}: {e}")

    # 3) ACTUALIZAR SYMLINK al primer keep_id
    if not keep_ids:
        logger.error("keep_ids is empty, cannot link best_so_far")
        return

    best = keep_ids[0]
    target = os.path.join(archive_dir, best)
    linkpath = os.path.join(base, link_name)

    # Remove old link or file
    try:
        if os.path.islink(linkpath) or os.path.exists(linkpath):
            os.unlink(linkpath)
    except Exception as e:
        logger.warning(f"Could not remove old symlink {linkpath}: {e}")

    # Create new symlink
    try:
        # Check if target exists before creating a link to it
        if os.path.isdir(target):
            os.symlink(target, linkpath)
            logger.info(f"Updated {link_name} -> {target}")
        else:
            logger.warning(f"Target for symlink {target} does not exist. Cannot create link.")
    except Exception as e:
        logger.error(f"Error creating symlink {linkpath} -> {target}: {e}")
