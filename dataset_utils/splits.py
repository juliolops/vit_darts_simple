# dataset_utils/splits.py
# Utility functions for dataset splitting
import numpy as np
from typing import Tuple, Sequence
from sklearn.model_selection import StratifiedShuffleSplit

def deterministic_stratified_indices(
    labels: np.ndarray, train_size: float, split_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=split_seed)
    train_idx, val_idx = next(sss.split(indices, labels))
    return train_idx.astype(np.int64), val_idx.astype(np.int64)