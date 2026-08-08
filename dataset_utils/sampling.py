# dataset_utils/sampling.py
import numpy as np
from torch.utils.data import Subset
from typing import Sequence

def class_balanced_indices(labels: Sequence[int], total_samples: int, num_classes: int, rng) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    per_cls = max(1, total_samples // num_classes)
    idx_by_cls = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    chosen = []
    for c in sorted(idx_by_cls):
        arr = idx_by_cls[c].copy()
        rng.shuffle(arr)             # <-- deterministic shuffle via provided RNG
        chosen.extend(arr[:per_cls].tolist())
    return np.asarray(chosen, dtype=int)

def apply_limit(dataset, labels, k, num_classes, rng):
    idx = class_balanced_indices(labels, k, num_classes, rng)
    return Subset(dataset, idx)
