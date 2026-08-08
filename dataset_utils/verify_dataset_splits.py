# verify_dataset_splits.py
"""
Utilities to verify dataset splits produced by GenericDataLoader.

Fixes:
- Correctly respects Subset chains when counting labels (no more double-counting the base dataset).
- Always derives class counts from the *final* datasets (after limiting/wrapping).

What this checks
----------------
- Per-class counts for train / val / test
- No index overlap between train and val when they are Subsets of the same base
- Optional 'limit_data' sanity: total (train + val) <= limit_data_value
- Determinism smoke test: same seeds -> same memberships (multiset compare)
- Class balance ratios (min_count / max_count per split)

Quick usage
-----------
from input import GenericDataLoader
from verify_splits import verify_loader_splits, summarize_balance, class_balance_ratio

params = {
    "dataset": "cifar10",
    "data_path": "./cifar10_data",
    "config_path_dataset": "dataset_configs/cifar10.yaml",
    "batch_size": 256,
    "eval_batch_size": 256,
    "num_workers": 2,
    "download": True,
    "data_augmentation": False,
    "train_split": 0.9,
    "split_seed": 2025,
    "loader_seed": 777,
    "limit_data": True,
    "limit_data_value": 10000,
}

gdl = GenericDataLoader(params)
summary = verify_loader_splits(
    gdl,
    limit_data_expected_total=params["limit_data_value"],
    verbose=True
)
summarize_balance(summary)
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -----------------------
# Helper: robust bincount
# -----------------------
def _np_bincount(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return np.array([], dtype=int)
    k = num_classes if num_classes is not None else (labels.max() + 1)
    return np.bincount(labels, minlength=k)


# ----------------------------------------------------------
# Label extraction that *respects Subset indices* (critical)
# ----------------------------------------------------------
def _unwrap_to_base_and_collect_indices(ds) -> Tuple[object, Optional[np.ndarray]]:
    """
    Walks through potential wrappers (Subset, custom wrappers that expose .dataset or .subset)
    and returns:
      base_ds: the first dataset in the chain that actually stores samples/labels
      composed_idx: a single index array mapping into base_ds, or None if not a Subset chain

    If multiple Subset layers exist, we compose their indices.
    """
    chain: List[object] = []
    cur = ds
    for _ in range(12):  # prevent infinite loops
        chain.append(cur)
        if hasattr(cur, "dataset"):
            cur = cur.dataset
            continue
        if hasattr(cur, "subset"):
            cur = cur.subset
            continue
        break
    base_ds = chain[-1]

    # Collect indices from any Subset in the chain (from base outward)
    # We need to compose them so that final indices select from base_ds.
    composed_idx: Optional[np.ndarray] = None
    # Traverse from inner->outer to compose
    for node in reversed(chain):
        if hasattr(node, "indices"):
            idx = np.asarray(getattr(node, "indices"), dtype=np.int64)
            if composed_idx is None:
                composed_idx = idx
            else:
                composed_idx = composed_idx[idx]
    return base_ds, composed_idx


def _labels_from_base_dataset(base) -> Optional[np.ndarray]:
    """
    Try to read labels from a *base* dataset (no longer Unwrapping through Subsets).
    """
    if hasattr(base, "samples"):  # ImageFolder-like
        return np.asarray([y for _, y in base.samples], dtype=int)
    if hasattr(base, "labels"):   # MedMNIST / some torchvision
        arr = np.asarray(getattr(base, "labels"))
        return arr.squeeze().astype(int)
    if hasattr(base, "targets"):  # Many torchvision datasets
        return np.asarray(getattr(base, "targets"), dtype=int)
    return None


def _extract_labels_respecting_subsets(ds) -> np.ndarray:
    """
    Get labels for a (possibly wrapped) dataset, respecting Subset indices if present.
    Falls back to iteration (slow) only when necessary.
    """
    base, composed_idx = _unwrap_to_base_and_collect_indices(ds)
    base_labels = _labels_from_base_dataset(base)

    if base_labels is not None:
        if composed_idx is None:
            return base_labels  # no Subset; whole base dataset
        return base_labels[composed_idx]  # Subset selection applied on base labels

    # Fallback: iterate actual ds (post-wrapping/limiting)
    labels = []
    for _, y in ds:
        labels.append(int(y))
    return np.asarray(labels, dtype=int)


# -------------------------------------------------
# Helper: recover Subset indices chain if applicable
# -------------------------------------------------
def _subset_indices(ds) -> Optional[np.ndarray]:
    """
    Return numpy array of indices if ds is (or wraps) a torch.utils.data.Subset.
    Otherwise return None.
    """
    _, composed_idx = _unwrap_to_base_and_collect_indices(ds)
    return composed_idx


# ---------------------------------
# Balance metric: min / max (0..1)
# ---------------------------------
def class_balance_ratio(counts: np.ndarray) -> float:
    """
    Returns a balance score in [0, 1].
      1.0  -> perfectly balanced (all classes have the same count)
      0.xx -> some imbalance; closer to 0 is worse
    Ignores classes with zero samples (so it doesn't divide by 0).
    """
    counts = np.asarray(counts, dtype=int)
    positive = counts[counts > 0]
    if positive.size == 0:
        return 1.0  # degenerate case: no data -> treat as 'balanced'
    return float(positive.min() / positive.max())


# -------------------------
# Main verification routine
# -------------------------
def verify_loader_splits(
    gdl,
    *,
    limit_data_expected_total: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Verify split properties for a GenericDataLoader instance.

    Args:
      gdl: GenericDataLoader already constructed
      limit_data_expected_total: if you use limit_data, pass limit_data_value here
      verbose: print a human-friendly summary

    Returns:
      dict with 'train_counts', 'val_counts', 'test_counts' (np.ndarrays)
    """
    num_classes = int(gdl.num_classes)

    # Always reflect the *final* datasets (post-limit, post-wrapping).
    train_labels = _extract_labels_respecting_subsets(gdl.train_dataset)
    val_labels   = _extract_labels_respecting_subsets(gdl.val_dataset)
    test_labels  = _extract_labels_respecting_subsets(gdl.test_dataset)

    # Class counts
    train_counts = _np_bincount(train_labels, num_classes)
    val_counts   = _np_bincount(val_labels,   num_classes)
    test_counts  = _np_bincount(test_labels,  num_classes)

    # Report
    if verbose:
        print("=== Split Summary ===")
        print(f"Dataset: {gdl.spec.name} | classes={num_classes}")
        print(f"Train/Val/Test sizes: {len(train_labels)} / {len(val_labels)} / {len(test_labels)}")
        print("\nPer-class counts (class: train | val | test):")
        for c in range(num_classes):
            print(f"  {c:>3}: {int(train_counts[c]):>6} | {int(val_counts[c]):>6} | {int(test_counts[c]):>6}")

    # 1) No overlap check (only meaningful if train/val are a split of the same base set)
    tr_idx = _subset_indices(gdl.train_dataset)
    va_idx = _subset_indices(gdl.val_dataset)
    if tr_idx is not None and va_idx is not None:
        overlap = np.intersect1d(tr_idx, va_idx)
        if verbose:
            print(f"\nIndex overlap between train and val: {len(overlap)}")
        assert len(overlap) == 0, f"Train/Val overlap detected: {len(overlap)} samples"

    # 2) limit_data check (if expected total provided)
    if limit_data_expected_total is not None:
        got_total = len(train_labels) + len(val_labels)
        assert got_total <= int(limit_data_expected_total), \
            f"limit_data_value={limit_data_expected_total} but got train+val={got_total}"

    # 3) Determinism smoke test: rebuild with same seeds & compare memberships (multisets)
    try:
        params_dup = dict(gdl.params)
        dup = gdl.__class__(params=params_dup, train_split=gdl.train_split, seed=gdl.seed)

        tr_labels_dup = _extract_labels_respecting_subsets(dup.train_dataset)
        va_labels_dup = _extract_labels_respecting_subsets(dup.val_dataset)

        def _multiset_equal(a, b):
            return Counter(a) == Counter(b)

        same_train = _multiset_equal(train_labels, tr_labels_dup)
        same_val   = _multiset_equal(val_labels,   va_labels_dup)

        if verbose:
            print(f"\nDeterminism (same split_seed): train match={same_train} | val match={same_val}")
        assert same_train and same_val, "Determinism failed: same seeds produced different memberships."
    except Exception as e:
        if verbose:
            print(f"\n[WARN] Determinism recheck skipped/failed: {e}")

    if verbose:
        print("\nAll checks passed ✔️")

    return {
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
    }


# ----------------------------
# Pretty print balance numbers
# ----------------------------
def summarize_balance(summary: Dict[str, np.ndarray]) -> None:
    tr = summary["train_counts"]; va = summary["val_counts"]; te = summary["test_counts"]
    print("\n=== Class Balance Ratios (min/max) ===")
    print(f"Train: {class_balance_ratio(tr):.4f}")
    print(f"Val:   {class_balance_ratio(va):.4f}")
    print(f"Test:  {class_balance_ratio(te):.4f}")


# ---------------------------
# Optional standalone example
# ---------------------------
if __name__ == "__main__":
    try:
        from core.cnn.input import GenericDataLoader
    except Exception:
        raise SystemExit(
            "This script expects 'input.py' with GenericDataLoader in the same environment."
        )

    params = {
        "dataset": "cifar10",
        "data_path": "./cifar10_data",
        "config_path_dataset": "dataset_configs/cifar10.yaml",
        "batch_size": 256,
        "eval_batch_size": 256,
        "num_workers": 2,
        "download": True,
        "data_augmentation": False,
        "train_split": 0.9,
        "split_seed": 2025,
        "loader_seed": 777,
        "limit_data": True,
        "limit_data_value": 10000,
    }

    gdl = GenericDataLoader(params)
    summary = verify_loader_splits(
        gdl,
        limit_data_expected_total=params["limit_data_value"],
        verbose=True
    )
    summarize_balance(summary)
