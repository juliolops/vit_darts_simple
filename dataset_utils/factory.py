# dataset_utils/factory.py
from __future__ import annotations
import os
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torchvision.datasets as tvd
from sklearn.model_selection import StratifiedShuffleSplit

import medmnist
from medmnist import INFO

from .sampling import apply_limit
from .splits import deterministic_stratified_indices

def _get_labels_from_torchvision_dataset(ds) -> np.ndarray:
    labels = getattr(ds, "targets", None)
    if labels is None:
        labels = getattr(ds, "labels", None)
    if labels is None:
        raise RuntimeError("Cannot find labels for torchvision dataset (no .targets/.labels).")
    return np.asarray(labels).astype(int)



def _maybe_apply_limit_balanced(
    train_ds, val_ds, train_labels, val_labels, num_classes: int,
    train_size: float, split_seed: int, limit_data: bool, limit_value: int
):
    if not limit_data:
        return train_ds, val_ds
    total_current = len(train_ds) + len(val_ds)
    if limit_value <= 0 or limit_value >= total_current:
        # nothing to do
        return train_ds, val_ds

    k_total = int(limit_value)
    k_train = max(1, int(round(k_total * train_size)))
    k_val   = max(1, k_total - k_train)

    rng = np.random.default_rng(int(split_seed))

    # Safety clamps
    k_train = min(k_train, len(train_ds))
    k_val   = min(k_val,   len(val_ds))

    train_ds = apply_limit(train_ds, train_labels, k_train, num_classes, rng)
    val_ds   = apply_limit(val_ds,   val_labels,   k_val,   num_classes, rng)
    return train_ds, val_ds


def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in ("true", "1", "yes", "y", "t"):
            return True
        if xl in ("false", "0", "no", "n", "f"):
            return False
    return bool(x)


class _TransformWrapper(Dataset):
    """Apply a transform lazily to the samples of a wrapped subset.

    Parameters
    ----------
    subset : torch.utils.data.Dataset
        Underlying dataset/subset yielding ``(x, y)`` pairs.
    tfm : callable or None
        Transform applied to ``x`` on access; if None, ``x`` passes through.

    Notes
    -----
    ``__getitem__`` returns ``(tfm(x), y)``. Consolidates the two identical
    ``_Wrap`` inner classes previously defined inside ``build_datasets``.
    """
    def __init__(self, subset, tfm):
        self.subset = subset; self.tfm = tfm
    def __len__(self): return len(self.subset)
    def __getitem__(self, i):
        x, y = self.subset[i]
        return (self.tfm(x) if self.tfm else x), y


def build_datasets(
    *,
    spec,                      # DatasetSpec (name, family, num_classes, shape, ...)
    params: dict,              # your params dict
    train_transform,           # transform for train
    eval_transform,            # transform for val/test
    data_path: str,            # root path
    download: bool,            # allow download if needed
    train_split: float,        # fraction for train
    split_seed: int            # seed for deterministic split / limit
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, np.ndarray, np.ndarray]:
    """
    Returns: (train_dataset, val_dataset, test_dataset, train_labels, val_labels)
    Labels arrays are aligned with the returned (possibly limited) train/val datasets.
    """
    ds_name = spec.name.lower()

    # ---- Torchvision family: build once, split deterministically, then wrap transforms
    if hasattr(tvd, ds_name.upper()):
        trainval_raw = getattr(tvd, ds_name.upper())(data_path, train=True, download=download, transform=None)
        test_dataset = getattr(tvd, ds_name.upper())(data_path, train=False, download=download, transform=eval_transform)

        labels_full = _get_labels_from_torchvision_dataset(trainval_raw)
        train_idx, val_idx = deterministic_stratified_indices(labels_full, train_split, split_seed)

        train_subset = Subset(trainval_raw, train_idx)
        val_subset   = Subset(trainval_raw, val_idx)

        # Build label arrays that align with subsets (for limit_data)
        train_labels = labels_full[train_idx]
        val_labels   = labels_full[val_idx]

        limit_flag  = _coerce_bool(params.get("limit_data", False))
        limit_value = int(params.get("limit_data_value", 0) or 0)
        
        # Deterministic balanced limit (optional)
        train_subset, val_subset = _maybe_apply_limit_balanced(
            train_subset, val_subset, train_labels, val_labels,
            spec.num_classes, train_split, split_seed,
            limit_data=limit_flag,
            limit_value=limit_value,
        )

        # Finally inject transforms via small wrapper
        train_dataset = _TransformWrapper(train_subset, train_transform)
        val_dataset   = _TransformWrapper(val_subset,   eval_transform)

        # Recompute labels if limiting changed the order/size
        # (apply_limit keeps sample order subset; labels remain consistent)
        if isinstance(train_dataset.subset, Subset):
            # no change needed; labels arrays already aligned
            pass

        return train_dataset, val_dataset, test_dataset, train_labels, val_labels

    # ---- MedMNIST family: train/val/test provided by dataset; only limit (optional)
    if ds_name in INFO and hasattr(medmnist, INFO[ds_name]["python_class"]):
        med_cls = getattr(medmnist, INFO[ds_name]["python_class"])
        train_dataset = med_cls(root=data_path, split="train", download=download, transform=train_transform, as_rgb=True)
        val_dataset   = med_cls(root=data_path, split="val",   download=download, transform=eval_transform, as_rgb=True)
        test_dataset  = med_cls(root=data_path, split="test",  download=download, transform=eval_transform, as_rgb=True)

        train_labels = train_dataset.labels.squeeze().astype(int)
        val_labels   = val_dataset.labels.squeeze().astype(int)

        limit_flag  = _coerce_bool(params.get("limit_data", False))
        limit_value = int(params.get("limit_data_value", 0) or 0)
        
        train_dataset, val_dataset = _maybe_apply_limit_balanced(
            train_dataset, val_dataset, train_labels, val_labels,
            spec.num_classes, train_split, split_seed,
            limit_data=limit_flag,
            limit_value=limit_value,
        )
        return train_dataset, val_dataset, test_dataset, np.asarray(train_labels), np.asarray(val_labels)

    # ---- ATLETA (ImageFolder layout)
    if "atleta" in ds_name:
        try:
            train_dataset = tvd.ImageFolder(root=os.path.join(data_path, "train"), transform=train_transform)
            val_dataset   = tvd.ImageFolder(root=os.path.join(data_path, "val"),   transform=eval_transform)
            test_dataset  = tvd.ImageFolder(root=os.path.join(data_path, "test"),  transform=eval_transform)
        except Exception as e:
            raise ValueError(f"ATLETA dataset path is invalid or incomplete: {data_path} ({e})")

        # Labels for limiting
        train_labels = np.asarray([y for _, y in train_dataset.samples], dtype=int)
        val_labels   = np.asarray([y for _, y in val_dataset.samples], dtype=int)

        limit_flag  = _coerce_bool(params.get("limit_data", False))
        limit_value = int(params.get("limit_data_value", 0) or 0)
        
        train_dataset, val_dataset = _maybe_apply_limit_balanced(
            train_dataset, val_dataset, train_labels, val_labels,
            spec.num_classes, train_split, split_seed,
            limit_data=limit_flag,
            limit_value=limit_value,
        )
        return train_dataset, val_dataset, test_dataset, train_labels, val_labels

    # ---- Person/Face Binary Datasets (ImageFolder layout with missing test set handling)
    if "person" in ds_name or "face" in ds_name:
        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        
        # 1. Use the validation set as the test set
        test_dataset = tvd.ImageFolder(root=val_path, transform=eval_transform)
        
        # 2. Load the original training data to prepare for splitting
        trainval_raw = tvd.ImageFolder(root=train_path, transform=None)
        
        class_to_idx = trainval_raw.class_to_idx
        positive_class_name = 'person' if 'person' in ds_name else 'face'
        
        if positive_class_name in class_to_idx:
            positive_idx = class_to_idx[positive_class_name]
            params['positive_class_idx'] = positive_idx
        else:
            params['positive_class_idx'] = 1
        
        labels_full = np.asarray([y for _, y in trainval_raw.samples], dtype=int)
        
        # 3. Create new train/val split from the original training data
        train_idx, val_idx = deterministic_stratified_indices(labels_full, train_split, split_seed)
        train_subset = Subset(trainval_raw, train_idx)
        val_subset   = Subset(trainval_raw, val_idx)
        
        train_labels = labels_full[train_idx]
        val_labels   = labels_full[val_idx]

        # 4. Apply data limiting (if configured)
        limit_flag  = _coerce_bool(params.get("limit_data", False))
        limit_value = int(params.get("limit_data_value", 0) or 0)
        train_subset, val_subset = _maybe_apply_limit_balanced(
            train_subset, val_subset, train_labels, val_labels,
            spec.num_classes, train_split, split_seed,
            limit_data=limit_flag,
            limit_value=limit_value,
        )
        
        # 5. Wrap subsets with the appropriate transforms
        train_dataset = _TransformWrapper(train_subset, train_transform)
        val_dataset   = _TransformWrapper(val_subset,   eval_transform)

        return train_dataset, val_dataset, test_dataset, train_labels, val_labels

    # ---- Unknown family
    raise NotImplementedError(f"Unknown dataset family for {spec.name}.")