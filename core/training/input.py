# input.py
"""
Generic data loader for PyTorch, supporting various datasets and data augmentation.

THIS VERSION RELIES **ONLY** ON YAML CONFIGS.
- Expects params["config_path_dataset"] to point to a YAML file in dataset_configs/.
- Uses dataset_utils modules for transforms and dataset construction.
"""

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from utils.helpers import create_info_file, check_file_exists
import torchvision.datasets  # imported to ensure availability for factory

# Local helpers you created
from dataset_utils.configs import DatasetSpec, load_dataset_spec
from dataset_utils.transformations import build_transforms
from dataset_utils.factory import build_datasets


# -------------------------
# Utilities
# -------------------------
def _make_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _supports_pin_memory_device() -> bool:
    try:
        from inspect import signature
        return "pin_memory_device" in signature(DataLoader).parameters
    except Exception:
        return False


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


def _compute_sampled_mean_std(dataset, max_batches: int = 10, batch_size: int = 256) -> Tuple[list, list]:
    """
    Compute mean/std over a capped number of batches (memory/time safe).
    Assumes dataset returns (C,H,W) tensors, or PIL with ToTensor() applied by caller.
    """
    from torch.utils.data import DataLoader as _DL
    loader = _DL(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    n_seen = 0
    mean = None
    M2 = None
    batches = 0
    for x, _ in loader:
        x = x.float()
        b, c, h, w = x.shape
        x = x.view(b, c, -1)  # (b, c, hw)
        batch_mean = x.mean(dim=(0, 2))  # (c,)
        batch_var = x.var(dim=(0, 2), unbiased=False)  # (c,)

        if mean is None:
            mean = batch_mean
            M2 = batch_var
            n_seen = 1
        else:
            n_seen += 1
            mean = mean + (batch_mean - mean) / n_seen
            M2 = M2 + (batch_var - M2) / n_seen

        batches += 1
        if batches >= max_batches:
            break

    if mean is None:
        raise RuntimeError("Unable to compute sampled mean/std: dataset appears empty.")
    std = torch.sqrt(M2 + 1e-8)
    return mean.tolist(), std.tolist()


# -------------------------
# Main loader (YAML-only)
# -------------------------
class GenericDataLoader:
    """A generic data loader for PyTorch, supporting various datasets and data augmentation."""

    def __init__(self, params: dict, train_split=0.9, seed: Optional[int] = None, info: dict = {}):
        """
        Initialize the GenericDataLoader.

        Parameters:
            params (dict): must include:
                - dataset: dataset name (e.g., "cifar10")
                - data_path: root data folder
                - config_path_dataset: path to YAML config describing this dataset
            train_split (float): Split ratio for training data (torchvision family).
            seed (int): Fallback seed when the config lacks split_seed/loader_seed.
            info (dict): Unused (kept for API compatibility).
        """
        self.params = params
        self.train_split = float(train_split)
        assert 0.0 <= self.train_split <= 1.0, "[!] train_split should be in the range [0, 1]."

        # Enforce YAML-only workflow
        if "config_path_dataset" not in self.params or not os.path.isfile(self.params["config_path_dataset"]):
            raise FileNotFoundError(
                "config_path_dataset is required and must point to a valid YAML file. "
                "Place your file under dataset_configs/ and set params['config_path_dataset'] accordingly."
            )

        # Fallback seed for split/loader when not given in the config.
        # NOTE: must NOT touch the global RNGs (random/torch) here — doing so
        # used to reseed them with time() and silently break --seed runs.
        # All loader randomness goes through local generators (_make_gen)
        # seeded from split_seed/loader_seed (or this fallback).
        self.seed = int(params.get("seed", 42)) if seed is None else seed

        # Paths & policy
        self.data_path = self.params["data_path"]
        os.makedirs(self.data_path, exist_ok=True)
        self.download = _coerce_bool(self.params.get("download", True))

        # Load DatasetSpec strictly from YAML
        spec_from_yaml = load_dataset_spec(self.params)
        if spec_from_yaml is None:
            # load_dataset_spec should not return None given the check above,
            # but keep a defensive error in case someone overrides it.
            raise RuntimeError("Failed to load dataset spec from YAML; check your config file.")

        self.ds_name = spec_from_yaml.name.lower()

        # If stats missing and YAML indicates sampling, compute sampled stats (cap batches)
        mean, std = spec_from_yaml.mean, spec_from_yaml.std
        if (mean is None or std is None) and spec_from_yaml.stats_mode == "sample":
            if spec_from_yaml.family == "torchvision":
                ds_cls = getattr(torchvision.datasets, self.params["dataset"].upper())
                tmp_ds = ds_cls(self.data_path, train=True, download=self.download, transform=ToTensor())
            else:
                raise RuntimeError(f"No mean/std for {spec_from_yaml.name} and stats_mode={spec_from_yaml.stats_mode}")
            mean, std = _compute_sampled_mean_std(tmp_ds, max_batches=int(self.params.get("stats_max_batches", 10)))

        # Finalize spec
        self.num_classes = int(spec_from_yaml.num_classes)
        channels, height, width = spec_from_yaml.shape
        self.spec = DatasetSpec(
            name=spec_from_yaml.name,
            family=spec_from_yaml.family,
            shape=(int(channels), int(height), int(width)),
            num_classes=self.num_classes,
            mean=mean,
            std=std,
            task=spec_from_yaml.task,
            stats_mode="known",  # finalized now
        )

        # Transforms (centralized)
        self.train_transform, self.eval_transform = build_transforms(
            self.spec, _coerce_bool(self.params.get("data_augmentation", False))
        )

        # Persist info once
        self.info_dict = {
            "dataset": self.params["dataset"],
            "seed": self.params.get("split_seed", self.seed),
            "shape": [channels, height, width],
            "mean": mean,
            "std": std,
            "num_classes": self.num_classes,
            "task": self.spec.task,
        }
        info_path = os.path.join(self.data_path, "data_info.txt")
        if not check_file_exists(info_path):
            create_info_file(out_path=self.data_path, info_dict=self.info_dict)

        # Build datasets ONCE via factory (split + optional limit are handled inside)
        self.train_dataset, self.val_dataset, self.test_dataset, self._train_labels, self._val_labels = build_datasets(
            spec=self.spec,
            params=self.params,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
            data_path=self.data_path,
            download=self.download,
            train_split=float(self.params.get("train_split", self.train_split)),
            split_seed=int(self.params.get("split_seed", self.seed)),
        )

    def get_loader(self, for_train: bool = True, pin_memory_device: Optional[str] = "cuda"):
        """
        Get data loader for training or validation/testing.

        Parameters:
            for_train (bool): If True, returns (train_loader, val_loader); otherwise, returns test_loader.
        """
        # Deterministic DataLoader shuffling
        g = _make_gen(int(self.params.get("loader_seed", self.seed)))
        # pin_memory only helps (and is only supported) for CUDA host->device
        # transfers; on MPS/CPU it's a no-op that just prints a warning, so
        # skip it there instead of asking PyTorch to ignore it every batch.
        use_pinned = bool(pin_memory_device) and str(pin_memory_device).startswith("cuda")
        common = dict(num_workers=int(self.params.get("num_workers", 4)), pin_memory=use_pinned, generator=g)
        if use_pinned and _supports_pin_memory_device():
            common["pin_memory_device"] = pin_memory_device

        drop_last = False

        if not for_train:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=int(self.params["eval_batch_size"]),
                shuffle=False,
                drop_last=False,
                **common,
            )
            self.info_dict["test_records"] = len(self.test_dataset)
            create_info_file(out_path=self.data_path, info_dict=self.info_dict)
            return test_loader

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(self.params["batch_size"]),
            shuffle=True,
            drop_last=drop_last,
            **common,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=int(self.params["eval_batch_size"]),
            shuffle=False,
            drop_last=False,
            **common,
        )

        # Update counts & persist
        self.info_dict["train_records"] = len(self.train_dataset)
        self.info_dict["valid_records"] = len(self.val_dataset)
        self.info_dict["test_records"] = len(self.test_dataset)
        create_info_file(out_path=self.data_path, info_dict=self.info_dict)

        return train_loader, val_loader
