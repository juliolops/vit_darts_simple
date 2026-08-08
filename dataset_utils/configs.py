# dataset_utils/configs.py
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os, yaml

@dataclass
class DatasetSpec:
    name: str
    family: str                    # "torchvision" | "medmnist" | "custom_local"
    shape: Tuple[int,int,int]      # (C,H,W)
    num_classes: int
    mean: Optional[List[float]] = None
    std:  Optional[List[float]] = None
    task: str = "classification"   # or "multilabel" | "regression"
    stats_mode: str = "known"      # "known" | "sample" | "none"

def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dataset_spec(params: dict) -> Optional[DatasetSpec]:
    """
    Optional helper: if params['config_path'] exists, load DatasetSpec from YAML.
    Otherwise return None (we'll fall back to your current hardcoded dicts).
    """
    cfg_path = params.get("config_path_dataset")
    if not cfg_path:
        return None
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = load_yaml_config(cfg_path)
    # Minimal validation
    required = ["name","family","shape","num_classes","task"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing fields in {cfg_path}: {missing}")
    return DatasetSpec(
        name=cfg["name"],
        family=cfg["family"],
        shape=tuple(cfg["shape"]),
        num_classes=int(cfg["num_classes"]),
        mean=cfg.get("mean"),
        std=cfg.get("std"),
        task=cfg.get("task","classification"),
        stats_mode=cfg.get("stats_mode","known"),
    )