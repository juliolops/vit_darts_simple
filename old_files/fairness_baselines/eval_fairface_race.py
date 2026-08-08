#!/usr/bin/env python
# fairness/eval_fairface_race.py
"""
Evaluate FACE vs NON-FACE classifiers on FairFace by race, compute fairness, and save JSON.

✅ What’s new (vs. your previous script):
- Saves results to a JSON file (single checkpoint OR a whole directory).
- Auto-detects the backbone architecture from the checkpoint name *or* by trying the candidates.
- Splits fairness math into a reusable function `compute_fairness_metrics(...)`.
- Safer checkpoint loading (prefers weights_only=True when available).
- Works with the torchvision-only baselines you’re training.

Usage examples
--------------
# 1) Evaluate a single checkpoint
python fairness/eval_fairface_race.py \
    --fairface_csv data/FairFace/0.25/fairface_val.csv \
    --ckpt checkpoints/facebin_resnet18.pt \
    --out_json fairness/results_resnet18.json

# 2) Evaluate a whole folder (auto-detects models like facebin_resnet50.pt, ...):
python fairness/eval_fairface_race.py \
    --fairface_csv data/FairFace/0.25/fairface_val.csv \
    --ckpt_dir checkpoints \
    --out_json fairness/results_all.json

# 3) Quick dry-run on first N images to sanity-check the pipeline
python fairness/eval_fairface_race.py \
    --fairface_csv data/FairFace/0.25/fairface_val.csv \
    --ckpt checkpoints/facebin_resnet18.pt \
    --limit 200 \
    --out_json fairness/results_resnet18_200.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms as T
from PIL import Image
import pandas as pd

# ------------------------
# Torch AMP imports (safe)
# ------------------------
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ------------------------
# Registry of torchvision models we support
# (matches your training baselines)
# ------------------------
def _set_fc(m, nc): m.fc = nn.Linear(m.fc.in_features, nc)
def _set_cls(m, nc):
    # Replace the last Linear layer in a Sequential classifier
    idx = None
    for i in reversed(range(len(m.classifier))):
        if isinstance(m.classifier[i], nn.Linear):
            idx = i; break
    if idx is None:
        raise RuntimeError("Classifier tail has no Linear layer; adjust for this model.")
    in_f = m.classifier[idx].in_features
    m.classifier[idx] = nn.Linear(in_f, nc)

def _set_convnext_head(m, nc):
    # ConvNeXt classifier: [LayerNorm, Flatten, Linear]
    lin = m.classifier[-1]
    m.classifier[-1] = nn.Linear(lin.in_features, nc)

ARCH_REGISTRY = {}
try: ARCH_REGISTRY["resnet18"]         = (models.resnet18,        models.ResNet18_Weights.IMAGENET1K_V1,         _set_fc)
except Exception: pass
try: ARCH_REGISTRY["resnet50"]         = (models.resnet50,        models.ResNet50_Weights.IMAGENET1K_V2,         _set_fc)
except Exception: pass
try: ARCH_REGISTRY["mobilenet_v3_large"]= (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, _set_cls)
except Exception: pass
try: ARCH_REGISTRY["efficientnet_v2_s"]= (models.efficientnet_v2_s,models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, _set_cls)
except Exception: pass
try: ARCH_REGISTRY["convnext_tiny"]    = (models.convnext_tiny,   models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,    _set_convnext_head)
except Exception: pass


# ------------------------
# Helpers
# ------------------------
def _safe_load_checkpoint(path: str):
    """Load a checkpoint safely. Prefer weights_only=True when available."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location="cpu")

def _extract_state_dict(obj):
    """Extract a state_dict from various checkpoint formats."""
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        else:
            sd = obj
    else:
        raise ValueError("Unsupported checkpoint object type (expected dict/state_dict).")
    # strip possible 'module.' prefixes (DDP)
    return {k.replace("module.", ""): v for k, v in sd.items()}

def _get_weights_meta(weights) -> Tuple[List[float], List[float]]:
    """Return (mean, std) robustly across torchvision versions."""
    # Newer torchvision: weights.meta dict with 'mean'/'std'
    try:
        meta = getattr(weights, "meta", None)
        if isinstance(meta, dict):
            mean = meta.get("mean", IMAGENET_MEAN)
            std  = meta.get("std", IMAGENET_STD)
            return list(mean), list(std)
    except Exception:
        pass
    # Fallback: try weights.transforms() for a Normalize layer
    try:
        tf = weights.transforms()
        for t in getattr(tf, "transforms", []):
            if isinstance(t, T.Normalize):
                return list(t.mean), list(t.std)
    except Exception:
        pass
    return IMAGENET_MEAN, IMAGENET_STD

def _build_model_and_transform(arch: str, num_classes: int = 2, img_size: int = 224):
    """Instantiate model + eval transform (Resize+CenterCrop) for a given arch."""
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Available: {list(ARCH_REGISTRY.keys())}")
    ctor, weights_enum, head_setter = ARCH_REGISTRY[arch]

    # Try to instantiate with weights for correct normalization; fallback gracefully
    try:
        weights = getattr(weights_enum, "DEFAULT", weights_enum)
        model = ctor(weights=weights)
        mean, std = _get_weights_meta(weights)
    except Exception:
        try:
            model = ctor(pretrained=True)  # old API
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        except Exception:
            model = ctor()
            mean, std = IMAGENET_MEAN, IMAGENET_STD

    head_setter(model, num_classes)

    eval_resize = 256 if img_size == 224 else int(img_size * 256 / 224)
    tf_eval = T.Compose([
        T.Resize(eval_resize),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return model, tf_eval

def _infer_arch_from_filename(ckpt_path: str) -> Optional[str]:
    """Guess arch from checkpoint filename like facebin_resnet50.pt"""
    name = Path(ckpt_path).name.lower()
    for arch in ARCH_REGISTRY.keys():
        if arch in name:
            return arch
    return None

def _auto_select_arch_for_state_dict(state_dict) -> str:
    """Try loading the state_dict into each candidate arch and pick the best match."""
    scores = []
    for arch in ARCH_REGISTRY.keys():
        try:
            m, _ = _build_model_and_transform(arch, num_classes=2)
            missing, unexpected = m.load_state_dict(state_dict, strict=False)
            score = len(missing) + len(unexpected)
            scores.append((score, arch))
        except Exception:
            continue
    if not scores:
        raise RuntimeError("Could not match checkpoint to any supported architecture.")
    scores.sort(key=lambda x: x[0])
    return scores[0][0] == float("inf") and scores[0][1] or scores[0][1]

def load_model_auto(ckpt_path: str) -> Tuple[torch.nn.Module, T.Compose, str]:
    """Load a model by auto-detecting the architecture from filename or by trial."""
    obj = _safe_load_checkpoint(ckpt_path)
    sd  = _extract_state_dict(obj)

    arch = _infer_arch_from_filename(ckpt_path)
    if arch is None:
        arch = _auto_select_arch_for_state_dict(sd)

    model, tf_eval = _build_model_and_transform(arch, num_classes=2)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[info] load_state_dict for {arch}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, tf_eval, arch


# ------------------------
# Fairness evaluation
# ------------------------
def compute_fairness_metrics(per_group_tpr: Dict[str, float], beta: float = 0.2) -> Dict[str, float]:
    """
    Compute fairness metrics given per-group true positive rates.
    Returns:
        - min_group_tpr
        - max_min_gap    (max(per_group) - min(per_group))
        - spd_sum        (sum of deviations from the worst group)
        - fairness_score max(0, (beta - spd_sum)/beta)
    """
    if not per_group_tpr:
        return {"min_group_tpr": 0.0, "max_min_gap": 0.0, "spd_sum": 0.0, "fairness_score": 0.0}

    vals = list(per_group_tpr.values())
    acc_min = min(vals)
    acc_max = max(vals)
    spd_sum = sum((v - acc_min) for v in vals)
    fairness = max(0.0, (beta - spd_sum) / beta)
    return {
        "min_group_tpr": float(acc_min),
        "max_min_gap": float(acc_max - acc_min),
        "spd_sum": float(spd_sum),
        "fairness_score": float(fairness),
    }


# ------------------------
# Core evaluation (inference loop)
# ------------------------
def evaluate_on_fairface(
    model: torch.nn.Module,
    tf_eval: T.Compose,
    fairface_csv: str,
    device: torch.device,
    batch_size: int = 128,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, int], int]:
    """
    Run inference on FairFace (all positives) and compute per-group TPRs.
    Returns:
        per_group_tpr, per_group_counts, total_samples
    """
    df = pd.read_csv(fairface_csv)
    if limit is not None and limit > 0:
        df = df.head(limit)

    groups = sorted(df["race"].unique().tolist())
    correct = {g: 0 for g in groups}
    total   = {g: 0 for g in groups}

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(enabled=(amp_device == "cuda"))  # not strictly needed for eval

    # Batch up manually
    xs, gs = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
            except Exception:
                continue
            xs.append(tf_eval(img))
            gs.append(row["race"])
            if len(xs) == batch_size:
                x = torch.stack(xs).to(device)
                with autocast(device_type=amp_device, enabled=(amp_device == "cuda")):
                    logits = model(x)
                preds = logits.argmax(1).cpu().tolist()
                for p, g in zip(preds, gs):
                    # Ground truth is face (1) for all rows
                    correct[g] += int(p == 1)
                    total[g] += 1
                xs, gs = [], []

        if xs:
            x = torch.stack(xs).to(device)
            with autocast(device_type=amp_device, enabled=(amp_device == "cuda")):
                logits = model(x)
            preds = logits.argmax(1).cpu().tolist()
            for p, g in zip(preds, gs):
                correct[g] += int(p == 1)
                total[g] += 1

    per_group_tpr = {g: (correct[g] / total[g] if total[g] > 0 else 0.0) for g in groups}
    n_total = int(sum(total.values()))
    return per_group_tpr, total, n_total


# ------------------------
# JSON writer
# ------------------------
def write_results_json(
    out_path: str,
    meta: Dict,
    entries: List[Dict],
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": meta,
        "results": entries,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n💾 Saved JSON → {out_path}")


# ------------------------
# CLI
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fairface_csv", required=True, help="CSV with columns: image_path,race")
    ap.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path (.pt)")
    ap.add_argument("--ckpt_dir", type=str, default=None, help="Directory with checkpoints to evaluate")
    ap.add_argument("--out_json", type=str, default="fairness/results.json", help="Where to save the JSON report")
    ap.add_argument("--beta", type=float, default=0.2, help="Fairness beta (for (beta - SPD_sum)/beta)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="Optional: evaluate only first N rows for a quick pass")
    args = ap.parse_args()

    # Validate sources
    if bool(args.ckpt) == bool(args.ckpt_dir):
        raise SystemExit("Please provide exactly one of --ckpt or --ckpt_dir.")

    # Collect checkpoints
    if args.ckpt_dir:
        ckpts = sorted([str(p) for p in Path(args.ckpt_dir).glob("*.pt")])
        if not ckpts:
            raise SystemExit(f"No .pt checkpoints found under {args.ckpt_dir}")
    else:
        ckpts = [args.ckpt]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = {
        "fairface_csv": str(Path(args.fairface_csv).resolve()),
        "beta": args.beta,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "limit": args.limit if args.limit is not None else "full",
    }
    entries: List[Dict] = []

    for ckpt_path in ckpts:
        print(f"\n→ Evaluating: {ckpt_path}")
        try:
            model, tf_eval, arch = load_model_auto(ckpt_path)
        except Exception as e:
            print(f"[warn] Could not load model from {ckpt_path}: {e}")
            continue

        model.to(device)

        per_group_tpr, counts, n_total = evaluate_on_fairface(
            model=model,
            tf_eval=tf_eval,
            fairface_csv=args.fairface_csv,
            device=device,
            batch_size=args.batch,
            limit=args.limit,
        )
        fairness = compute_fairness_metrics(per_group_tpr, beta=args.beta)

        entry = {
            "ckpt": str(Path(ckpt_path).resolve()),
            "arch": arch,
            "n_total": n_total,
            "per_race_tpr": {k: float(v) for k, v in per_group_tpr.items()},
            "per_race_counts": {k: int(v) for k, v in counts.items()},
            "metrics": fairness,
        }
        # Add a simple overall (macro) mean TPR for reference
        if per_group_tpr:
            entry["overall_mean_tpr"] = float(sum(per_group_tpr.values()) / len(per_group_tpr))
        entries.append(entry)

        # Print a concise summary to stdout too
        print("Per-race TPR:")
        for g in sorted(per_group_tpr.keys()):
            print(f"  {g:>16}: {per_group_tpr[g]:.4f}  (n={counts[g]})")
        print(f"Min group TPR : {fairness['min_group_tpr']:.4f}")
        print(f"Max–Min gap   : {fairness['max_min_gap']:.4f}")
        print(f"SPD_sum       : {fairness['spd_sum']:.4f}")
        print(f"Fairness score: {fairness['fairness_score']:.4f}")

    if not entries:
        raise SystemExit("No models were evaluated successfully; aborting without JSON.")

    write_results_json(args.out_json, meta, entries)


if __name__ == "__main__":
    main()
