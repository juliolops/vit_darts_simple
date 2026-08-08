#!/usr/bin/env python
# fairness/eval_facet_skintone.py
"""
Evaluate FACE vs NON-FACE checkpoints on FACET, grouping by skin tone.

Supports:
    - HARD labels via 'skin_tone_final' (1..10)
    - SOFT labels via 'skin_tone_probs' (length-10 JSON array)

Fairness:
    SPD_sum = sum_g (TPR_g - min_g TPR_g)
    Fairness = max(0, (beta - SPD_sum)/beta)      [beta=0.2 default]

Usage:
# Build a CSV first (with hard + soft):
python fairness/build_facet_dataset.py \
    --base_dir facet_data \
    --ann_csv facet_data/annotations/annotations.csv \
    --img_dirs facet_data/imgs_1 facet_data/imgs_2 facet_data/imgs_3 \
    --out_csv facet_data/facet_eval.csv \
    --hard_strategy median_round --seed 42

# Evaluate a single checkpoint (both hard and soft)
python fairness/eval_facet_skintone.py \
    --facet_csv facet_data/facet_eval.csv \
    --ckpt checkpoints/facebin_resnet18.pt \
    --out_json fairness/facet_resnet18.json

# Evaluate an entire directory of checkpoints, soft only
python fairness/eval_facet_skintone.py \
    --facet_csv facet_data/facet_eval.csv \
    --ckpt_dir checkpoints \
    --mode soft \
    --out_json fairness/facet_all_soft.json

Fast FACET evaluator:
- Batched GPU inference with AMP
- Multi-worker DataLoader
- Optional on-disk crop cache to avoid re-cropping next runs
- Outputs JSON with hard/soft (both) fairness

Usage:
python fairness/eval_facet_skintone.py \
    --facet_csv facet_data/facet_eval.csv \
    --ckpt_dir checkpoints_personbin \
    --mode both --beta 0.2 \
    --device cuda:1 --batch 128 --num_workers 4 --prefetch 4 \
    --cache_dir .cache/facet_crops \
    --out_json fairness/facet_results.json
"""
import argparse, json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, models
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
import pandas as pd
from collections import OrderedDict

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------- Model registry (match your trainer) ----------
def _set_fc(m, nc): m.fc = nn.Linear(m.fc.in_features, nc)
def _set_cls(m, nc):
    idx = None
    for i in reversed(range(len(m.classifier))):
        if isinstance(m.classifier[i], nn.Linear):
            idx = i; break
    if idx is None:
        raise RuntimeError("No Linear layer in classifier; adjust head replacement.")
    in_f = m.classifier[idx].in_features
    m.classifier[idx] = nn.Linear(in_f, nc)
def _set_convnext_head(m, nc):
    lin = m.classifier[-1]
    m.classifier[-1] = nn.Linear(lin.in_features, nc)

ARCH = {}
try: ARCH["resnet18"]          = (models.resnet18,          models.ResNet18_Weights.IMAGENET1K_V1,           _set_fc)
except: pass
try: ARCH["resnet50"]          = (models.resnet50,          models.ResNet50_Weights.IMAGENET1K_V2,           _set_fc)
except: pass
try: ARCH["mobilenet_v3_large"]= (models.mobilenet_v3_large,models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, _set_cls)
except: pass
try: ARCH["efficientnet_v2_s"] = (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,  _set_cls)
except: pass
try: ARCH["convnext_tiny"]     = (models.convnext_tiny,     models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,      _set_convnext_head)
except: pass

def _get_weights_meta(weights):
    try:
        meta = getattr(weights, "meta", None)
        if isinstance(meta, dict):
            return list(meta.get("mean", IMAGENET_MEAN)), list(meta.get("std", IMAGENET_STD))
    except: pass
    return IMAGENET_MEAN, IMAGENET_STD

def _build_model_and_transform(arch: str, img_size: int = 224):
    ctor, weights_enum, head = ARCH[arch]
    try:
        weights = getattr(weights_enum, "DEFAULT", weights_enum)
        m = ctor(weights=weights); mean, std = _get_weights_meta(weights)
    except:
        try:
            m = ctor(pretrained=True); mean, std = IMAGENET_MEAN, IMAGENET_STD
        except:
            m = ctor(); mean, std = IMAGENET_MEAN, IMAGENET_STD
    head(m, 2)
    resize = 256 if img_size == 224 else int(img_size * 256 / 224)
    tf = T.Compose([T.Resize(resize), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(mean, std)])
    return m, tf

# ---------- Checkpoint loader (preserve _metadata) ----------
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict): sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict): sd = obj["state_dict"]
        else: sd = obj
    else:
        raise ValueError("Unsupported checkpoint object.")
    meta = getattr(sd, "_metadata", None)
    new_sd = OrderedDict((k.replace("module.",""), v) for k, v in sd.items())
    if meta is not None:
        try:
            new_sd._metadata = {k.replace("module.",""): v for k, v in meta.items()}  # type: ignore[attr-defined]
        except Exception:
            pass
    return new_sd

def load_model_auto(ckpt_path: str):
    name = Path(ckpt_path).name.lower()
    arch = None
    for k in ARCH.keys():
        if k in name: arch = k; break
    if arch is None:
        raise RuntimeError(f"Could not infer arch from filename: {ckpt_path}")
    m, tf = _build_model_and_transform(arch, 224)
    # try safe first, fallback to full (keeps metadata)
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
        sd  = _extract_state_dict(obj)
        m.load_state_dict(sd, strict=False)
    except Exception:
        obj = torch.load(ckpt_path, map_location="cpu")
        sd  = _extract_state_dict(obj)
        m.load_state_dict(sd, strict=False)
    m.eval()
    return m, tf, arch

# ---------- Dataset ----------
class FacetEvalSet(Dataset):
    def __init__(self, csv_path: str, tf: T.Compose, cache_dir: Optional[str]=None):
        df = pd.read_csv(csv_path)
        needed = {"image_path","x","y","width","height","skin_tone_probs"}
        if not needed.issubset(df.columns):
            raise ValueError(f"CSV must have columns: {sorted(list(needed))}")
        self.df = df
        self.tf = tf
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self): return len(self.df)

    def _cached_path(self, row) -> Path:
        # stable name: <stem>_x_y_w_h.jpg under cache/
        p = Path(row["image_path"])
        tag = f"{p.stem}_{int(row['x'])}_{int(row['y'])}_{int(row['width'])}_{int(row['height'])}.jpg"
        return self.cache / tag

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self.cache:
            cpath = self._cached_path(row)
            if cpath.exists():
                img = Image.open(cpath).convert("RGB")
            else:
                img = Image.open(row["image_path"]).convert("RGB")
                x, y, w, h = float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])
                img = img.crop((x, y, x+w, y+h))
                # write-through cache (best-effort)
                try:
                    img.save(cpath, quality=90, subsampling=1)
                except Exception:
                    pass
        else:
            img = Image.open(row["image_path"]).convert("RGB")
            x, y, w, h = float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])
            img = img.crop((x, y, x+w, y+h))

        x = self.tf(img)  # tensor CHW normalized
        # parse soft labels
        probs = row["skin_tone_probs"]
        # ensure JSON-ish list → list[float]
        if isinstance(probs, str):
            probs = json.loads(probs)
        return x, probs  # ground truth is always positive

# ---------- Fairness ----------
def fairness_from_preds(
    preds: torch.Tensor,  # shape [N], 0/1
    soft_probs: List[List[float]],  # list of length-10 lists
    mode: str = "both",
    beta: float = 0.2,
) -> Dict:
    import numpy as np
    preds = preds.cpu().numpy().astype(int)
    P = np.array([np.array(p, dtype=float) for p in soft_probs])  # [N,10]
    tone_ix = np.arange(10)

    out = {}
    # HARD: argmax tone
    if mode in ("hard","both"):
        hard = P.argmax(axis=1)  # [N], in 0..9
        per = {}
        counts = {}
        for g in tone_ix:
            mask = (hard == g)
            n = int(mask.sum())
            counts[str(g+1)] = n
            per[str(g+1)] = float(preds[mask].mean()) if n > 0 else 0.0
        vals = list(per.values())
        acc_min, acc_max = (min(vals), max(vals)) if vals else (0.0,0.0)
        spd_sum = sum(v-acc_min for v in vals)
        fair = max(0.0, (beta - spd_sum)/beta)
        out["hard"] = {
            "per_tone_tpr": per, "counts": counts,
            "overall_mean_tpr": float(sum(vals)/max(1,len(vals))),
            "metrics": {"min_group_tpr": float(acc_min), "max_min_gap": float(acc_max-acc_min),
                        "spd_sum": float(spd_sum), "fairness": float(fair)}
        }

    # SOFT: expected TPR by tone
    if mode in ("soft","both"):
        per = {}
        denom = {}
        nume = {}
        for g in range(10):
            pg = P[:, g]
            denom_g = float(pg.sum())
            nume_g  = float((preds * pg).sum())
            denom[str(g+1)] = denom_g
            per[str(g+1)]   = (nume_g / denom_g) if denom_g > 0 else 0.0
            nume[str(g+1)]  = nume_g
        vals = list(per.values())
        acc_min, acc_max = (min(vals), max(vals)) if vals else (0.0,0.0)
        spd_sum = sum(v-acc_min for v in vals)
        fair = max(0.0, (beta - spd_sum)/beta)
        out["soft"] = {
            "per_tone_tpr": per, "denom": denom,
            "overall_mean_tpr": float(sum(vals)/max(1,len(vals))),
            "metrics": {"min_group_tpr": float(acc_min), "max_min_gap": float(acc_max-acc_min),
                        "spd_sum": float(spd_sum), "fairness": float(fair)}
        }
    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facet_csv", required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--mode", type=str, default="both", choices=["hard","soft","both"])
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache_dir", type=str, default=None, help="Optional on-disk crop cache")
    ap.add_argument("--out_json", type=str, default="fairness/facet_results.json")
    args = ap.parse_args()

    if bool(args.ckpt) == bool(args.ckpt_dir):
        raise SystemExit("Provide exactly one of --ckpt or --ckpt_dir.")

    ckpts = [args.ckpt] if args.ckpt else sorted([str(p) for p in Path(args.ckpt_dir).glob("*.pt")])
    if not ckpts:
        raise SystemExit("No checkpoints to evaluate.")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    meta = {
        "facet_csv": str(Path(args.facet_csv).resolve()),
        "mode": args.mode, "beta": args.beta,
        "device": str(device), "limit": args.limit if args.limit else "full",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    results: List[Dict] = []

    # Build dataset/loader once (used for all models)
    # Use resnet18 normalization stats for tf (close enough across ImageNet models)
    _, tf_eval = _build_model_and_transform(list(ARCH.keys())[0], 224)
    ds = FacetEvalSet(args.facet_csv, tf_eval, cache_dir=args.cache_dir)
    if args.limit: ds.df = ds.df.head(args.limit)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                    prefetch_factor=args.prefetch, persistent_workers=(args.num_workers>0))

    # Collect soft labels once
    soft_labels: List[List[float]] = [json.loads(p) if isinstance(p,str) else p for p in ds.df["skin_tone_probs"].tolist()]

    for ckpt in ckpts:
        print(f"\n→ Evaluating: {ckpt}")
        model, _, arch = load_model_auto(ckpt)
        model.to(device)

        preds = []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=amp_device, enabled=(amp_device=="cuda")):
                    logits = model(x)
                preds.append(logits.argmax(1).cpu())
        preds = torch.cat(preds, dim=0)  # [N]
        fair = fairness_from_preds(preds, soft_labels, mode=args.mode, beta=args.beta)

        entry = {
            "ckpt": str(Path(ckpt).resolve()),
            "arch": arch,
            "n_total": int(len(ds)),
            **fair
        }
        results.append(entry)

        # quick console summary
        for sec in ("hard","soft"):
            if sec in fair:
                m = fair[sec]["metrics"]
                print(f"[{arch}][{sec}] fairness={m['fairness']:.4f}  minTPR={m['min_group_tpr']:.4f}  gap={m['max_min_gap']:.4f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "results": results}, f, indent=2)
    print(f"\n💾 Saved JSON → {args.out_json}")

if __name__ == "__main__":
    main()