#!/usr/bin/env python
# fairness/train_facebin_models.py
"""
Simple, torchvision-only baselines for PERSON/FACE vs NON-* on your binary datasets.

Included backbones (train those your torchvision supports):
    - resnet18
    - resnet50
    - efficientnet_v2_s
    - convnext_tiny
    - mobilenet_v3_large
    - mnasnet1_0

Robustness:
- Falls back to ImageNet stats if weights.meta is missing.
- Tries ctor(weights=...), then ctor(pretrained=True), then ctor().
- Saves state_dict only.
- Uses torch.amp (with fallback to torch.cuda.amp on older PyTorch).
- Dataset loader autodetects ('face','non_face') or ('person','non_person') and supports jpg/jpeg/png.

Usage examples
--------------
# Train all (auto-detect class folder names)
python fairness/train_facebin_models.py --data_root data/cocobin_data --epochs 10 --bs 128

# Explicit class names
python fairness/train_facebin_models.py --data_root data/cocobin_data \
    --pos_name person --neg_name non_person --epochs 10 --bs 128

# Pick device explicitly
python fairness/train_facebin_models.py --data_root data/cocobin_data --device cuda:0
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Callable, Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.transforms as T
from torchvision import models

# AMP (new API, fallback for older PyTorch)
try:
    from torch.amp import autocast, GradScaler
except ImportError:  # older PyTorch
    from torch.cuda.amp import autocast, GradScaler  # type: ignore

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------- Dataset ----------------
class BinaryFolderDataset(Dataset):
    """
    Expects: root/{train,val}/{<pos_name>,<neg_name>}/*.(jpg|jpeg|png)
    Autodetects ('face','non_face') or ('person','non_person') if names not given.
    """
    def __init__(self, root, split, tfm=None,
                pos_name: Optional[str]=None, neg_name: Optional[str]=None):
        split_root = Path(root) / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split folder: {split_root}")

        # Autodetect if not provided
        if pos_name is None or neg_name is None:
            subs = {d.name for d in split_root.iterdir() if d.is_dir()}
            if {"face", "non_face"}.issubset(subs):
                pos_name, neg_name = "face", "non_face"
            elif {"person", "non_person"}.issubset(subs):
                pos_name, neg_name = "person", "non_person"
            else:
                raise RuntimeError(
                    f"Could not autodetect class folders in {split_root}. "
                    f"Found: {sorted(subs)} — expected either "
                    f"('face','non_face') or ('person','non_person'), "
                    f"or pass --pos_name/--neg_name."
                )

        self.pos_name, self.neg_name = pos_name, neg_name
        self.samples: List[tuple] = []
        exts = {".jpg", ".jpeg", ".png"}
        for cls, y in [(self.pos_name, 1), (self.neg_name, 0)]:
            cls_dir = split_root / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in exts:
                    self.samples.append((str(p), y))
        if not self.samples:
            raise RuntimeError(f"No images found in {split_root} for classes {self.pos_name}/{self.neg_name}")
        self.tfm = tfm

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        x = Image.open(p).convert("RGB")
        return (self.tfm(x) if self.tfm else x), y


# ------------- Utils -------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _replace_last_linear_in_sequential(seq: nn.Sequential, nc: int):
    # find last Linear in the seq (handles Dropout, etc.)
    idx = None
    for i in reversed(range(len(seq))):
        if isinstance(seq[i], nn.Linear):
            idx = i
            break
    if idx is None:
        raise RuntimeError("Classifier tail has no Linear layer; adjust head replacement.")
    in_f = seq[idx].in_features
    seq[idx] = nn.Linear(in_f, nc)


def _get_weights_meta(weights) -> Tuple[List[float], List[float]]:
    """Return (mean, std) robustly across torchvision versions."""
    # Newer torchvision: weights.meta is a dict with 'mean'/'std'
    try:
        meta = getattr(weights, "meta", None)
        if isinstance(meta, dict):
            mean = meta.get("mean", IMAGENET_MEAN)
            std  = meta.get("std", IMAGENET_STD)
            return list(mean), list(std)
    except Exception:
        pass
    # Fallback: try weights.transforms() to infer normalization
    try:
        tf = weights.transforms()
        for t in getattr(tf, "transforms", []):
            if isinstance(t, T.Normalize):
                return list(t.mean), list(t.std)
    except Exception:
        pass
    # Oldest fallback: hard-coded ImageNet stats
    return IMAGENET_MEAN, IMAGENET_STD


def _build_with_weights_or_pretrained(ctor, weights_enum):
    """
    Try ctor(weights=weights_enum.DEFAULT) → ctor(weights=weights_enum) →
    ctor(pretrained=True) → ctor() as last resort.
    Returns (model, mean, std).
    """
    # 1) enum.DEFAULT (newer API)
    try:
        weights = getattr(weights_enum, "DEFAULT", weights_enum)
        m = ctor(weights=weights)
        mean, std = _get_weights_meta(weights)
        return m, mean, std
    except Exception:
        pass
    # 2) enum instance (some versions accept)
    try:
        m = ctor(weights=weights_enum)
        mean, std = _get_weights_meta(weights_enum)
        return m, mean, std
    except Exception:
        pass
    # 3) pretrained=True (older API)
    try:
        m = ctor(pretrained=True)
        return m, IMAGENET_MEAN, IMAGENET_STD
    except Exception:
        pass
    # 4) scratch
    m = ctor()
    return m, IMAGENET_MEAN, IMAGENET_STD


# ------------- Model factory (torchvision only) -------------
def make_tv_model_and_transforms(arch: str, num_classes: int = 2, img_size: int = 224):
    """
    Build a torchvision model with ImageNet weights (if available), replace the head,
    and return (model, tf_train, tf_val).
    """
    arch = arch.lower().strip()

    def _set_fc(m, nc): m.fc = nn.Linear(m.fc.in_features, nc)
    def _set_cls(m, nc): _replace_last_linear_in_sequential(m.classifier, nc)
    def _set_convnext_head(m, nc):
        # ConvNeXt classifier: [LayerNorm, Flatten, Linear]
        lin = m.classifier[-1]
        m.classifier[-1] = nn.Linear(lin.in_features, nc)

    reg: Dict[str, Tuple[Callable, object, Callable]] = {}

    # Populate only what exists in the installed torchvision version.
    try: reg["resnet18"] = (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, _set_fc)
    except Exception: pass
    try: reg["resnet50"] = (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, _set_fc)
    except Exception: pass
    try: reg["mobilenet_v3_large"] = (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, _set_cls)
    except Exception: pass
    try: reg["efficientnet_v2_s"] = (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, _set_cls)
    except Exception: pass
    try: reg["convnext_tiny"] = (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, _set_convnext_head)
    except Exception: pass

    if arch not in reg:
        raise ValueError(f"'{arch}' not available in your torchvision. Choose from: {list(reg.keys())}")

    ctor, weights_enum, head_setter = reg[arch]
    model, mean, std = _build_with_weights_or_pretrained(ctor, weights_enum)
    head_setter(model, num_classes)

    # Build transforms (augment on train; center-crop on val)
    tf_train = T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_resize = 256 if img_size == 224 else int(img_size * 256 / 224)
    tf_val = T.Compose([
        T.Resize(eval_resize),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return model, tf_train, tf_val


# ------------- Train one backbone -------------
def train_one_model(
    arch: str,
    data_root: Path,
    out_dir: Path,
    device: torch.device,
    pos_name: Optional[str],
    neg_name: Optional[str],
    epochs: int = 10,
    bs: int = 128,
    lr: float = 1e-3,
    wd: float = 1e-4,
    img_size: int = 224,
    freeze_backbone: bool = False,
    opt_name: str = "adamw",
    num_workers: int = 4,
    results_csv: Optional[Path] = None,
):
    print(f"\n=== [{arch}] building model & transforms ===")
    model, tf_train, tf_val = make_tv_model_and_transforms(arch, 2, img_size)

    # Freeze backbone (optional linear probe)
    if freeze_backbone:
        for p in model.parameters(): p.requires_grad = False
        head_params = []
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            head_params = list(model.fc.parameters())
        elif hasattr(model, "classifier"):
            head_params = list(model.classifier.parameters())
        for p in head_params: p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]

    # Data
    train_ds = BinaryFolderDataset(
        data_root, "train", tfm=tf_train,
        pos_name=pos_name, neg_name=neg_name
    )
    val_ds = BinaryFolderDataset(
        data_root, "val", tfm=tf_val,
        pos_name=pos_name, neg_name=neg_name
    )
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin)

    model.to(device)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(enabled=(amp_device == "cuda"))

    # Optimizer & schedule
    opt = optim.AdamW(trainable, lr=lr, weight_decay=wd) if opt_name=="adamw" \
            else optim.SGD(trainable, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    total_steps = epochs * max(1, len(train_dl))
    warmup = max(50, int(0.03 * total_steps))
    def lr_lambda(step):
        if step < warmup: return float(step)/float(max(1,warmup))
        pct = (step - warmup)/float(max(1,total_steps - warmup))
        return 0.5*(1.0 + torch.cos(torch.tensor(pct*3.1415926535))).item()
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"facebin_{arch}.pt"
    best = 0.0

    print(f"=== [{arch}] epochs={epochs} bs={bs} lr={lr} wd={wd} device={device} ===")
    for ep in range(1, epochs+1):
        # train
        model.train(); tr_loss=0.0; tr_hit=0; tr_tot=0
        for x, y in train_dl:
            x, y = x.to(device), torch.as_tensor(y, device=device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=amp_device, enabled=(amp_device == "cuda")):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            tr_loss += float(loss)*x.size(0)
            tr_hit  += (logits.argmax(1)==y).sum().item()
            tr_tot  += x.size(0)
        tr_acc = tr_hit / max(1, tr_tot); tr_loss /= max(1, tr_tot)

        # val
        model.eval(); va_loss=0.0; va_hit=0; va_tot=0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), torch.as_tensor(y, device=device)
                logits = model(x)
                loss = crit(logits, y)
                va_loss += float(loss)*x.size(0)
                va_hit  += (logits.argmax(1)==y).sum().item()
                va_tot  += x.size(0)
        va_acc = va_hit / max(1, va_tot); va_loss /= max(1, va_tot)
        print(f"[{arch}][{ep:02d}/{epochs}] train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | lr {sched.get_last_lr()[0]:.6f}")

        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ saved best to {ckpt_path} (val_acc={best:.4f})")

    # optional: write a quick results CSV
    if results_csv is not None:
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        new_file = not results_csv.exists()
        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file: w.writerow(["arch","best_val_acc","ckpt_path"])
            w.writerow([arch, f"{best:.6f}", str(ckpt_path.resolve())])

    print(f"=== [{arch}] done. Best val_acc={best:.4f} | ckpt={ckpt_path} ===")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root with train/ and val/ subfolders")
    ap.add_argument("--archs", type=str,
                    default="resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large",
                    help="Comma-separated torchvision archs to train.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--opt", type=str, default="adamw", choices=["adamw","sgd"])
    ap.add_argument("--freeze_backbone", action="store_true", help="Train only final head.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--results_csv", type=str, default="checkpoints/baseline_results.csv")
    ap.add_argument("--pos_name", type=str, default=None, help="Positive class folder name (e.g., 'face' or 'person').")
    ap.add_argument("--neg_name", type=str, default=None, help="Negative class folder name (e.g., 'non_face' or 'non_person').")
    ap.add_argument("--device", type=str, default=None, help="e.g., 'cuda:0', 'cuda:1', or 'cpu'. Default picks best available.")
    args = ap.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device cleanly (no env var hacks)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Probe availability and keep only models that build
    requested = [a.strip() for a in args.archs.split(",") if a.strip()]
    available = []
    for a in requested:
        try:
            m, _, _ = make_tv_model_and_transforms(a, 2, args.img_size)
            del m
            available.append(a)
        except Exception as e:
            print(f"[warn] Skipping '{a}' (not available or build failed): {e}")

    if not available:
        raise SystemExit("No requested architectures are available in your torchvision build.")

    print(f"Device: {device}")
    print(f"Architectures to train: {available}")
    results_csv = Path(args.results_csv)

    for arch in available:
        train_one_model(
            arch=arch,
            data_root=data_root,
            out_dir=out_dir,
            device=device,
            pos_name=args.pos_name,
            neg_name=args.neg_name,
            epochs=args.epochs,
            bs=args.bs,
            lr=args.lr,
            wd=args.wd,
            img_size=args.img_size,
            freeze_backbone=args.freeze_backbone,
            opt_name=args.opt,
            num_workers=args.num_workers,
            results_csv=results_csv,
        )

if __name__ == "__main__":
    main()
