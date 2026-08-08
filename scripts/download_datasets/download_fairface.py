#!/usr/bin/env python
# fairness/download_fairface.py
"""
Materialize FairFace images and CSVs from the HuggingFaceM4/FairFace **Parquet** dataset
WITHOUT running any remote code.

It will:
  1) Snapshot the dataset repo locally (parquet files only).
  2) Load the parquet shards for the chosen variant (0.25 or 1.25) using `datasets`.
  3) Extract image bytes from the `image` column and save them as .jpg files.
  4) Write CSVs with absolute paths and race strings: image_path,race

Outputs:
  <repo_root>/data/FairFace/<variant>/
    train/images/*.jpg
    validation/images/*.jpg
    fairface_train.csv
    fairface_val.csv
  <repo_root>/data/FairFace/_snapshot/   # the raw parquet snapshot

Usage:
  pip install -U huggingface_hub datasets pyarrow pillow tqdm
  python fairness/download_fairface_from_parquet.py
  # or pick the 1.25 variant:
  python fairness/download_fairface_from_parquet.py --variant 1.25
  # or change output root:
  python fairness/download_fairface_from_parquet.py --out_root /mnt/fastdisk/data

Notes:
- We rely on HF's dataset structure (parquet + 'image' feature) as shown on the card/viewer.  # see citations
"""

import argparse
import csv
import io
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download
from datasets import load_dataset, Image as HFImage, DatasetDict
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # be tolerant to partial files


# ----------------------- helpers -----------------------

def repo_root_from_script() -> Path:
    # assumes this file lives at <repo_root>/fairness/download_fairface_from_parquet.py
    return Path(__file__).resolve().parent.parent

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def find_variant_parquets(snapshot_dir: Path, variant: str) -> Dict[str, List[str]]:
    """
    Find parquet shards for the chosen variant (e.g., '0.25' or '1.25').
    Returns a dict: {'train': [paths...], 'validation': [paths...]} (absolute paths as strings).
    """
    var_dir = snapshot_dir / variant
    if not var_dir.exists():
        # Some repos put files under data/<variant>
        var_dir = snapshot_dir / "data" / variant
    if not var_dir.exists():
        raise SystemExit(f"✖ Could not find variant dir under snapshot: {snapshot_dir} (looked for '{variant}' and 'data/{variant}')")

    train_glob = sorted(str(p) for p in var_dir.glob("train-*.parquet"))
    val_glob   = sorted(str(p) for p in var_dir.glob("validation-*.parquet"))

    if not train_glob:
        # Accept single-file naming too
        train_glob = sorted(str(p) for p in var_dir.rglob("*train*.parquet"))
    if not val_glob:
        val_glob   = sorted(str(p) for p in var_dir.rglob("*valid*.parquet"))

    if not train_glob or not val_glob:
        raise SystemExit(f"✖ Could not locate parquet shards for split(s). Found:\n  train: {len(train_glob)} files\n  val:   {len(val_glob)} files\n  variant root: {var_dir}")

    return {"train": train_glob, "validation": val_glob}

def load_parquet_as_dataset(files_by_split: Dict[str, List[str]]) -> DatasetDict:
    """
    Use the 'parquet' builder to load local shards.
    We then cast the 'image' column to HF Image(decode=False), so each row has:
       ex['image'] -> {'bytes': b'...'} or {'path': <possibly None>}
    """
    data_files = {k: v for k, v in files_by_split.items()}
    dsdict = load_dataset("parquet", data_files=data_files)
    # Ensure 'image' is the right type (sometimes it already is).
    for split in list(dsdict.keys()):
        if "image" in dsdict[split].column_names:
            dsdict[split] = dsdict[split].cast_column("image", HFImage(decode=False))
    return dsdict

def race_name_lookup(dsdict: DatasetDict):
    """
    Get race label names, robustly.
    - Prefer ClassLabel names from features if present.
    - Otherwise, fall back to the order seen on the dataset card/viewer.
    """
    # Try to read from any split's features
    for split in ("train", "validation"):
        if split in dsdict and "race" in dsdict[split].features:
            feat = dsdict[split].features["race"]
            try:
                return getattr(feat, "names", None)
            except Exception:
                pass
    # Fallback (order shown on HF viewer for HuggingFaceM4/FairFace)
    # 0 East Asian, 1 Indian, 2 Black, 3 White, 4 Middle Eastern, 5 Latino_Hispanic, 6 Southeast Asian
    return ["East Asian","Indian","Black","White","Middle Eastern","Latino_Hispanic","Southeast Asian"]

def save_split_images_and_csv(ds, split: str, out_img_dir: Path, out_csv: Path, race_names: List[str]) -> int:
    """
    Iterate the split; save images and write standardized CSV (image_path,race).
    Returns number of rows written.
    """
    ensure_dir(out_img_dir)
    ensure_dir(out_csv.parent)

    n = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["image_path","race"])

        for i, ex in enumerate(tqdm(ds, desc=f"Saving {split}")):
            imf = ex["image"]
            # imf is dict-like: {'bytes': b'...'} or may include 'path'
            b = None
            if isinstance(imf, dict):
                b = imf.get("bytes", None)
                # If parquet stored paths (unlikely here), try to read bytes from disk.
                if b is None and imf.get("path"):
                    with open(imf["path"], "rb") as fh:
                        b = fh.read()
            if b is None:
                # As an absolute fallback, try to get PIL through datasets (rarely needed)
                # but we avoid it to keep things robust/offline.
                raise RuntimeError("No image bytes found in row (unexpected parquet format).")

            img = Image.open(io.BytesIO(b)).convert("RGB")
            fname = f"{split}_{i:06d}.jpg"
            outp  = out_img_dir / fname
            img.save(outp, quality=95)

            # race can be int or string depending on parquet; normalize to string
            r = ex["race"]
            if isinstance(r, (int, float)) and 0 <= int(r) < len(race_names):
                r_str = race_names[int(r)]
            else:
                r_str = str(r)

            w.writerow([str(outp.resolve()), r_str])
            n += 1
    return n


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Download FairFace parquet snapshot and materialize images + CSVs.")
    ap.add_argument("--variant", type=str, default="0.25", choices=["0.25","1.25"], help="FairFace crop padding variant.")
    ap.add_argument("--out_root", type=str, default="datasets", help="Output root (default: <repo_root>/data).")
    ap.add_argument("--repo_id", type=str, default="HuggingFaceM4/FairFace", help="HF dataset repo id (parquet).")
    args = ap.parse_args()

    repo_root = repo_root_from_script()
    data_root = Path(args.out_root).resolve() if args.out_root else (repo_root / "data")

    snap_dir = data_root / "FairFace" / "_snapshot"
    ensure_dir(snap_dir)

    print(f"↓ Snapshotting {args.repo_id} into {snap_dir} (parquet only; no code)...")
    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(snap_dir),
        local_dir_use_symlinks=False,  # harmless deprecation warning on recent hub versions
        allow_patterns=None,
        ignore_patterns=["*.md",".gitattributes",".git/*"]
    )
    snapshot = Path(local_path)

    print(f"   • Using snapshot at: {snapshot}")
    files_by_split = find_variant_parquets(snapshot, args.variant)

    print("↓ Loading parquet shards as a local dataset ...")
    dsdict = load_parquet_as_dataset(files_by_split)
    race_names = race_name_lookup(dsdict)

    base = data_root / "FairFace" / args.variant
    train_img_dir = base / "train" / "images"
    val_img_dir   = base / "validation" / "images"
    train_csv     = base / "fairface_train.csv"
    val_csv       = base / "fairface_val.csv"

    n_train = n_val = 0
    if "train" in dsdict:
        n_train = save_split_images_and_csv(dsdict["train"], "train", train_img_dir, train_csv, race_names)
    if "validation" in dsdict:
        n_val = save_split_images_and_csv(dsdict["validation"], "validation", val_img_dir, val_csv, race_names)

    print("\n✅ Done.")
    print(f"  Train: {train_img_dir}  (images: {n_train})")
    print(f"  Val  : {val_img_dir}    (images: {n_val})")
    print(f"  Train CSV: {train_csv}")
    print(f"  Val   CSV: {val_csv}")
    print("\nNext (example):")
    print("  python fairness/eval_fairface_race.py \\")
    print("    --ckpt checkpoints/facebin_resnet18.pt \\")
    print(f"    --fairface_csv {val_csv}")

if __name__ == "__main__":
    main()
