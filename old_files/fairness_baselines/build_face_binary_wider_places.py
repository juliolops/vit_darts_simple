#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_face_binary_wider_places.py

Create a **face vs non-face** classification dataset from:
  • WIDER FACE (positives)  -> crops of face boxes
  • Places365 / OpenImages (negatives) -> random patches matched to positive crop size

This version includes a **robust WIDER split parser** that works with both:
  1) the standard format (filename.jpg, <num_boxes>, then <num_boxes> lines), and
  2) variants where the <num_boxes> line is missing (we infer boxes until next *.jpg line).

Example:
    python fairness/build_face_binary_wider_places.py \
        --wider_root data/WIDER \
        --neg_root   data/PLACES365/val \
        --out_dir    facebin_data

Outputs (by default):
  facebin_data/
    train/{face,non_face}/...
    val/{face,non_face}/...
  plus CSV manifests: train_face.csv, train_non_face.csv, val_*.csv
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
from tqdm import tqdm


# ---------------------------
# Robust WIDER split parsing
# ---------------------------

def _try_parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _parse_box_line(line: str) -> Optional[Tuple[int, int, int, int]]:
    """
    WIDER box lines have at least 4 integers: x y w h [blur expr illum invalid occ pose]
    We only need the first 4.
    """
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = map(int, parts[:4])
        # Filter clearly invalid boxes
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)
    except Exception:
        return None


def read_wider_split(txt_path: str) -> List[Tuple[str, List[Tuple[int, int, int, int]]]]:
    """
    Robustly parse WIDER split file into a list of (image_rel_path, boxes).

    Handles both:
        A) Standard format:
            <rel_path.jpg>
            <num_boxes>
            x y w h <...>
            ...
        B) Variant without <num_boxes>:
            <rel_path.jpg>
            x y w h <...>
            x y w h <...>
            <next_rel_path.jpg>
    """
    items: List[Tuple[str, List[Tuple[int, int, int, int]]]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    i = 0
    N = len(lines)
    while i < N:
        line = lines[i]

        # Skip anything that doesn't look like an image path
        if not line.lower().endswith(".jpg"):
            i += 1
            continue

        rel_path = line
        i += 1
        boxes: List[Tuple[int, int, int, int]] = []

        # If there is an integer here, it's the count line
        if i < N:
            maybe_count = _try_parse_int(lines[i])
        else:
            maybe_count = None

        if maybe_count is not None:
            # Standard format with explicit count
            i += 1
            cnt = maybe_count
            for _ in range(cnt):
                if i >= N:
                    break
                box = _parse_box_line(lines[i])
                i += 1
                if box is not None:
                    boxes.append(box)

        else:
            # Variant format: keep reading box-like lines until we hit the next *.jpg
            while i < N and not lines[i].lower().endswith(".jpg"):
                box = _parse_box_line(lines[i])
                i += 1
                if box is not None:
                    boxes.append(box)

        items.append((rel_path, boxes))

    return items


# ---------------------------
# Imaging helpers
# ---------------------------

def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def expand_box(x: int, y: int, w: int, h: int, scale: float, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * scale, h * scale
    x0 = int(round(cx - nw / 2.0))
    y0 = int(round(cy - nh / 2.0))
    x1 = int(round(cx + nw / 2.0))
    y1 = int(round(cy + nh / 2.0))
    x0, y0 = clamp(x0, 0, W - 1), clamp(y0, 0, H - 1)
    x1, y1 = clamp(x1, 0, W), clamp(y1, 0, H)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def random_neg_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    W, H = img.size
    tw, th = min(target_w, W), min(target_h, H)
    if W == tw and H == th:
        return img.copy()
    import random as pyrand
    x0 = pyrand.randint(0, max(0, W - tw))
    y0 = pyrand.randint(0, max(0, H - th))
    return img.crop((x0, y0, x0 + tw, y0 + th))


def collect_images(root: str, exts=(".jpg", ".jpeg", ".png")) -> List[Path]:
    paths = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(exts):
                paths.append(Path(dp) / fn)
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Split processing
# ---------------------------

def process_split(
    wider_root: Path,
    split: str,
    neg_pool: List[Path],
    out_dir: Path,
    min_face: int,
    expand: float,
    max_pos_per_img: int,
    neg_per_pos: int,
):
    img_root = wider_root / f"WIDER_{split}" / "images"
    split_file = wider_root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Cannot find split file: {split_file}")

    items = read_wider_split(str(split_file))

    face_dir = out_dir / split / "face"
    neg_dir = out_dir / split / "non_face"
    ensure_dir(face_dir)
    ensure_dir(neg_dir)

    # manifests
    man_face = open(out_dir / f"{split}_face.csv", "w", encoding="utf-8")
    man_neg = open(out_dir / f"{split}_non_face.csv", "w", encoding="utf-8")
    print("image_path", file=man_face)
    print("image_path", file=man_neg)

    pos_count = 0
    neg_count = 0

    for rel, boxes in tqdm(items, desc=f"[{split}] cropping"):
        img_path = img_root / rel
        if not img_path.exists():
            # Some annotations point to images not present in your local copy; skip them
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        W, H = img.size
        boxes_iter = boxes if max_pos_per_img <= 0 else boxes[:max_pos_per_img]

        for (x, y, w, h) in boxes_iter:
            if w < min_face or h < min_face:
                continue

            # positive crop (expanded)
            box = expand_box(x, y, w, h, expand, W, H)
            if box is None:
                continue

            px0, py0, px1, py1 = box
            pos = img.crop((px0, py0, px1, py1))

            # save positive
            pos_out = face_dir / f"{img_path.stem}_{px0}_{py0}_{px1}_{py1}.jpg"
            try:
                pos.save(pos_out, quality=95)
            except Exception:
                continue
            print(pos_out.as_posix(), file=man_face)
            pos_count += 1

            # negatives matched in spatial size
            tw, th = pos.size
            for _ in range(neg_per_pos):
                neg_src = random.choice(neg_pool)
                try:
                    nimg = Image.open(neg_src).convert("RGB")
                except Exception:
                    continue
                crop = random_neg_crop(nimg, tw, th)
                neg_out = neg_dir / f"{neg_src.stem}_{tw}x{th}_{random.randint(0, 999999)}.jpg"
                try:
                    crop.save(neg_out, quality=95)
                    print(neg_out.as_posix(), file=man_neg)
                    neg_count += 1
                except Exception:
                    continue

    man_face.close()
    man_neg.close()
    print(f"→ {split}: saved {pos_count} faces and {neg_count} non-faces at {out_dir}/{split}/")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wider_root", required=True, type=str, help="Root of WIDER (contains WIDER_train/, WIDER_val/, wider_face_split/)")
    ap.add_argument("--neg_root", required=True, type=str, help="Root of negatives (Places365/OpenImages)")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory (will create {train,val}/{face,non_face})")
    ap.add_argument("--min_face", type=int, default=24, help="Skip faces smaller than this (in pixels)")
    ap.add_argument("--expand", type=float, default=1.20, help="Scale factor around bbox to include context")
    ap.add_argument("--max_pos_per_img", type=int, default=20, help="Crop at most N faces per WIDER image (<=0 for all)")
    ap.add_argument("--neg_per_pos", type=int, default=1, help="Negatives to sample per positive crop")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    wider_root = Path(args.wider_root)
    neg_root = Path(args.neg_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect negatives pool
    neg_pool = collect_images(str(neg_root))

    for split in ("train", "val"):
        process_split(
            wider_root=wider_root,
            split=split,
            neg_pool=neg_pool,
            out_dir=out_dir,
            min_face=args.min_face,
            expand=args.expand,
            max_pos_per_img=args.max_pos_per_img,
            neg_per_pos=args.neg_per_pos,
        )


if __name__ == "__main__":
    main()
