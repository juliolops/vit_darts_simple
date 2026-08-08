#!/usr/bin/env python
# fairness/build_person_binary_coco_places.py
"""
Build a PERSON vs NON-PERSON **ROI-crop** dataset for binary classification.

✅ What’s new (compared to your previous version):
- Add **COCO-only negatives** (no Places365 contamination):
  • Image-level negatives: crops from COCO images that have **no person annotations**.
  • Hard background negatives: crops from COCO images **with persons** but sampled to
    avoid **any overlap** with person boxes (IoU=0 / no intersection).
- Keep **Places** option for backwards compatibility.
- Keep per-split **1:1 balance** (#positives == #negatives).
- Deterministic with `--seed`.
- Optional **FACET leakage guard** (skip COCO images that appear in your FACET CSV).

Output layout (ready for your trainer):
    out_dir/
    train/{person,non_person}/*.jpg
    val/  {person,non_person}/*.jpg

Examples
--------
# COCO-only negatives (recommended)
python fairness/build_person_binary.py \
    --coco_root data/COCO_sub \
    --out_dir   data/personbin_data \
    --neg_mode  coco_only \
    --neg_ratio_hard_bg 0.5 \
    --expand 1.25 --min_box 32 --max_per_image 8 \
    --neg_max_per_image 8 --seed 42

# Legacy: Places365 negatives
python fairness/build_person_binary.py \
    --coco_root coco_subset \
    --neg_root  data/PLACES365/val \
    --out_dir   data/personbin_data \
    --neg_mode  places \
    --expand 1.25 --min_box 32 --max_per_image 8 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from pycocotools.coco import COCO
except Exception as e:
    raise SystemExit(
        "pycocotools is required. Install with:\n"
        "  pip install pycocotools\n\n"
        f"Import error: {e}"
    )

from tqdm import tqdm


# --------------------- IO helpers ---------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_crop(img: Image.Image, box: Tuple[int, int, int, int], out_path: Path) -> bool:
    """Crop (L,T,R,B) with clamp and save JPEG; return True on success."""
    W, H = img.size
    L, T, R, B = map(int, box)
    L = max(0, min(L, W))
    T = max(0, min(T, H))
    R = max(0, min(R, W))
    B = max(0, min(B, H))
    if R <= L or B <= T:
        return False
    try:
        crop = img.crop((L, T, R, B))
        crop.save(out_path, format="JPEG", quality=90, subsampling=1)
        return True
    except Exception:
        return False


def list_images_recursive(root: Path, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


# --------------------- Geometry ---------------------
def clamp_and_expand_box(
    x: float, y: float, w: float, h: float, W: int, H: int, expand: float
) -> Tuple[int, int, int, int]:
    """Expand box around its center by `expand` and clamp to image bounds. Returns (L,T,R,B)."""
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    ww = w * expand
    hh = h * expand
    L = int(round(cx - 0.5 * ww))
    T = int(round(cy - 0.5 * hh))
    R = int(round(cx + 0.5 * ww))
    B = int(round(cy + 0.5 * hh))
    L = max(0, L)
    T = max(0, T)
    R = min(W, R)
    B = min(H, B)
    if R <= L or B <= T:
        return 0, 0, 0, 0
    return L, T, R, B


def rects_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """Return True if rectangles (L,T,R,B) intersect with positive area."""
    L1, T1, R1, B1 = a
    L2, T2, R2, B2 = b
    inter_w = max(0, min(R1, R2) - max(L1, L2))
    inter_h = max(0, min(B1, B2) - max(T1, T2))
    return inter_w > 0 and inter_h > 0


# --------------------- COCO utilities ---------------------
def load_coco(coco_root: Path, split: str) -> Tuple[COCO, Path]:
    ann = coco_root / "annotations" / f"instances_{split}2017.json"
    if not ann.exists():
        raise SystemExit(f"Missing COCO annotations: {ann}")
    coco = COCO(str(ann))
    img_dir = coco_root / f"{split}2017"
    if not img_dir.exists():
        raise SystemExit(f"Missing COCO images dir: {img_dir}")
    return coco, img_dir


def get_person_cat_id(coco: COCO) -> int:
    cats = coco.loadCats(coco.getCatIds(catNms=["person"]))
    if not cats:
        raise SystemExit("Could not find 'person' category in COCO.")
    return cats[0]["id"]


def coco_img_has_person(coco: COCO, img_id: int, person_cat_id: int) -> bool:
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[person_cat_id], iscrowd=None)
    return len(ann_ids) > 0


def load_facet_exclude_basenames(facet_csv: Optional[Path]) -> set[str]:
    """Return set of basenames to exclude from COCO -> to avoid FACET leakage."""
    if facet_csv is None or not facet_csv.exists():
        return set()
    import pandas as pd

    df = pd.read_csv(facet_csv)
    col = None
    for candidate in ("image_path", "img_path", "path"):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        return set()
    return {Path(p).name for p in df[col].dropna().astype(str).tolist()}


# --------------------- Negatives from COCO ---------------------
def sample_square_sizes_from_distribution(
    pos_sizes: Sequence[int], n: int, lo: int = 64, hi: int = 512
) -> List[int]:
    """
    Draw sizes (edge length) for negative crops.
    If we have a positive distribution, sample from it; else fallback to [lo, hi].
    """
    if len(pos_sizes) == 0:
        return [random.randint(lo, hi) for _ in range(n)]
    # Sample with replacement from positive min(side) sizes (clip to [lo,hi])
    pool = [int(max(lo, min(hi, s))) for s in pos_sizes]
    return [int(random.choice(pool)) for _ in range(n)]


def sample_random_square_within(W: int, H: int, size: int) -> Tuple[int, int, int, int]:
    if size > W or size > H:
        size = min(W, H)
    if W == size:
        x0 = 0
    else:
        x0 = random.randint(0, W - size)
    if H == size:
        y0 = 0
    else:
        y0 = random.randint(0, H - size)
    return (x0, y0, x0 + size, y0 + size)


# --------------------- Main builder ---------------------
def build_split(
    split: str,
    coco_root: Path,
    out_root: Path,
    neg_mode: str,
    neg_root: Optional[Path],
    expand: float,
    min_box: int,
    max_per_image: int,
    neg_max_per_image: int,
    neg_ratio_hard_bg: float,
    seed: int,
    facet_exclude_csv: Optional[Path],
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    coco, img_dir = load_coco(coco_root, split)
    person_id = get_person_cat_id(coco)
    facet_excl = load_facet_exclude_basenames(facet_exclude_csv)

    out_pos = out_root / split / "person"
    out_neg = out_root / split / "non_person"
    ensure_dir(out_pos)
    ensure_dir(out_neg)

    # --------- PASS 1: positives (person crops) ---------
    # Collect:
    # - saved positives (for balancing)
    # - per-image person boxes (for hard background negatives)
    # - distribution of positive crop sizes (min side) for negative sizing
    per_image_boxes: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    pos_sizes: List[int] = []
    n_pos_saved = 0

    img_ids = coco.getImgIds()
    # Optional: exclude images that show up in FACET
    if facet_excl:
        img_ids = [i for i in img_ids if coco.loadImgs([i])[0]["file_name"] not in facet_excl]

    pbar = tqdm(img_ids, desc=f"[{split}] COCO person crops")
    for img_id in pbar:
        anns = coco.getAnnIds(imgIds=[img_id], catIds=[person_id], iscrowd=None)
        if not anns:
            continue
        anns_data = coco.loadAnns(anns)
        # cap per-image positives
        per_image_count = 0
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        img_path = img_dir / file_name
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img.size

        # deterministic per-image order
        for ann in sorted(anns_data, key=lambda a: a.get("id", 0)):
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = float(ann["bbox"][0]), float(ann["bbox"][1]), float(ann["bbox"][2]), float(ann["bbox"][3])
            if min(w, h) < float(min_box):
                continue
            L, T, R, B = clamp_and_expand_box(x, y, w, h, W, H, expand)
            if R <= L or B <= T:
                continue

            # save crop
            out_name = f"{img_id}_{int(ann.get('id', 0))}.jpg"
            out_path = out_pos / out_name
            ok = save_crop(img, (L, T, R, B), out_path)
            if not ok:
                continue
            n_pos_saved += 1
            per_image_boxes[img_id].append((L, T, R, B))
            pos_sizes.append(min(R - L, B - T))

            per_image_count += 1
            if per_image_count >= max_per_image:
                break

    # --------- PASS 2: negatives (balanced to #positives) ---------
    n_negs_target = n_pos_saved
    n_negs_saved = 0

    if n_negs_target == 0:
        print(f"[{split}] No positives saved. Skipping negatives.")
        return

    if neg_mode == "places":
        if neg_root is None or not neg_root.exists():
            raise SystemExit(f"[{split}] neg_mode=places, but --neg_root not found: {neg_root}")
        imgs = list_images_recursive(neg_root)
        if not imgs:
            raise SystemExit(f"[{split}] No images found under {neg_root}")
        random.shuffle(imgs)
        # We will write one crop per negative file (center-crop-like random)
        # until reaching the target count.
        i = 0
        with tqdm(total=n_negs_target, desc=f"[{split}] Places negatives") as pbar:
            while n_negs_saved < n_negs_target and i < len(imgs):
                p = imgs[i]
                i += 1
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    continue
                W, H = im.size
                # sample size from positive dist for better match
                s = sample_square_sizes_from_distribution(pos_sizes, 1)[0]
                L, T, R, B = sample_random_square_within(W, H, s)
                out_name = f"pl_{i}_{L}_{T}_{R}_{B}.jpg"
                out_path = out_neg / out_name
                ok = save_crop(im, (L, T, R, B), out_path)
                if ok:
                    n_negs_saved += 1
                    pbar.update(1)

    elif neg_mode == "coco_only":
        # Split target into image-level (no-person images) and hard background
        n_hard = int(round(n_negs_target * float(neg_ratio_hard_bg)))
        n_image_lvl = n_negs_target - n_hard

        # ---------- (A) Image-level negatives: COCO images with **no person** ----------
        all_img_ids = coco.getImgIds()
        # Exclude any images we used in positives (to keep sources disjoint), and any FACET-excluded
        used_pos_imgs = set(per_image_boxes.keys())
        no_person_ids = []
        for iid in all_img_ids:
            if iid in used_pos_imgs:
                continue
            info = coco.loadImgs([iid])[0]
            if facet_excl and info["file_name"] in facet_excl:
                continue
            if not coco_img_has_person(coco, iid, person_id):
                no_person_ids.append(iid)
        random.shuffle(no_person_ids)

        # Target sizes drawn from positive distribution
        sizes_A = sample_square_sizes_from_distribution(pos_sizes, n_image_lvl)

        A_saved = 0
        with tqdm(total=n_image_lvl, desc=f"[{split}] COCO negatives (no-person images)") as pbar:
            img_ptr = 0
            size_ptr = 0
            while A_saved < n_image_lvl and img_ptr < len(no_person_ids):
                iid = no_person_ids[img_ptr]
                img_ptr += 1
                info = coco.loadImgs([iid])[0]
                p = img_dir / info["file_name"]
                if not p.exists():
                    continue
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    continue
                W, H = im.size
                per_im_saved = 0
                # Save up to neg_max_per_image crops (or until we fill the quota)
                for _ in range(neg_max_per_image):
                    if A_saved >= n_image_lvl or size_ptr >= len(sizes_A):
                        break
                    s = sizes_A[size_ptr]
                    size_ptr += 1
                    L, T, R, B = sample_random_square_within(W, H, s)
                    out_name = f"cocoNP_{iid}_{L}_{T}_{R}_{B}.jpg"
                    ok = save_crop(im, (L, T, R, B), out_neg / out_name)
                    if ok:
                        A_saved += 1
                        per_im_saved += 1
                        pbar.update(1)
                # continue to next image

        # ---------- (B) Hard background negatives: from images **with persons**, avoid overlap ----------
        sizes_B = sample_square_sizes_from_distribution(pos_sizes, n_hard)
        person_img_ids = list(per_image_boxes.keys())
        random.shuffle(person_img_ids)

        B_saved = 0
        with tqdm(total=n_hard, desc=f"[{split}] COCO negatives (hard bg)") as pbar:
            img_ptr = 0
            size_ptr = 0
            while B_saved < n_hard and img_ptr < len(person_img_ids):
                iid = person_img_ids[img_ptr]
                img_ptr += 1
                info = coco.loadImgs([iid])[0]
                p = img_dir / info["file_name"]
                if not p.exists():
                    continue
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    continue
                W, H = im.size
                boxes = per_image_boxes[iid]  # list of (L,T,R,B) for person crops

                per_im_saved = 0
                trials = 0
                # Try a reasonable number of random proposals per image
                while per_im_saved < neg_max_per_image and B_saved < n_hard and size_ptr < len(sizes_B) and trials < 200:
                    s = sizes_B[size_ptr]
                    size_ptr += 1
                    L, T, R, B = sample_random_square_within(W, H, s)
                    candidate = (L, T, R, B)
                    # reject if intersects any person crop
                    bad = False
                    for bb in boxes:
                        if rects_intersect(candidate, bb):
                            bad = True
                            break
                    trials += 1
                    if bad:
                        continue
                    out_name = f"cocoBG_{iid}_{L}_{T}_{R}_{B}.jpg"
                    ok = save_crop(im, candidate, out_neg / out_name)
                    if ok:
                        B_saved += 1
                        per_im_saved += 1
                        pbar.update(1)

        n_negs_saved = A_saved + B_saved

    else:
        raise SystemExit(f"Unknown --neg_mode '{neg_mode}'. Use 'places' or 'coco_only'.")

    print(f"→ {split}: saved {n_pos_saved} positives and {n_negs_saved} negatives at {out_root / split}")


# --------------------- CLI ---------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build PERSON/NON-PERSON ROI-crop dataset (COCO + optional Places).")
    ap.add_argument("--coco_root", type=str, required=True,
                    help="Folder containing train2017/, val2017/, annotations/instances_*.json")
    ap.add_argument("--neg_root", type=str, default=None,
                    help="Root of Places365 (used only when --neg_mode=places).")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for person/non_person crops.")
    ap.add_argument("--expand", type=float, default=1.25,
                    help="Box expansion factor around COCO bbox center for positives.")
    ap.add_argument("--min_box", type=int, default=32,
                    help="Skip positives whose min(w,h) < min_box (before expansion).")
    ap.add_argument("--max_per_image", type=int, default=8,
                    help="Cap positives per COCO image (prevents crowd oversampling).")
    ap.add_argument("--neg_mode", type=str, choices=["places", "coco_only"], default="coco_only",
                    help="Source of NON-PERSON negatives.")
    ap.add_argument("--neg_max_per_image", type=int, default=8,
                    help="Max negative crops per image (for coco_only).")
    ap.add_argument("--neg_ratio_hard_bg", type=float, default=0.5,
                    help="For coco_only: fraction of negatives from hard background (rest from no-person images).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--facet_exclude_csv", type=str, default=None,
                    help="Optional CSV from FACET with an image_path column; matches basenames to exclude (avoid leakage).")
    return ap.parse_args()


def main():
    args = parse_args()
    coco_root = Path(args.coco_root)
    out_root = Path(args.out_dir)
    neg_root = Path(args.neg_root) if args.neg_root else None
    facet_csv = Path(args.facet_exclude_csv) if args.facet_exclude_csv else None

    # Build splits
    build_split(
        split="train",
        coco_root=coco_root,
        out_root=out_root,
        neg_mode=args.neg_mode,
        neg_root=neg_root,
        expand=args.expand,
        min_box=args.min_box,
        max_per_image=args.max_per_image,
        neg_max_per_image=args.neg_max_per_image,
        neg_ratio_hard_bg=args.neg_ratio_hard_bg,
        seed=args.seed,
        facet_exclude_csv=facet_csv,
    )
    build_split(
        split="val",
        coco_root=coco_root,
        out_root=out_root,
        neg_mode=args.neg_mode,
        neg_root=neg_root,
        expand=args.expand,
        min_box=args.min_box,
        max_per_image=args.max_per_image,
        neg_max_per_image=args.neg_max_per_image,
        neg_ratio_hard_bg=args.neg_ratio_hard_bg,
        seed=args.seed + 1,  # slightly different stream for val
        facet_exclude_csv=facet_csv,
    )


if __name__ == "__main__":
    main()
