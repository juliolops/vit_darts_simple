# moq-nas/core/fairness/build.py

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

# Ensure pycocotools is installed for COCO processing
try:
    from pycocotools.coco import COCO
except ImportError:
    print("Warning: pycocotools not found. Please install it (`pip install pycocotools`) to use COCO building functions.")
    COCO = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    _RESAMPLING_SPACE = getattr(Image, "Resampling", Image)
    RESAMPLE_BICUBIC = getattr(_RESAMPLING_SPACE, "BICUBIC", Image.BICUBIC)
    RESAMPLE_LANCZOS = getattr(
        _RESAMPLING_SPACE, "LANCZOS", getattr(Image, "ANTIALIAS", Image.BICUBIC)
    )
except Exception:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_LANCZOS = getattr(Image, "ANTIALIAS", Image.BICUBIC)

# --- General Helper Functions ---

def _ensure_dir(p: Path):
    """Creates a directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)

def _save_crop(img: Image.Image, box: Tuple[int, int, int, int], out_path: Path) -> bool:
    """Saves a cropped region of an image, returns True on success."""
    try:
        crop = img.crop(box)
        crop.save(out_path, format="JPEG", quality=90)
        return True
    except Exception:
        return False

# --- 1. FACET CSV Builder Functions -------------------------------------------

def build_facet_csv(ann_csv: str, img_dirs: List[str], out_csv: str):
    """
    Processes raw FACET annotations to create a standardized evaluation CSV.
    """
    print("Building standardized FACET evaluation CSV...")
    df = pd.read_csv(ann_csv)
    path_map = {p.name: str(p) for dir_path in img_dirs for p in Path(dir_path).glob("*") if p.is_file()}
    df["image_path"] = df["filename"].map(path_map)
    df.dropna(subset=["image_path"], inplace=True)

    hard_labels, soft_probs = [], []
    skin_tone_cols = [f"skin_tone_{i}" for i in range(1, 11)]
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing FACET annotations"):
        counts = [int(row.get(col, 0)) for col in skin_tone_cols]
        total_votes = sum(counts)
        probs = [c / total_votes for c in counts] if total_votes > 0 else [0.0] * 10
        # Simple hard label: highest vote, ties broken by highest tone index
        hard_label = np.argmax(counts) + 1 if total_votes > 0 else None
        soft_probs.append(json.dumps(probs))
        hard_labels.append(hard_label)

    bbox_data = df["bounding_box"].apply(json.loads)
    df_out = pd.DataFrame({
        "image_path": df["image_path"],
        "x": bbox_data.apply(lambda b: b.get("x")), "y": bbox_data.apply(lambda b: b.get("y")),
        "width": bbox_data.apply(lambda b: b.get("width")), "height": bbox_data.apply(lambda b: b.get("height")),
        "skin_tone_final": hard_labels, "skin_tone_probs": soft_probs,
    })
    df_out.dropna(inplace=True)
    df_out["skin_tone_final"] = df_out["skin_tone_final"].astype(int)
    _ensure_dir(Path(out_csv).parent)
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Successfully created FACET CSV with {len(df_out)} rows at: {out_csv}")

# --- 2. Face Binary Dataset Builder (from WIDER) ------------------------------

def _read_wider_annotations(txt_path: str) -> List[Tuple[str, List[Tuple[int, int, int, int]]]]:
    """Robustly parses WIDER Face annotation files."""
    items = []
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines):
        if not lines[i].lower().endswith(".jpg"): i += 1; continue
        path = lines[i]; i += 1
        num_boxes_line = lines[i]; i += 1
        try:
            num_boxes = int(num_boxes_line)
            box_data = lines[i:i+num_boxes]; i += num_boxes
        except ValueError: # Handles format without explicit box count
            box_data = [num_boxes_line]
            while i < len(lines) and not lines[i].lower().endswith(".jpg"):
                box_data.append(lines[i]); i += 1

        boxes = []
        for line in box_data:
            parts = [int(p) for p in line.split()[:4]]
            if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                boxes.append(tuple(parts))
        if boxes: items.append((path, boxes))
    return items

def _build_face_binary_split(split: str, wider_root: Path, neg_pool: List[Path], out_dir: Path, **kwargs):
    """Processes one split (train/val) for the face binary dataset."""
    ann_file = wider_root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    img_dir = wider_root / f"WIDER_{split}" / "images"
    annotations = _read_wider_annotations(str(ann_file))

    out_pos = out_dir / split / "face"; _ensure_dir(out_pos)
    out_neg = out_dir / split / "non_face"; _ensure_dir(out_neg)
    
    pos_count = 0
    for img_path_str, boxes in tqdm(annotations, desc=f"[{split}] Building face dataset"):
        img_path = img_dir / img_path_str
        if not img_path.exists(): continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception: continue
        
        for i, (x, y, w, h) in enumerate(boxes):
            if w < kwargs.get('min_face', 20) or h < kwargs.get('min_face', 20): continue
            
            # Save positive crop
            box = (x, y, x + w, y + h)
            out_pos_path = out_pos / f"{img_path.stem}_{i}.jpg"
            if _save_crop(img, box, out_pos_path):
                pos_count += 1
    
    # Generate balanced negatives
    print(f"[{split}] Found {pos_count} positive face samples. Generating balanced negatives...")
    neg_count = 0
    pbar_neg = tqdm(total=pos_count, desc=f"[{split}] Sampling non-face negatives")
    while neg_count < pos_count:
        neg_src_path = random.choice(neg_pool)
        try:
            neg_img = Image.open(neg_src_path).convert("RGB")
            nW, nH = neg_img.size
            # Use a random size for more diversity
            tw, th = random.randint(32, 256), random.randint(32, 256)
            if nW <= tw or nH <= th: continue
            
            nx0, ny0 = random.randint(0, nW - tw), random.randint(0, nH - th)
            box = (nx0, ny0, nx0 + tw, ny0 + th)
            out_neg_path = out_neg / f"{neg_src_path.stem}_{random.randint(0,99999)}.jpg"
            if _save_crop(neg_img, box, out_neg_path):
                neg_count += 1
                pbar_neg.update(1)
        except Exception: continue
    pbar_neg.close()
    print(f"✅ [{split}] split built: {pos_count} faces, {neg_count} non-faces.")

def build_face_binary_dataset(wider_root: str, neg_root: str, out_dir: str, **kwargs):
    """Builds a FACE vs NON_FACE dataset from WIDER Face and a negative source."""
    wider_root, neg_root, out_dir = Path(wider_root), Path(neg_root), Path(out_dir)
    print(f"Building FACE/NON_FACE dataset at: {out_dir}")
    random.seed(kwargs.get('seed', 42))
    
    print("Collecting negative image pool...")
    neg_pool = list(neg_root.glob("**/*.jpg")) + list(neg_root.glob("**/*.png"))
    if not neg_pool: raise FileNotFoundError(f"No negative images (.jpg, .png) found in {neg_root}")

    for split in ['train', 'val']:
        _build_face_binary_split(split, wider_root, neg_pool, out_dir, **kwargs)

# --- 3. Person Binary Dataset Builder (from COCO) -----------------------------

def _build_person_binary_split(split: str, coco_root: Path, out_dir: Path, **kwargs):
    """
    Builds one split (train/val) for the PERSON vs NON_PERSON dataset, balanced 1:1.

    Positives:
        - Cap at most 2 crops per image.
        - Pad bbox ~15% for context, clip to image bounds.
        - Keep only "person-like" boxes via geometry filters:
              * min size: min(w,h) >= 64
              * aspect ratio: 0.35 <= w/h <= 2.5
              * area fraction: (w*h)/(W*H) >= 0.01
            * relative height in frame: h/H >= 0.35
        - (Optional) If keypoints exist, require >= 4 labeled keypoints.

    Negatives:
        - Random background crop from non-person images; keep sampling until 1:1.

    Notes:
        - Resizing to fixed squares (e.g., 96×96) should be done later with your letterbox exporter.
    """
    import random

    random.seed(kwargs.get('seed', 42))

    ann_file = coco_root / "annotations" / f"instances_{split}2017.json"
    img_dir = coco_root / f"{split}2017"
    coco = COCO(str(ann_file))
    person_cat_id = coco.getCatIds(catNms=['person'])[0]

    img_ids = coco.getImgIds()
    person_img_ids = coco.getImgIds(catIds=[person_cat_id])
    non_person_img_ids = list(set(img_ids) - set(person_img_ids))
    random.shuffle(non_person_img_ids)

    out_pos = out_dir / split / "person"; _ensure_dir(out_pos)
    out_neg = out_dir / split / "non_person"; _ensure_dir(out_neg)

    def _clip_box(x0, y0, x1, y1, W, H):
        x0 = max(0, min(int(x0), W - 1))
        y0 = max(0, min(int(y0), H - 1))
        x1 = max(0, min(int(x1), W - 1))
        y1 = max(0, min(int(y1), H - 1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    # ---------------------------
    # 1) Positives (cap and filters)
    # ---------------------------
    pos_count = 0
    for img_id in tqdm(person_img_ids, desc=f"[{split}] Cropping person positives"):
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[person_cat_id], iscrowd=False)
        if not ann_ids:
            continue

        img_info = coco.loadImgs([img_id])[0]
        img_path = img_dir / img_info['file_name']
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        W, H = img.size
        anns = coco.loadAnns(ann_ids)

        # Hard cap: at most 2 person crops per image
        if len(anns) > 2:
            anns = random.sample(anns, 2)

        for i, ann in enumerate(anns):
            x, y, w, h = ann['bbox']

            # Skip clearly tiny boxes first (preserve your original guard)
            if w < kwargs.get('min_person_w', 32) or h < kwargs.get('min_person_h', 32):
                continue

            # 1) Padding (~15%) around bbox center
            px = 0.15 * w
            py = 0.15 * h
            cx = x + w / 2.0
            cy = y + h / 2.0
            x0 = cx - (w / 2.0 + px)
            y0 = cy - (h / 2.0 + py)
            x1 = cx + (w / 2.0 + px)
            y1 = cy + (h / 2.0 + py)

            # 2) Clip to image bounds
            clipped = _clip_box(x0, y0, x1, y1, W, H)
            if clipped is None:
                continue
            x0, y0, x1, y1 = clipped
            pw, ph = x1 - x0, y1 - y0

            # 3) Geometry-based filters (favor whole/upper-body)
            if pw < 64 or ph < 64:
                continue
            ar = pw / float(ph)
            if ar < 0.35 or ar > 2.5:
                continue
            area_frac = (pw * ph) / float(W * H)
            if area_frac < 0.01:
                continue
            rel_h = ph / float(H)
            if rel_h < 0.35:
                continue

            # 4) Optional keypoints filter (if available)
            kpts = ann.get('keypoints')
            if kpts:
                # COCO keypoints: [x1,y1,v1, x2,y2,v2, ...]; v>0 labeled, v>1 visible
                visible = sum(1 for i_k in range(2, len(kpts), 3) if kpts[i_k] > 0)
                if visible < 4:
                    continue

            # Save positive crop
            box = (x0, y0, x1, y1)
            out_pos_path = out_pos / f"{img_path.stem}_{i}.jpg"
            if _save_crop(img, box, out_pos_path):
                pos_count += 1

    # ---------------------------
    # 2) Negatives (1:1 balance)
    # ---------------------------
    target_neg = pos_count
    neg_count = 0
    pbar_neg = tqdm(total=target_neg, desc=f"[{split}] Sampling non-person negatives")

    img_ptr = 0
    while neg_count < target_neg:
        if img_ptr >= len(non_person_img_ids):
            img_ptr = 0
            random.shuffle(non_person_img_ids)

        np_img_id = non_person_img_ids[img_ptr]
        img_ptr += 1

        img_info = coco.loadImgs([np_img_id])[0]
        img_path = img_dir / img_info['file_name']
        if not img_path.exists():
            continue

        try:
            neg_img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        nW, nH = neg_img.size
        if nW < 64 or nH < 64:
            continue

        # single random crop per iteration; cap crop size by image dims
        tw = random.randint(64, min(224, nW))
        th = random.randint(64, min(224, nH))
        if nW <= tw or nH <= th:
            continue

        nx0 = random.randint(0, nW - tw)
        ny0 = random.randint(0, nH - th)
        box = (nx0, ny0, nx0 + tw, ny0 + th)

        out_neg_path = out_neg / f"{img_path.stem}_{random.randint(0, 99999)}.jpg"
        if _save_crop(neg_img, box, out_neg_path):
            neg_count += 1
            pbar_neg.update(1)

    pbar_neg.close()
    print(f"✅ [{split}] split built: {pos_count} persons, {neg_count} non-persons.")


def build_person_binary_dataset(coco_root: str, out_dir: str, **kwargs):
    """Builds a PERSON vs NON_PERSON binary dataset from COCO."""
    if COCO is None:
        raise ImportError("Please install pycocotools (`pip install pycocotools`) to build datasets from COCO.")
    
    print(f"Building PERSON/NON_PERSON dataset at: {out_dir}")
    for split in ['train', 'val']:
        _build_person_binary_split(split, Path(coco_root), Path(out_dir), **kwargs)

# --- 4) Make a square resized mirror of an existing dataset -------------------

_CLASS_DIRS = {"person", "non_person", "face", "non_face"}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def _letterbox_square(img: Image.Image, fill=(0,0,0)) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill)
    out.paste(img, ((side - w)//2, (side - h)//2))
    return out

def _resize_and_save(src_path: Path, dst_path: Path, target: int, mode: str, quality: int) -> bool:
    try:
        im = Image.open(src_path).convert("RGB")
    except Exception:
        return False

    if mode == "center_crop":
        im = _center_crop_square(im)
    elif mode == "letterbox":
        im = _letterbox_square(im)
    else:
        raise ValueError("mode must be 'center_crop' or 'letterbox'")

    # Use compat resample constant (works for old/new Pillow)
    im = im.resize((target, target), RESAMPLE_BICUBIC)

    dst_path = dst_path.with_suffix(".jpg")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        im.save(dst_path, "JPEG", quality=quality, optimize=True)
        return True
    except Exception:
        return False

def build_square_resized_version(src_root: str, dst_root: str, target: int = 96, mode: str = "letterbox", quality: int = 90):
    """
    Mirror an existing split/class folder dataset (train/val with person/non_person or face/non_face)
    into a new root, resizing everything to target x target.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    for split in ("train", "val"):
        split_dir = src_root / split
        if not split_dir.exists():
            continue
        for class_dir in (d for d in split_dir.iterdir() if d.is_dir() and d.name in _CLASS_DIRS):
            imgs = [p for p in class_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS]
            for src_img in tqdm(imgs, desc=f"[{split}] {class_dir.name} -> {target}x{target}"):
                rel = src_img.relative_to(split_dir)        # keep any nested structure
                dst_img = (dst_root / split / rel).with_suffix(".jpg")
                _resize_and_save(src_img, dst_img, target=target, mode=mode, quality=quality)

    print(f"✅ Resized dataset at: {dst_root} (size={target}x{target}, mode={mode})")
