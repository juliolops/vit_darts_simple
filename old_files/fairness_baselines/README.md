# Fairness Baselines (Person/Face vs Non-*): Training & Evaluation Guide

This guide explains how to build datasets, train baseline binary classifiers, and evaluate **fairness** on **FACET (skin tone)** and **FairFace (race)**.

---

## What you’ll get
- **Two dataset options** for training 2‑class classifiers:
  - **FACE vs NON‑FACE** (WIDER Face + Places365) → `facebin_data/`
  - **PERSON vs NON‑PERSON** (COCO) → `cocobin_data/`
- **Torchvision-only** baselines: `resnet18`, `resnet50`, `efficientnet_v2_s`, `convnext_tiny`, `mobilenet_v3_large`, `mnasnet1_0`
- **Fairness** evaluation:
  - by **skin tone** on **FACET** (hard **and** soft labeling supported)
  - by **race** on **FairFace**

> **TL;DR flow**: build (or reuse) one of the binary datasets → train baselines → build FACET eval CSV → evaluate on FACET (skin tone) → optionally evaluate on FairFace (race).

---

## Repo layout (relevant files)

```
fairness/
  download_wider_places.py            # WIDER (faces) + Places365 (negatives) downloader
  build_face_binary_wider_places.py   # Build FACE/NON-FACE dataset from WIDER + Places
  build_person_binary_coco_places.py  # (optional) Build PERSON/NON-PERSON from COCO + Places
  train_facebin_models.py             # Torchvision-only baselines trainer (binary)
  build_facet_dataset.py         # Build FACET eval CSV (hard & soft skin-tone labels)
  eval_facet_skintone.py              # Evaluate fairness on FACET by skin tone
  download_fairface.py                # Download/prepare FairFace (race) validation CSV
  eval_fairface_race.py               # Evaluate fairness on FairFace by race
  download_coco_subset.sh             # Download/prepare COCO subset
```

You also have your prebuilt **COCO person/non-person** dataset under `cocobin_data/`.

---

## Environment

```bash
# create & activate a conda env (example)
conda create -n moqnas python=3.10 -y
conda activate moqnas

# core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pillow pandas tqdm scikit-learn

# dataset-specific deps
pip install pycocotools # for COCO
pip install datasets huggingface_hub # for FairFace downloader variant
```

> If you use a different CUDA/PyTorch build, adjust the torch index‑url accordingly. CPU‑only is also fine; it’ll just be slower.

---

## Option A — FACE vs NON‑FACE (WIDER + Places365)

### 1) Download datasets
This saves WIDER under `data/WIDER` and Places (val split) under `data/PLACES365/val`.

```bash
python fairness/download_wider_places.py
```

### 2) Build the binary dataset
This crops faces from WIDER and samples balanced negatives from Places into `facebin_data/`:

```bash
python fairness/build_face_binary_wider_places.py \
  --wider_root data/WIDER \
  --neg_root   data/PLACES365/val \
  --out_dir    facebin_data
```

**Resulting structure**:

```
facebin_data/
  train/
    face/*.jpg
    non_face/*.jpg
  val/
    face/*.jpg
    non_face/*.jpg
```

### 3) Train baselines
The trainer auto-detects `face/non_face` folders. Pick your GPU with `--device`.

```bash
python fairness/train_facebin_models.py \
  --data_root facebin_data \
  --device cuda:0 \
  --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large,mnasnet1_0 \
  --epochs 10 --bs 128 --out_dir checkpoints_facebin
```

This saves checkpoints like `checkpoints_facebin/facebin_resnet18.pt` and writes a small CSV summary if `--results_csv` is set (default `checkpoints/baseline_results.csv`).

---

## Option B — PERSON vs NON‑PERSON (COCO + Places365)

> Build the dataset using `build_person_binary_coco_places.py`. 

### Build the dataset
```bash
./download_coco_subset.sh

python fairness/build_person_binary_coco_places.py \
    --coco_root data/COCO_sub \
    --neg_root  data/PLACES365 \
    --out_dir   data/personbin_data
```

**Structure**:
```
cocobin_data/
  train/
    person/*.jpg
    no_person/*.jpg
  val/
    person/*.jpg
    no_person/*.jpg
```

### Train baselines on COCO personbin
Pass the folder names explicitly (or rely on autodetect).

```bash
python fairness/train_facebin_models.py \
  --data_root data/personbin_data \
  --pos_name person --neg_name non_person \
  --device cuda:1 \
  --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large \
  --epochs 10 --bs 128 --out_dir checkpoints_personbin
```

---

## Build FACET evaluation CSV (skin tone)

FACET provides **person** boxes and **multi‑annotator skin‑tone votes** across 10 bins (1..10). We need a CSV that points to the image, the **person bbox**, and the **skin‑tone label(s)**.

The builder below creates:
- **Hard** label column `skin_tone_final` using a configurable tie strategy.
- **Soft** label column `skin_tone_probs` as a length‑10 JSON of vote probabilities.
- BBox columns: `x,y,width,height`
- `image_path` resolved from provided image directories.

```bash
python fairness/build_facet_dataset.py \
    --ann_csv facet_data/annotations/annotations.csv \
    --img_dirs facet_data/imgs_1 facet_data/imgs_2 \
    --out_csv facet_data/facet_eval.csv \
    --hard_strategy median_round \
    --visible_face_col visible_face
```

**Tie strategies** (`--hard_strategy`):
- `median_round` (recommended; weighted median of votes, then round)
- `mode_high` (pick highest among tied maxima)
- `mode_low` (lowest among tied maxima)
- `mode_random` (random among ties; use `--seed`)
- `drop_ties` (omit rows with ties at the top)
- `none` (no hard label; soft only)

---

## Evaluate fairness on FACET (skin tone)

We evaluate **TPR per skin tone** on **FACET person crops** using your binary classifier. You can report **hard** (from `skin_tone_final`) and/or **soft** (weighted by `skin_tone_probs`) fairness.

Fairness metric:
- `SPD_sum = Σ_g (TPR_g − min_g TPR_g)`
- `Fairness = max(0, (β − SPD_sum)/β)` with default `β = 0.2`

### Single checkpoint
```bash
python fairness/eval_facet_skintone.py \
  --facet_csv facet_data/facet_eval.csv \
  --ckpt checkpoints_facebin/facebin_resnet18.pt \
  --mode both --beta 0.2 \
  --out_json fairness/facet_resnet18.json
```

### All checkpoints in a folder
```bash
python fairness/eval_facet_skintone.py \
  --facet_csv facet_data/facet_eval.csv \
  --ckpt_dir checkpoints_facebin \
  --mode both --beta 0.2 \
  --out_json fairness/facet_facebin_results.json
```

> Output is a JSON with per‑tone TPRs, counts/weights, and the fairness summary for each model.

---

## (Optional) Evaluate on FairFace (race)

This checks TPR across **race** groups to complement FACET’s skin tone fairness.

### 1) Download/prepare FairFace
```bash
python fairness/download_fairface.py
# produces: data/FairFace/0.25/fairface_val.csv  (and image folder)
```

### 2) Evaluate checkpoints by race
```bash
python fairness/eval_fairface_race.py \
  --fairface_csv data/FairFace/0.25/fairface_val.csv \
  --ckpt_dir checkpoints_facebin \
  --out_json fairness/fairface_facebin_results.json
```

Or point it at `checkpoints_coco_personbin` if training on COCO person/non‑person.

---

## Recommended protocol (to stay unbiased & reproducible)

1. **Don’t train on FACET**. Use it only for test‑time fairness.  
2. **Use a single decision rule** across all groups (we use argmax). Don’t per‑group tune thresholds.  
3. **Crop consistently**: FACET’s person bounding boxes are used at evaluation (apples‑to‑apples if you trained on COCO person crops).  
4. **Report both hard & soft fairness** for FACET to make tie handling explicit.  
5. **Show per‑group counts** and, ideally, CIs/bootstraps (tone/race are imbalanced).

---

## JSON outputs (shape)

**FACET (`eval_facet_skintone.py`)**:
```json
{
  "metadata": {
    "facet_csv": "/abs/path/facet_eval.csv",
    "mode": "both",
    "beta": 0.2,
    "device": "cuda:0"
  },
  "results": [
    {
      "ckpt": "/abs/path/checkpoints_facebin/facebin_resnet18.pt",
      "arch": "resnet18",
      "hard": {
        "per_tone_tpr": {"1":"...","2":"...","10":"..."},
        "counts": {"1":123, "...":"..."},
        "overall_mean_tpr": 0.95,
        "metrics": {"min_group_tpr":0.90,"max_min_gap":0.08,"spd_sum":0.10,"fairness":0.50}
      },
      "soft": {
        "per_tone_tpr": {"1":"...","2":"...","10":"..."},
        "denom": {"1":122.7,"...":"..."},
        "overall_mean_tpr": 0.96,
        "metrics": {"min_group_tpr":0.91,"max_min_gap":0.07,"spd_sum":0.08,"fairness":0.60}
      }
    }
  ]
}
```

**FairFace (`eval_fairface_race.py`)**:
```json
{
  "metadata": {
    "fairface_csv": "/abs/path/fairface_val.csv",
    "beta": 0.2,
    "device": "cuda:0",
    "limit": "full"
  },
  "results": [
    {
      "ckpt": "/abs/path/checkpoints_facebin/facebin_resnet18.pt",
      "arch": "resnet18",
      "n_total": 11867,
      "per_race_tpr": {"White":0.98,"Black":0.96,"...":"..."},
      "per_race_counts": {"White":4000,"Black":1200,"...":"..."},
      "overall_mean_tpr": 0.97,
      "metrics": {"min_group_tpr":0.94,"max_min_gap":0.04,"spd_sum":0.06,"fairness":0.70}
    }
  ]
}
```

---

## Troubleshooting

- **WIDER download via gdown fails** (Google Drive quota / permissions): rerun later or manually download the WIDER archives to `data/_tmp_downloads/wider/` and let the script extract them; or install from alternate mirrors if available.
- **Places365 val not found after extract**: ensure `data/PLACES365/val/` exists and contains class subfolders with images. Re‑run the downloader with verbose logs.
- **`KeyError: 'mean'` / torchvision weights meta**: the trainer already falls back to ImageNet stats—no action needed.
- **Mixed precision deprecation warnings**: the trainer uses **`torch.amp`** (or falls back) — no warnings expected.
- **No images found**: check your folder names. The dataset loader expects `face/non_face` **or** `person/no_person`. You can force names via `--pos_name`/`--neg_name`.
- **Device selection**: pick `--device cuda:0` (or `cuda:1`, or `cpu`). Don’t rely on `CUDA_VISIBLE_DEVICES` at runtime.

---

## Quick commands (copy‑paste)

### Train on **FACE/NON‑FACE**
```bash
python fairness/train_facebin_models.py \
  --data_root facebin_data \
  --device cuda:0 \
  --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large \
  --epochs 10 --bs 128 --out_dir checkpoints_facebin
```

### Train on **PERSON/NON‑PERSON (COCO)** (reuse your `personbin_data/`)
```bash
python fairness/train_facebin_models.py \
  --data_root data/personbin_data \
  --pos_name person --neg_name no_person \
  --device cuda:1 \
  --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large \
  --epochs 10 --bs 128 --out_dir checkpoints_coco_personbin
```

### FACET fairness (skin tone)
```bash
# build CSV (once)
python fairness/build_facet_dataset.py \
    --ann_csv facet_data/annotations/annotations.csv \
    --img_dirs facet_data/imgs_1 facet_data/imgs_2 \
    --out_csv facet_data/facet_eval.csv \
    --hard_strategy median_round \
    --visible_face_col visible_face

# evaluate all checkpoints in a folder
python fairness/eval_facet_skintone.py \
  --facet_csv facet_data/facet_eval.csv \
  --ckpt_dir checkpoints_coco_personbin \
  --mode both --beta 0.2 \
  --out_json fairness/facet_facebin_results.json
```

### FairFace fairness (race)
```bash
# download/prepare (once)
python fairness/download_fairface.py

# evaluate all checkpoints in a folder
python fairness/eval_fairface_race.py \
  --fairface_csv data/FairFace/0.25/fairface_val.csv \
  --ckpt_dir checkpoints_facebin \
  --device cuda:1 \
  --out_json fairness/fairface_facebin_results.json
```

---

## Notes on interpretation

- FACET fairness focuses on **missed detections** across **skin tone** bins. Because tones exist only for *people*, we report **TPR parity** (not FPR).
- Use the **same decision rule** globally (we use argmax). Per‑group thresholds can hide bias.
- Report both **hard** and **soft** FACET fairness. Soft fairness avoids arbitrary tie‑breaking when annotators disagree on tone.
- If you trained on **COCO person crops**, evaluating on **FACET person crops** keeps domains aligned (person ROI in both). If you trained on **WIDER face crops**, you can still evaluate on FACET person crops—just be aware of potential context differences.

---

## License & attribution

- WIDER Face, Places365, COCO, FACET, and FairFace are owned by their respective authors/organizations. Follow their licenses/datasheets.
- This code uses **PyTorch** + **torchvision** (no `timm`) and standard Python libs.

Happy training & testing ✨
