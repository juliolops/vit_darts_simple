# MoQ-NAS: Multi-Objective Quantized Neural Architecture Search

This repository contains the official implementation of **MoQ-NAS**, a framework for finding hardware‑efficient and quantized neural network architectures using multi‑objective evolutionary algorithms.

---

## Fairness Baselines Workflow

This section details the end‑to‑end workflow for training and evaluating baseline models on fairness metrics. The process is broken down into four main stages:

1. **Download Datasets**: Download all required raw datasets.
2. **Prepare Datasets**: Process the raw data into binary classification formats.
3. **Train Baseline Models**: Train standard architectures (e.g., ResNet) on the prepared datasets.
4. **Evaluate Fairness**: Measure fairness of the trained models on the FairFace and FACET benchmarks.

---

### 1) Download Raw Datasets

Run the download scripts to fetch the **COCO**, **WIDER Face**, **Places365**, and **FairFace** datasets. These scripts will place the data in the `data/` directory.

```bash
# Download COCO subset, WIDER Face, and Places365
bash scripts/download_datasets/download_coco_subset.sh
python scripts/download_datasets/download_wider_places.py

# Download the FairFace dataset
python scripts/download_datasets/download_fairface.py
```

---

### 2) Prepare Datasets for Training

Process the raw data into the binary **PERSON vs. NON-PERSON** and **FACE vs. NON-FACE** formats required for training. The output will be saved to the `datasets/` directory by default.

```bash
# Build the PERSON vs. NON-PERSON dataset from COCO
python scripts/fairness_baseline/prepare_data.py --build_person

# Build the FACE vs. NON-FACE dataset from WIDER Face and Places365
python scripts/fairness_baseline/prepare_data.py --build_face

# Build FACET evaluation CSV
python scripts/fairness_baseline/prepare_data.py --build_facet
```

---

### 2.1) Create Resized Square Mirrors

After building `personbin_data/` and `facebin_data/`, you can generate **square, resized** mirrors at any target size (keeping the same `train/val/{class}/...` structure).

Two square strategies are supported:

- `center_crop`: crops the center to square, then resizes (keeps subject centered).
- `letterbox` (default): pads to square with no content lost, then resizes.

The output directory defaults to `<src_dir>_<resize_target>` if not specified explicitly.

#### Option A — Build originals and resized mirrors in one go
```bash
# Build PERSON and FACE, then create 96×96 mirrors
python scripts/fairness_baseline/prepare_data.py \
  --build_person --person_out_dir datasets/personbin_data \
  --build_face   --face_out_dir   datasets/facebin_data \
  --make_person_resized --make_face_resized \
  --resize_target 96 --resize_mode letterbox --jpg_quality 90
```

#### Option B — Create mirrors from already-built roots
```bash
# Create a 96×96 mirror of the PERSON dataset
python scripts/fairness_baseline/prepare_data.py \
  --make_person_resized \
  --person_src_for_resize datasets/personbin_data \
  --resize_target 96 --resize_mode letterbox --jpg_quality 90
# → output: datasets/personbin_data_96/

# Create a 48×48 mirror from the existing 96×96 dataset
python scripts/fairness_baseline/prepare_data.py \
  --make_person_resized \
  --person_src_for_resize datasets/personbin_data_96 \
  --resize_target 48 --resize_mode letterbox --jpg_quality 90
# → output: datasets/personbin_data_48/

# Override the output directory explicitly
python scripts/fairness_baseline/prepare_data.py \
  --make_person_resized \
  --person_src_for_resize datasets/personbin_data \
  --person_resized_out_dir datasets/personbin_data_128 \
  --resize_target 128 --resize_mode letterbox --jpg_quality 90
```

**Resulting layout (same for every target size):**
```
datasets/personbin_data_<N>/
  train/{person,non_person}/...
  val/{person,non_person}/...

datasets/facebin_data_<N>/
  train/{face,non_face}/...
  val/{face,non_face}/...
```

> Notes:
> - `--resize_target` accepts any integer (e.g., 48, 64, 96, 128).
> - Adjust `--jpg_quality` (1–100) to trade disk size vs. fidelity (default: 90).
> - Each target size needs its own `dataset_configs/person_bin_<N>.yaml` and experiment config pointing to the matching `data_path`. See `dataset_configs/person_bin_48.yaml` and `dataset_configs/person_bin_96.yaml` as examples.



---

### 3) Train Baseline Models

This module trains baseline CNN models for the fairness experiments using the **centralized configuration system**.  
It relies on the same `GenericDataLoader` used by the evolutionary algorithms, guaranteeing **consistent data preprocessing and splits** across all experiments.

> **Tip:** Always run these commands from the project root directory, e.g. `~/MoQ-NAS`.


#### 3.1. Basic Usage

You must provide:

- `--config_file`: YAML configuration file defining the dataset.
- `--data_path`: Root folder where datasets are stored.
- `--experiment_path`: Output folder where logs and checkpoints will be saved.

### Example: Train Baselines on the PERSON Dataset

```bash
# Example: Train all avaible networks
python scripts/fairness_baseline/train.py \
    --config_file dataset_configs/person_bin_96.yaml \
    --data_path datasets/personbin_data_96 \
    --experiment_path checkpoints/baselines_head_96_limit \
    --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large,mnasnet1_0 \
    --max_epochs 10
```

### Example: Train Baselines on the FACE Dataset

```bash
# Example: Train models on the FACE dataset
python scripts/fairness_baseline/train.py \
    --config_file dataset_configs/face_bin_96.yaml \
    --data_path datasets/facebin_data_96 \
    --experiment_path checkpoints/baselines_head_96_limit \
    --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large,mnasnet1_0 \
    --max_epochs 10
```

#### 3.2. Advanced Options (Freezing & Data Limiting)

You can:

- **Freeze the backbone**, training only the classification head.
- **Limit the number of training images** for rapid debugging or data-efficiency studies.

### Example: Freeze Backbone and Limit Training to 10000 Images

```bash
# Example: Freeze backbone and limit training to 10000 images
python scripts/fairness_baseline/train.py \
    --config_file dataset_configs/person_bin_96.yaml \
    --data_path datasets/personbin_data_96 \
    --experiment_path checkpoints/baselines_head_96_limit \
    --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large,mnasnet1_0 \
    --max_epochs 1 \
    --freeze_backbone \
    --limit_data \
    --limit_data_value 10000
```

#### 3.3. Key Arguments

- `--config_file`  
  Path to the YAML config file (e.g., `dataset_configs/person_bin.yaml`).

- `--data_path`  
  Root directory containing your datasets (e.g., `datasets`).

- `--experiment_path`  
  Directory where checkpoints and logs will be created (e.g., `experiments/baselines_person`).

- `--archs`  
  Comma-separated list of architectures to train  
  (must be available in `core/fairness/models.py`), e.g.:  
  `resnet18,resnet50,efficientnet_v2_s`.

- `--freeze_backbone`  
  If set, freezes the feature extractor and trains only the classifier head.

- `--limit_data` / `--limit_data_value`  
  Use these flags to restrict the dataset size (e.g., for few-shot scenarios or fast debugging):  
  - `--limit_data`: Enable data limiting.  
  - `--limit_data_value`: Number of samples to use (e.g., `500`).

---

```bash
# Example: Train ResNet18 on the 'facebin_data' dataset
python scripts/fairness_baseline/train.py \
    --data_root datasets/facebin_data \
    --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large,mnasnet1_0 \
    --epochs 10 \
    --freeze_backbone \
    --out_dir checkpoints/baselines_head_only
```

---

### 4) Evaluate Model Fairness

Evaluate the fairness of your trained models using the centralized evaluation script. This uses a unified `FairnessMetric` to ensure consistent calculations across datasets.

> **Tip:** Ensure `--img_size` and `--resize_mode` match your training settings (e.g., 96x96, letterbox).

```bash
# Example: Evaluate 'personbin' models (96x96) on the FACET dataset
python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir checkpoints/baselines_head_96_limit/baselines \
    --dataset_name facet \
    --csv_path datasets/facet_data/facet_eval.csv \
    --filter person \
    --img_size 96 \
    --resize_mode letterbox \
    --cache_dir .cache/facet_crops
```

```bash
# Example: Evaluate 'facebin' models (96x96) on the FairFace dataset
python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir checkpoints/baselines_head_96_limit/baselines \
    --dataset_name fairface \
    --csv_path datasets/FairFace/0.25/fairface_val.csv \
    --filter face \
    --img_size 96 \
    --resize_mode letterbox
```

The evaluation script will generate a detailed `fairness_results_*.json` file in your checkpoint directory.

---

## Directory Expectations

- `datasets/` – raw datasets downloaded by the scripts.
- `datasets/` – processed binary datasets for training.
- `checkpoints/baselines/` – trained model weights and result summaries.
- `.cache/` – optional caches used during evaluation.

---

## Reproducibility

For deterministic runs, consider setting random seeds and CUDA flags as appropriate for your environment.
