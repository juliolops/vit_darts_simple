# ViT-NAS: Multi-Objective Attention-Head Pruning on CIFAR-10

This repository searches for **pruned Vision Transformers** on **CIFAR-10**, trading **accuracy** against **hardware cost** (FLOPs, parameter count, inference time). It runs in two phases: DARTS learns which attention heads matter, then a multi-objective genetic algorithm decides how aggressively to prune each block.

The search compares a Quantum-Inspired Evolutionary Algorithm (MO-QNAS) against classic Pareto/decomposition-based Genetic Algorithms (NSGA-II, NSGA-III, MOEA/D).

## How it works

**DARTS decides *which* heads matter; the GA decides *how much* to prune.**

- **Phase 1 (DARTS)** learns one importance weight (*alpha*) per attention head of a pretrained `vit_base_patch16_224`.
- **Phase 2 (GA)** evolves a chromosome with **one gene per transformer block** (12 for ViT-Base). Each gene is the **percentage of heads that block keeps** — 20% to 90%, in steps of 10. A gene of 40% keeps the 40% of that block's heads with the largest alpha.

Pruning is **surgical, not masking**: the block's `qkv`/`proj` layers are rebuilt holding only the surviving heads, so a pruned candidate really is smaller and cheaper (85.8M → 62.2M parameters at 20% heads). Each candidate is then fine-tuned with **only the classifier head trainable**, so its accuracy reflects the pruned representation rather than a full retraining of the backbone.

## Features

- **Four multi-objective search algorithms:**
  - **Multi-Objective QNAS (MO-QNAS)** — quantum-inspired probability-distribution search.
  - **NSGA-II** — Pareto dominance + crowding distance.
  - **NSGA-III** — Pareto dominance + reference-direction niching (better for ≥3 objectives).
  - **MOEA/D** — decomposition into scalar subproblems over a neighborhood of weight vectors.
- **Configurable objectives:** any combination of accuracy, FLOPs, parameter count and measured inference time, declared per experiment and validated at startup.
- **Experiment-Matrix Launcher:** `launch.py` expands a YAML matrix into one run per (experiment × repeat), schedules them across GPU slots, and assigns an explicit seed to every repeat for reproducibility.
- **Selectable Training Precision:** `fp32`, `fp16`, or `bf16` from the config.
- **Runs on CUDA, Apple Silicon (MPS), or CPU** — the evaluation engine picks the best available device automatically (CUDA → MPS → CPU); MPS is a single shared GPU, so use a small `--threads`/`workers_per_gpu` there.
- **Evaluation Cache:** an optional cache reuses the metrics of candidates already evaluated.
- **Checkpointing & Resume:** every algorithm saves its full search state (population, Pareto archive, algorithm-specific state, and all RNG) at every generation, and can resume an interrupted run bit-identically.

## Project Structure

```
├── algorithms/
│   ├── ga/                       # NSGA-II, NSGA-III, MOEA/D (+ shared GA infrastructure)
│   ├── pareto/                    # Shared Pareto operators (dominance, diversity, hypervolume)
│   └── qnas/                      # MO-QNAS (+ shared QNAS infrastructure)
│
├── core/
│   ├── vit.py                     # ViT head pruning + DARTS alphas (the search space)
│   ├── training/                  # Trainer, data loader and metrics
│   │   └── metrics/               # Accuracy + HardwareMetrics (FLOPs, params, time, memory)
│   ├── config.py                  # Experiment configuration handler
│   ├── evaluation.py              # Population evaluation engine (work-stealing scheduler)
│   ├── eval_cache.py              # Optional unified evaluation cache
│   └── precision.py               # fp32 / fp16 / bf16 precision policy
│
├── dataset_utils/                 # CIFAR-10 loading, splitting and transforms
├── dataset_configs/
│   ├── cifar10_vit.yaml           # Dataset metadata (224x224 + ImageNet normalization)
│   └── cfg_obj.json               # Objective senses (maximize / minimize)
│
├── experiment_configs/vit/        # Search space, algorithm and training settings
├── experiment_matrices/           # YAML matrices consumed by launch.py
│
├── run_darts_alphas.py            # Phase 1: learn the attention-head alphas
├── run_all_evolution.py           # Phase 2: one evolution run (any algorithm)
├── launch.py                      # Experiment-matrix launcher (multiple runs)
└── vit_transformer_search.py      # The standalone DARTS implementation (reused by phase 1)
```

## Getting Started

### 1. Installation

```bash
git clone https://github.com/juliolops/vit_darts_simple.git
cd vit_darts_simple
pip install -r requirements.txt
```

### 2. Phase 1 — learn the alphas (DARTS)

```bash
python run_darts_alphas.py --epochs 1 --limit_train 2000 \
    --output darts_alphas/vit_base_cifar10.json
```

Reuses [`vit_transformer_search.py`](vit_transformer_search.py) (unchanged) and writes one alpha per head, per block, to JSON. **Run this once** — every search below reuses the file.

`vit_transformer_search.py` also still runs standalone (`python vit_transformer_search.py`) as the original self-contained DARTS demo.

### 3. Phase 2 — multi-objective search over the pruning percentages

```bash
python run_all_evolution.py \
    --algo nsga3 \
    --config_file experiment_configs/vit/config_vit_heads.yaml \
    --experiment_path experiment_vit/nsga3/run1 \
    --data_path data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10_vit.yaml \
    --population_size 12 --num_generations 20 \
    --multi_objective --seed 42 --log_level INFO
```

Key flags:

- `--algo`: Algorithm to run (`nsga2`, `nsga3`, `moead`, `moqnas`).
- `--config_file`: Experiment config (`experiment_configs/vit/...`).
- `--config_path_dataset`: Dataset metadata YAML (`dataset_configs/cifar10_vit.yaml`).
- `--experiment_path`: Directory where logs and results are saved.
- `--seed`: Global RNG seed (makes a run reproducible).
- `--use_cache`: Skip re-evaluating candidates already seen.
- `--resume`: Continue from `<experiment_path>/checkpoint.pkl`.

> Note on resume: `--resume` works for every algorithm. Without it, an existing
> checkpoint is ignored and the run restarts from generation 0 (safe default).
> Rerun the exact same command plus `--resume`; a mismatch aborts naming the
> differing field.

> Note on hardware: the evaluation engine picks CUDA if available, otherwise
> Apple Silicon MPS, otherwise CPU — no flag needed. Multi-GPU (CUDA) is
> controlled with `CUDA_VISIBLE_DEVICES`. On MPS, prefer a small
> `--threads`/`workers_per_gpu` (1-2) since it's one shared GPU.

### 4. A batch of runs (`launch.py`)

```bash
# Preview the exact commands without running anything
python launch.py experiment_matrices/vit_acc_flops.yaml --dry-run

# Launch the four multi-objective algorithms, 3 repeats each
python launch.py experiment_matrices/vit_acc_flops.yaml

# Resume an interrupted batch
python launch.py experiment_matrices/vit_acc_flops.yaml --resume
```

## Configuration

Each experiment is a YAML in `experiment_configs/vit/`. The `QNAS:` block holds the search space and algorithm hyperparameters; the `train:` block holds everything about training and objectives:

- `max_num_nodes` — chromosome length; **must equal the ViT's block count** (12 for `vit_base_patch16_224`).
- `function_dict` — the search space: one entry per pruning percentage.
- `vit_model_name` / `vit_alphas_path` — which ViT to prune and where its alphas live.
- `objectives` — e.g. `[best_accuracy, total_flops]`; validated at startup against `dataset_configs/cfg_obj.json` and the configured `metrics:` plugins (`Accuracy`, `HardwareMetrics`).
- `precision` — `fp32 | fp16 | bf16` (`bf16`/`fp16` need CUDA or MPS).

## Environment

Tested on Ubuntu 22.04 with an NVIDIA L40S, and on macOS with Apple Silicon.

```bash
conda create -n vitnas python=3.10
conda activate vitnas

# Linux + NVIDIA GPU:
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
# macOS (Apple Silicon, MPS):
pip install torch torchvision

pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
