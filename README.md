# MoQ-NAS: Multi-Objective Neural Architecture Search on CIFAR-10

This repository is a focused framework for **multi-objective** Neural Architecture Search (NAS) on **CIFAR-10**. It searches for CNN architectures that jointly optimize accuracy against hardware cost (parameter count, FLOPs, and/or measured inference time), comparing a Quantum-Inspired Evolutionary Algorithm (MO-QNAS) against classic Pareto/decomposition-based Genetic Algorithms (NSGA-II, NSGA-III, MOEA/D).

It also ships [`vit_transformer_search.py`](vit_transformer_search.py), a small, self-contained DARTS-style search over the attention heads of a pretrained Vision Transformer on CIFAR-10 — independent of the framework below (see [ViT attention-head search](#vit-attention-head-search-vit_transformer_searchpy)).

## Features

- **Four multi-objective search algorithms:**
  - **Multi-Objective QNAS (MO-QNAS)** — quantum-inspired probability-distribution search.
  - **NSGA-II** — Pareto dominance + crowding distance.
  - **NSGA-III** — Pareto dominance + reference-direction niching (better for ≥3 objectives).
  - **MOEA/D** — decomposition into scalar subproblems over a neighborhood of weight vectors.
- **Modular architecture:** algorithms, core CNN/training engine and utilities are cleanly separated.
- **Extensible CNN search space:** standard convolutions, depthwise/MBConv blocks, residual blocks, SE/CBAM attention, pooling.
- **Configurable objectives:** any combination of accuracy, parameter count, FLOPs/MACs and measured inference time, declared per experiment and validated at startup.
- **Experiment-Matrix Launcher:** `launch.py` expands a YAML matrix into one run per (experiment × repeat), schedules them across GPU slots, and assigns an explicit seed to every repeat for reproducibility.
- **Selectable Training Precision:** `fp32`, `fp16`, or `bf16` from the config.
- **Runs on CUDA, Apple Silicon (MPS), or CPU** — the evaluation engine picks the best available device automatically (CUDA → MPS → CPU); MPS is a single shared GPU, so use a small `--threads`/`workers_per_gpu` there.
- **Evaluation Cache:** an optional cache reuses the metrics of architectures already evaluated, keyed by network, hyperparameters and the full evaluation config (including precision).
- **Checkpointing & Resume:** every algorithm saves its full search state (population, evolved hyperparameters, Pareto archive, algorithm-specific state, and all RNG) at every generation, and can resume an interrupted run bit-identically.

## Project Structure

```
moqnas/
├── algorithms/
│   ├── ga/                   # NSGA-II, NSGA-III, MOEA/D (+ shared GA infrastructure)
│   ├── pareto/                # Shared Pareto operators (dominance, diversity, hypervolume, ref. directions)
│   └── qnas/                  # MO-QNAS (+ shared QNAS infrastructure) + checkpoint.py
│
├── core/
│   ├── cnn/                   # CNN model definitions, trainer, and metrics
│   ├── config.py               # Experiment configuration handler
│   ├── evaluation.py           # Population evaluation engine (work-stealing scheduler)
│   ├── eval_cache.py           # Optional unified evaluation cache
│   └── precision.py            # fp32 / fp16 / bf16 precision policy
│
├── dataset_utils/
│   ├── factory.py              # CIFAR-10 loading and splitting logic
│   └── transformations.py       # Data augmentation and transforms
│
├── utils/
│   └── helpers.py               # General utility functions (facade over io/dataset/...)
│
├── dataset_configs/
│   ├── cifar10.yaml             # Dataset metadata
│   └── cfg_obj.json             # Objective senses (maximize / minimize)
│
├── experiment_configs/
│   └── cifar_mo/                # Multi-objective configs (accuracy + hardware objectives)
│
├── experiment_matrices/         # YAML matrices consumed by launch.py
│
├── run_all_evolution.py         # Single entry point for one evolution run (any algorithm)
├── launch.py                    # Experiment-matrix launcher (multiple runs + GPU scheduling)
└── vit_transformer_search.py    # Standalone ViT attention-head DARTS search (see below)
```

- `algorithms/`: Core logic for NSGA-II/III, MOEA/D, MO-QNAS and the shared Pareto operators. The single-objective `GA`/`QNAS` base classes still live under `algorithms/ga/base_ga.py` and `algorithms/qnas/qnas2.py` as shared infrastructure (NSGA-II subclasses `GA`, MO-QNAS subclasses `QNAS`) — they are no longer selectable as standalone `--algo` choices.
- `core/`: The CNN builder/trainer, the evaluation engine, the cache and the precision policy.
- `dataset_utils/`: CIFAR-10 loading, preprocessing, and splitting.
- `dataset_configs/`: Dataset metadata and the objective-sense definitions (`cfg_obj.json`).
- `experiment_configs/cifar_mo/`: YAML files defining the search space, algorithm and training settings for each experiment (accuracy + hardware objectives).
- `experiment_matrices/`: YAML matrices that describe a batch of runs for `launch.py`.

## Getting Started

### 1. Installation

```bash
git clone https://github.com/juliolops/vit_darts_simple.git
cd vit_darts_simple
pip install -r requirements.txt
```

### 2. Configuration

Each experiment is controlled by a YAML config in `experiment_configs/cifar_mo/` (search space, algorithm and training settings), while `dataset_configs/cifar10.yaml` holds the dataset metadata and `dataset_configs/cfg_obj.json` the objective senses. Key things to set in an `experiment_configs/cifar_mo/*.yaml`:

- `train.function_dict` under `QNAS:` — the search space (layer types/params to choose from).
- Algorithm hyperparameters (`max_generations`, `population_size`, etc. — GA-family algorithms take `population_size`/`num_generations` from the CLI instead).
- Training parameters (`batch_size`, `max_epochs`, `optimizer`).
- Training precision (`precision: fp32 | fp16 | bf16`). `bf16`/`fp16` need CUDA or MPS hardware; `fp32` always works, including on CPU.
- The objective set (`objectives`, e.g. `[best_accuracy, total_params, total_flops]`); names are validated at startup against `dataset_configs/cfg_obj.json` and the configured `metrics:` plugins (`Accuracy`, `HardwareMetrics`, `ValidationLossFitness`, `ScalarizedFitness`).

### 3. Running an Experiment

There are two ways to run experiments: a **single run** with `run_all_evolution.py`, or a **batch of runs** with the matrix launcher `launch.py`.

#### 3.1 A single run (`run_all_evolution.py`)

```bash
python run_all_evolution.py \
    --algo nsga3 \
    --config_file experiment_configs/cifar_mo/config0_3_flops.yaml \
    --experiment_path experiment_cifar10/nsga3/exp1_repeat_1 \
    --data_path datasets/cifar10_data \
    --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --population_size 20 --num_generations 150 --seed 42 --log_level INFO
```

Key flags:

- `--algo`: Algorithm to run (`nsga2`, `nsga3`, `moead`, `moqnas`).
- `--config_file`: Experiment config (`experiment_configs/cifar_mo/...`).
- `--config_path_dataset`: Dataset metadata YAML (`dataset_configs/cifar10.yaml`).
- `--experiment_path`: Directory where logs, models, and results are saved.
- `--seed`: Global RNG seed (makes a run reproducible).
- `--log_level`: Verbosity of the log output.

More examples:

```bash
# Multi-objective with accuracy + parameters + FLOPs (fully reproducible, no measured-timing noise)
python run_all_evolution.py --algo moqnas \
    --config_file experiment_configs/cifar_mo/config0_3_flops.yaml \
    --experiment_path experiment_cifar10/moqnas/flops_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --seed 42 --log_level INFO

# MOEA/D
python run_all_evolution.py --algo moead \
    --config_file experiment_configs/cifar_mo/config0_3_acc_flops.yaml \
    --experiment_path experiment_cifar10/moead/exp1_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --population_size 20 --num_generations 150 --ref_divisions 12 --seed 42 --log_level INFO

# Enable the evaluation cache (skips re-training architectures already seen)
python run_all_evolution.py --algo nsga2 \
    --config_file experiment_configs/cifar_mo/config0_3.yaml \
    --experiment_path experiment_cifar10/nsga2/cached_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --use_cache --seed 42 --log_level INFO

# Resume an interrupted run from its last saved generation (any algorithm)
python run_all_evolution.py --algo nsga3 \
    --config_file experiment_configs/cifar_mo/config0_3.yaml \
    --experiment_path experiment_cifar10/nsga3/exp1_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --resume --seed 42 --log_level INFO
```

> Note on resume: `--resume` works for every algorithm and picks up from
> `<experiment_path>/checkpoint.pkl`. Without it, an existing checkpoint is
> ignored and the run restarts from generation 0 (safe default). Rerun the
> exact same command (same config, population, generations, objectives and
> precision) plus `--resume`; a mismatch aborts naming the differing field.

> Note on hardware: the evaluation engine picks CUDA if available, otherwise
> Apple Silicon MPS, otherwise CPU — no flag needed. Multi-GPU (CUDA) is
> controlled with `CUDA_VISIBLE_DEVICES` (e.g. `CUDA_VISIBLE_DEVICES=0,1 python
> run_all_evolution.py ...`). On MPS, prefer a small `--threads`/`workers_per_gpu`
> (1-2) since it's one shared GPU, not several.

#### 3.2 A batch of runs (`launch.py`)

`launch.py` expands an **experiment matrix** (a YAML file in `experiment_matrices/`) into one `run_all_evolution.py` invocation per (experiment × repeat), scheduling runs across the GPU slots declared in the matrix.

```bash
# Preview the exact commands without running anything
python launch.py experiment_matrices/acc_flops.yaml --dry-run

# Launch the four multi-objective algorithms (MO-QNAS, NSGA-II, NSGA-III, MOEA/D)
# on (best_accuracy, total_flops), 3 repeats each
python launch.py experiment_matrices/acc_flops.yaml

# Resume an interrupted batch
python launch.py experiment_matrices/acc_flops.yaml --resume
```

`--resume` can also be made the default for a matrix by adding `resume: true`
at its top level.

### 4. Environment Configuration

**Notes**:
- Works on CUDA (NVIDIA GPU), Apple Silicon (MPS), or CPU.
- Tested on Ubuntu 22.04 LTS with an NVIDIA L40S GPU, and on macOS with Apple Silicon (M-series).

#### 4.1 Miniconda Installation

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

#### 4.2 Conda Environment Creation

```bash
conda create -n moqnas python=3.10
conda activate moqnas
```

#### 4.3 Package Installation

```bash
# Linux + NVIDIA GPU:
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# macOS (Apple Silicon, MPS):
pip install torch torchvision torchaudio

pip install -r requirements.txt
```

## ViT attention-head search (`vit_transformer_search.py`)

A separate, self-contained script that runs a DARTS-style search over which
attention heads of a pretrained `vit_base_patch16_224` (via `timm`) matter
most for CIFAR-10, learning a softmax weighting (`alphas`) per block while
fine-tuning the MLP heads. It does not use anything from the rest of this
repo (no `core/`, `algorithms/`, or config files) and picks CUDA/MPS/CPU
automatically.

```bash
pip install timm
python vit_transformer_search.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
