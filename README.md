# MoQ-NAS: A Framework for Multi-Objective Neural Architecture Search

This repository provides a flexible and extensible framework for Neural Architecture Search (NAS), focusing on multi-objective optimization. It implements and compares Quantum-Inspired Evolutionary Algorithms (Q-NAS and MO-QNAS) against traditional Genetic Algorithms (GA, NSGA-II, NSGA-III).

## Features

- **Multiple Search Algorithms:** Includes implementations for:
  - **Quantum-Inspired NAS (QNAS)** for single-objective optimization.
  - **Multi-Objective QNAS (MO-QNAS).**
  - **Classic Genetic Algorithm (GA).**
  - **NSGA-II and NSGA-III** for multi-objective optimization.
- **Modular Architecture:** A clean, refactored structure that separates algorithms, core components, and utilities.
- **Extensible CNN Library:** A rich set of CNN building blocks, including standard convolutions, residual blocks, and attention mechanisms (SE, CBAM).
- **Flexible Configuration:** Easily configure experiments, search spaces, and network parameters using YAML files.
- **Configurable Multi-Objective Optimization:** Optimize for any combination of competing objectives — accuracy, parameter count, FLOPs/MACs, inference time — declared per experiment and validated at startup.
- **Experiment-Matrix Launcher:** A single launcher (`launch.py`) expands a YAML matrix into one run per (experiment × repeat), schedules them across GPU slots, and assigns an explicit seed to every repeat for reproducibility.
- **Selectable Training Precision:** Choose `fp32`, `fp16`, or `bf16` from the config; `bf16` removes the gradient-underflow noise that corrupts the search signal on heterogeneous, short-budget training.
- **Evaluation Cache:** An optional cache reuses the metrics of architectures that have already been evaluated, keyed by network, hyperparameters and the full evaluation configuration (including precision).
- **Checkpointing & Resume:** Every algorithm (MO-QNAS and the GA family: GA, NSGA-II, NSGA-III, MOEA/D) saves its full search state — population, evolved hyperparameters, Pareto archive, algorithm-specific state (quantum PMFs / MOEA/D ideal point / reference directions) and all RNG — at every generation, and can resume an interrupted run from the last boundary, bit-identically to an uninterrupted one.
- **Fairness Evaluation:** A post-processing step to evaluate model fairness across different demographic groups, such as skin tone and race.

## Project Structure

The codebase has been refactored into a modular architecture to improve clarity and maintainability.

```
moqnas/
├── algorithms/
│   ├── ga/                   # Classic Genetic Algorithms (GA, NSGA-II, NSGA-III, MOEA/D)
│   ├── pareto/               # Shared Pareto operators (dominance, diversity, hypervolume)
│   └── qnas/                 # Quantum-Inspired Algorithms (QNAS, MOQNAS) + checkpoint.py
│
├── core/
│   ├── cnn/                  # CNN model definitions, trainer, and metrics
│   ├── fairness/             # Fairness evaluation logic and data loaders
│   ├── config.py             # Experiment configuration handler
│   ├── evaluation.py         # Population evaluation engine (work-stealing scheduler)
│   ├── eval_cache.py         # Optional unified evaluation cache
│   └── precision.py          # fp32 / fp16 / bf16 precision policy
│
├── dataset_utils/
│   ├── factory.py            # Dataset loading and splitting logic
│   └── transformations.py    # Data augmentation and transforms
│
├── utils/
│   └── helpers.py            # General utility functions (facade over io/dataset/...)
│
├── dataset_configs/
│   ├── *.yaml                # Dataset metadata (cifar10.yaml, pathmnist.yaml, ...)
│   └── cfg_obj.json          # Objective senses (maximize / minimize)
│
├── experiment_configs/       # Experiment configs: search space, hyperparameters, objectives
│   ├── cifar/                # Single-objective and GA-family configs
│   ├── cifar_mo/             # Multi-objective configs (incl. *_flops variants)
│   └── ...                   # medmnist, fairness, ...
│
├── experiment_matrices/      # YAML matrices consumed by launch.py (ea, qfamily, smoke)
│
├── scripts/
│   ├── download_datasets/    # Script to download and prepare datasets like FairFace, WiderFace, Coco 
│   ├── fairness_baseline/    # Evaluate fairness of baseline models (e.g., ResNet, MobileNet)
│   └── readme.md             # Instructions to create person/face datasets for fairness evaluation
│
├── run_all_evolution.py      # Single entry point for one evolution run (any algorithm)
├── launch.py                 # Experiment-matrix launcher (multiple runs + GPU scheduling)
└── run_*.sh                  # Thin wrappers calling launch.py with a matrix
```

- `algorithms/`: Contains the core logic for all search algorithms and the shared Pareto operators.
- `core/`: Holds shared components essential for any experiment, including the CNN builder/trainer, the evaluation engine, the cache and the precision policy.
- `dataset_utils/`: Manages all data loading, preprocessing, and splitting.
- `utils/`: Contains helper functions used across the project.
- `dataset_configs/`: Dataset metadata YAMLs and the objective-sense definitions (`cfg_obj.json`).
- `experiment_configs/`: YAML files that define the search space, model parameters, objectives and training settings for each experiment.
- `experiment_matrices/`: YAML matrices that describe a batch of runs for `launch.py`.
- `run_all_evolution.py`: Runs a single evolution; `launch.py` orchestrates many runs from a matrix.

## Fairness Evaluation

The framework includes a **FairnessMetric** module designed to evaluate the performance of trained models across different demographic subgroups. This is treated as a **post-processing** step, meaning it runs on fully trained models to assess their fairness **without influencing the training process itself**.

### How It Works

The fairness evaluation is orchestrated by the `fairness_worker_cuda` function, which performs the following steps for each model architecture in a generation:

1. **Model Loading:** The worker loads a pre-trained model onto a specified CUDA device.
2. **Dataloader Creation:** It creates a special evaluation dataloader for fairness assessment using datasets like **Facet** or **FairFace**.
3. **Inference and Metric Calculation:** The `FairnessMetric` class runs inference on the evaluation dataset and calculates the **True Positive Rate (TPR)** for each demographic group.
4. **Fairness Score Computation:** Based on the per-group TPRs, it computes a final `fairness_score` and other summary metrics.

### Fairness Score Calculation

The primary metric, `fairness_score`, is derived from the per-group TPRs. The key components of this calculation are:

- **Per-Group TPR:** The True Positive Rate is calculated for each demographic group (e.g., for each skin tone in the Facet dataset or each race in the FairFace dataset).
- **Minimum Group TPR (`min_tpr`):** This is the lowest TPR observed across all groups.
- **Sum of Gaps (`spd_sum`):** This value represents the sum of the differences between each group's TPR and the `min_tpr`. A lower `spd_sum` indicates better fairness.
- **Fairness Score:** The final score is calculated as `max(0.0, (beta - spd_sum) / beta)`, where `beta` is a configurable hyperparameter. This score is normalized to a range between **0** and **1**, where **1** represents the best possible fairness.

For the **Facet** dataset, the TPR can be calculated in two ways:

- **hard method:** Assigns each image to a single skin tone class.
- **soft method:** Uses weighted probabilities for each skin tone class.

These fairness metrics are then saved alongside other evaluation results, allowing you to incorporate fairness as a key consideration in your multi-objective NAS experiments.


## Getting Started

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/DiegoPaezA/MoQ-NAS.git
cd MoQ-NAS
pip install -r requirements.txt
```

### 2. Configuration

Each experiment is controlled by a YAML config in `experiment_configs/` (the search space, algorithm and training settings), while `dataset_configs/` holds the dataset metadata and the objective senses (`cfg_obj.json`). Before running an experiment, you can create or modify an `experiment_configs/*.yaml` file to define:

- The dataset (`dataset`, `data_path`).  
- The search space (`function_dict`).  
- Algorithm hyperparameters (`max_generations`, `population_size`, etc.).  
- Training parameters (`batch_size`, `max_epochs`, `optimizer`).  
- Proxy-accuracy aggregation (`eval_window_agg: max | mean | last`, default
  `max`). Controls how `best_accuracy` is aggregated over the last
  `epochs_to_eval` validation epochs: `max` (current behavior), `mean`
  (recommended for new experiments — an unbiased estimator on the small
  validation set) or `last` (final epoch only). Model selection
  (`best_model.pth`) always keeps the best-val-accuracy epoch; only the
  reported scalar changes.  
- Training precision (`precision: fp32 | fp16 | bf16`). `bf16` needs native
  hardware support (Ampere/Ada, e.g. A100/L40S) and runs without a gradient
  scaler; the legacy `mixed_precision: true` flag still works and maps to
  `fp16`. Results are only comparable within the same precision.  
- The objective set (`objectives`, e.g. `[best_accuracy, total_params,
  total_flops]`); names are validated at startup against
  `dataset_configs/cfg_obj.json` and the configured metrics. Replacing the
  measured `cuda_inference_time` with deterministic `total_flops` makes whole
  runs bit-reproducible (see `experiment_configs/cifar_mo/config0_3_flops.yaml`).  


### 3. Running an Experiment

There are two ways to run experiments: a **single run** with `run_all_evolution.py`, or a **batch of runs** with the matrix launcher `launch.py`.

#### 3.1 A single run (`run_all_evolution.py`)

`run_all_evolution.py` is the single entry point for every algorithm; the `--algo` flag selects which one. For example, a Multi-Objective QNAS evolution:

```bash
python run_all_evolution.py \
    --algo moqnas \
    --config_file experiment_configs/cifar_mo/config0_3.yaml \
    --experiment_path experiment_cifar10_qfamily/moqnas/exp10_repeat_1 \
    --data_path datasets/cifar10_data \
    --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --seed 42 \
    --log_level INFO
```

Key flags:

- `--algo`: Algorithm to run (`ga`, `nsga2`, `nsga3`, `moead`, `qnas`, `moqnas`).
- `--config_file`: Experiment config (`experiment_configs/...`).
- `--config_path_dataset`: Dataset metadata YAML (`dataset_configs/...`).
- `--experiment_path`: Directory where logs, models, and results are saved.
- `--seed`: Global RNG seed (makes a run reproducible).
- `--log_level`: Verbosity of the log output.

More examples for different setups:

```bash
# NSGA-II with explicit population and generations
python run_all_evolution.py --algo nsga2 \
    --config_file experiment_configs/cifar/config0.yaml \
    --experiment_path experiment_cifar10_ea/nsga2/exp4_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --population_size 20 --num_generations 150 --seed 42 --log_level INFO

# Multi-objective with accuracy + parameters + FLOPs (a fully reproducible objective set)
python run_all_evolution.py --algo moqnas \
    --config_file experiment_configs/cifar_mo/config0_3_flops.yaml \
    --experiment_path experiment_cifar10_qfamily/moqnas/flops_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --seed 42 --log_level INFO

# Enable the evaluation cache (skips re-training architectures already seen)
python run_all_evolution.py --algo moqnas \
    --config_file experiment_configs/cifar_mo/config0_3.yaml \
    --experiment_path experiment_cifar10_qfamily/moqnas/cached_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --use_cache --seed 42 --log_level INFO

# Resume an interrupted run from its last saved generation (any algorithm)
python run_all_evolution.py --algo nsga2 \
    --config_file experiment_configs/cifar_mo/config0_3.yaml \
    --experiment_path experiment_cifar10_ea/nsga2/exp4_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --resume --seed 42 --log_level INFO
```

> Note on resume: `--resume` works for every algorithm and picks up from
> `<experiment_path>/checkpoint.pkl`. Without it, an existing checkpoint is
> ignored and the run restarts from generation 0 (safe default). Rerun the
> exact same command (same config, population, generations, objectives and
> precision) plus `--resume`; a mismatch aborts naming the differing field.

> Note on precision: select it in the config (`precision: fp32 | fp16 | bf16`).
> `bf16` requires native hardware support (Ampere/Ada, e.g. A100/L40S).
> Multi-GPU is controlled with `CUDA_VISIBLE_DEVICES` (e.g.
> `CUDA_VISIBLE_DEVICES=0,1 python run_all_evolution.py ...`); candidates are
> evaluated in parallel and balanced across the visible GPUs.

#### 3.2 A batch of runs (`launch.py`)

`launch.py` expands an **experiment matrix** (a YAML file in `experiment_matrices/`) into one `run_all_evolution.py` invocation per (experiment × repeat). It schedules the runs across the GPU slots declared in the matrix (one run per slot at a time), gives every repeat its own seed (`seed_base + repeat_index`), and records the exact command in each experiment directory.

A matrix looks like this:

```yaml
# experiment_matrices/qfamily.yaml
exp_root: experiment_cifar10_qfamily
gpus: [0]            # GPU slot pool; len(gpus) runs execute concurrently
repeats: 3
seed_base: 42        # repeat i (1-based) runs with seed_base + i

defaults:            # arguments common to every run
  data_path: datasets/cifar10_data
  dataset: cifar10
  config_path_dataset: dataset_configs/cifar10.yaml
  log_level: INFO

experiments:
  - algo: moqnas
    config: experiment_configs/cifar_mo/config0_2.yaml
    name: exp10
    overrides: {optimizer: AdamW, elite_mode: moead_topk}
    flags: [--multi_objective]
```

Run it:

```bash
# Preview the exact commands without running anything
python launch.py experiment_matrices/qfamily.yaml --dry-run

# Launch the MO-QNAS family (3 repeats, seeds 43/44/45)
python launch.py experiment_matrices/qfamily.yaml

# Launch the GA / NSGA family
python launch.py experiment_matrices/ea.yaml
```

To run several experiments in parallel, list more than one GPU in `gpus`
(e.g. `gpus: [0, 1]`) and/or add more entries under `experiments:`; the
launcher keeps every GPU slot busy and reports a summary at the end, so a
single failed run never stops the others.

The root-level `run_*.sh` scripts are now thin wrappers over the launcher,
so `./run_moqnas_1.sh` and `./run_moqnas_1.sh --dry-run` are equivalent to
calling `launch.py` with the matching matrix.

#### Example: the four multi-objective algorithms on (accuracy, FLOPs)

The matrix `experiment_matrices/acc_flops.yaml` runs **MO-QNAS, NSGA-II,
NSGA-III and MOEA/D** on the `(best_accuracy, total_flops)` objective set,
**3 repeats each**, on a **single GPU**. The objectives come from the config
(`experiment_configs/cifar_mo/config0_3_acc_flops.yaml`); because FLOPs are
deterministic (unlike measured inference time), these runs are fully
reproducible. NSGA-II/III and MOEA/D take `population_size` and
`num_generations` from the matrix, while MO-QNAS reads its population from the
config and uses `num_generations` only to override `max_generations`.

```yaml
# experiment_matrices/acc_flops.yaml
exp_root: experiment_cifar10_acc_flops
gpus: [0]            # single GPU -> the runs execute one after another
repeats: 3
seed_base: 42        # repeats use seeds 43, 44, 45

defaults:
  data_path: datasets/cifar10_data
  dataset: cifar10
  config_path_dataset: dataset_configs/cifar10.yaml
  log_level: INFO
  multi_objective: true       # keep the config's multi-objective setting
  population_size: 20         # used by nsga2/nsga3/moead (ignored by moqnas)
  num_generations: 150

experiments:
  - {algo: moqnas, config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: moqnas}
  - {algo: nsga2,  config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: nsga2}
  - {algo: nsga3,  config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: nsga3,
     overrides: {ref_divisions: 12}}
  - {algo: moead,  config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: moead,
     overrides: {ref_divisions: 12, moead_T: 20, moead_scalar: tchebycheff, moead_pneighbor: 0.9}}
```

```bash
# Preview the 12 commands (4 algorithms x 3 repeats) without running anything
python launch.py experiment_matrices/acc_flops.yaml --dry-run

# Launch them (sequentially, since gpus: [0])
python launch.py experiment_matrices/acc_flops.yaml
```

Results are written under
`experiment_cifar10_acc_flops/<algo>/<algo>_repeat_<i>/`. To run the four
algorithms concurrently instead, set `gpus: [0, 1, 2, 3]` (one algorithm per
GPU slot).

#### Resuming an interrupted batch

Every run (MO-QNAS and the GA family: GA, NSGA-II, NSGA-III, MOEA/D) writes a
`checkpoint.pkl` at each generation boundary. If a batch is interrupted (power
loss, preemption), relaunch the **same matrix** with `--resume`: each cell
picks up from its own `<experiment_path>/checkpoint.pkl` instead of starting
over, and the resumed search is bit-identical to an uninterrupted one.

```bash
# Same command that launched the batch, plus --resume
python launch.py experiment_matrices/acc_flops.yaml --resume
```

`--resume` can also be made the default for a matrix by adding `resume: true`
at its top level. Without `--resume` (and without the key), an existing
checkpoint is ignored and the run restarts from generation 0 — the safe
default for a fresh launch.

### 4. Environment Configuration

The following steps are used to configure the environment for the project.

- Miniconda Installation
- Conda Environment Creation
- Package Installation

**Notes**: 
- An NVIDIA GPU is required to run the project. 
- The project is tested on Ubuntu 22.04 LTS with NVIDIA L40S GPU.

#### 4.1 Miniconda Installation

Install Miniconda in the home directory. Refer to the [Miniconda Installation Guide](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) for more information.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

```bash
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
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt
```

## To-Do
- [ ] Add more search algorithms (e.g., DARTS, ENAS).
- [ ] Improve FP8 mixed-precision training support.
- [ ] Implement additional fairness metrics (e.g., Equal Opportunity, Demographic Parity).
- [ ] Add support for more datasets and tasks (e.g., object detection, segmentation).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
