# ViT-NAS: Multi-Objective ViT Pruning on CIFAR-10

Searches for **pruned Vision Transformers** on **CIFAR-10**, trading **accuracy** (maximize) against **FLOPs** (minimize), in two phases:

1. **DARTS** learns one importance weight (*alpha*) per **attention head** and per **MLP hidden neuron** of a pretrained `vit_base_patch16_224`. Each head output and each hidden activation is multiplied by `N · softmax(alphas)` over its block (N = 12 heads or 3072 neurons), so the weights average 1 and training starts from the exact pretrained model. On every step the alphas are updated on a validation batch and the **MLP + classifier weights** on a training batch; the attention layers keep their pretrained weights.
2. **A multi-objective genetic algorithm** ([pymoo](https://pymoo.org) NSGA-II or NSGA-III) evolves one gene per transformer block: the **percentage of heads that block keeps**, from 20% to 90%. A gene of 40% keeps the 40% of that block's heads with the largest alpha. With `prune_mlp: true` in `config.yaml`, a second gene per block sets the **percentage of MLP hidden neurons** kept, also chosen by their alphas (24 genes in total). The objectives are accuracy (maximize) and FLOPs (minimize).

| Chromosome | Genes | Pruned | Min. FLOPs (all genes at 20%) |
|---|---|---|---|
| `prune_mlp: false` | 12 | attention heads | ~24 G |
| `prune_mlp: true` | 24 | attention heads + MLP neurons | ~6.5 G |

The unpruned ViT-Base has 85.8M parameters and ~33.7G FLOPs; the MLP accounts for about two thirds of each block's compute.

Pruning is surgical: each block's `qkv`/`proj` (and `fc1`/`fc2`) layers are rebuilt with only the survivors, so a pruned candidate really is smaller. Each candidate then fine-tunes **its MLPs and the classifier head** for a few epochs on a class-balanced CIFAR-10 subset, while the **attention layers keep their pretrained weights frozen** (the same split used by the DARTS phase), and is scored on accuracy and FLOPs. The classifier and the pretrained MLPs have separate learning rates (`learning_rate`, `mlp_learning_rate` in `config.yaml`); a large MLP learning rate (1e-3) wipes out the pretrained features within a few steps.

## Data protocol

The CIFAR-10 **test set is held out for the final evaluation only**. Every phase uses the same stratified split of the CIFAR-10 train set (`train_split`, `split_seed` in `config.yaml`):

| Set | Source | DARTS | Search (GA) | `retrain.py` |
|---|---|---|---|---|
| train | 90% of CIFAR-10 train | MLP + classifier weights | trains each candidate | trains the final model |
| val | 10% of CIFAR-10 train | alphas | accuracy objective | picks the best epoch |
| **test** | official CIFAR-10 test (10,000) | — | — | **one final evaluation** |

The test set is only reachable through `core.data.build_test_dataset`, which only `retrain.py` calls, after training.

## Project structure

```
├── run_darts_alphas.py        # Phase 1: learn head + MLP-neuron alphas -> darts_alphas/*.json
├── vit_transformer_search.py  # DARTS attention/MLP wrappers + training epoch
├── run_all_evolution.py       # Phase 2: multi-objective search
├── retrain.py                 # Phase 3: retrain a Pareto candidate, evaluate once on the test set
├── config.yaml                # Search space, ViT and training settings
├── algorithms/
│   └── nsga.py                # pymoo NSGA-II/III ask-tell loop, Pareto archive + hypervolume
└── core/
    ├── vit.py                 # Alphas I/O + attention-head and MLP-neuron pruning
    ├── data.py                # CIFAR-10 train/val split, balanced subset, held-out test set
    ├── training.py            # Fine-tune MLPs + classifier, accuracy + FLOPs
    ├── evaluation.py          # Parallel evaluation of a population
    ├── config.py              # Config loading and validation
    └── utils.py               # Seeding, logging, device selection
```

## Running a quick test

### 1. Environment

The environment is built with [uv](https://docs.astral.sh/uv/), a fast Python package and environment manager. It needs no root access and no conda, and it downloads its own Python 3.10, so it works even on servers whose system Python is older (e.g. Python 3.8 on Ubuntu 20.04).

Requirements on the machine: an NVIDIA driver (check with `nvidia-smi`), `curl`, and internet access. The CUDA toolkit is **not** needed: the PyTorch wheels ship their own CUDA libraries.

#### 1.1 Choose where everything is stored

On shared servers the home directory often has a small quota. uv, PyTorch and Hugging Face write their caches and downloads (several GB) to `~` by default, which fails with `No space left on device`. Pick a directory with free space (here `/data-ext/$USER`) and point every cache there with an `env.sh` file:

```bash
export BASE=/data-ext/$USER          # a directory with plenty of free space
mkdir -p $BASE

cat > $BASE/env.sh << EOF
export PATH=$BASE/uv/bin:\$PATH                  # the uv executable
export UV_CACHE_DIR=$BASE/cache/uv              # uv download cache
export UV_PYTHON_INSTALL_DIR=$BASE/uv/python    # Python versions installed by uv
export PIP_CACHE_DIR=$BASE/cache/pip            # pip cache, if pip is ever used
export HF_HOME=$BASE/cache/hf                   # pretrained ViT weights (timm / Hugging Face)
export TORCH_HOME=$BASE/cache/torch             # torch hub cache
EOF

source $BASE/env.sh
```

If your home directory has enough space, you can skip this step and the `source $BASE/env.sh` lines below.

#### 1.2 Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$BASE/uv/bin UV_NO_MODIFY_PATH=1 sh
which uv        # should print $BASE/uv/bin/uv
```

#### 1.3 Create the virtual environment

From the repository root:

```bash
uv venv .venv --python 3.10      # downloads Python 3.10 if needed
source .venv/bin/activate
python --version                 # Python 3.10.x
```

The environment lives in `.venv/` inside the repository and is ignored by git.

#### 1.4 Install the dependencies (PyTorch first)

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

**Install PyTorch first, from the index that matches your driver.** `requirements.txt` does not list `torch`, but `timm` depends on it. If `torch` is not installed yet, `timm` pulls the newest PyTorch from PyPI, which may be built for a CUDA version your driver does not support. Pick the index from the `CUDA Version` shown by `nvidia-smi` (the maximum CUDA version the driver supports):

| `nvidia-smi` CUDA Version | Index URL |
|---|---|
| 12.1 – 12.3 | `https://download.pytorch.org/whl/cu121` |
| 12.4 – 12.5 | `https://download.pytorch.org/whl/cu124` |
| ≥ 12.6 | `https://download.pytorch.org/whl/cu126` |
| 11.8 – 12.0 | `https://download.pytorch.org/whl/cu118` |

#### 1.5 Check the GPUs

```bash
python -c "import sys, torch; print('Python', sys.version.split()[0], '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
```

Expected: `CUDA: True` and the number of GPUs listed by `nvidia-smi`. With `CUDA: False` everything runs on the CPU, which is far too slow for a ViT-Base.

#### 1.6 Every new terminal

```bash
source /data-ext/$USER/env.sh                 # the env.sh from step 1.1 (adjust if BASE differs)
cd /path/to/vit_darts_simple && source .venv/bin/activate
```

#### 1.7 Long runs over SSH

A process started in a plain SSH terminal dies when the connection drops. Run the phases inside `tmux`:

```bash
tmux new -s vit          # start a session (then activate the environment as in 1.6)
# Ctrl+B then D          # detach: the run keeps going
tmux attach -t vit       # reattach later
```

To keep a copy of the output, append `2>&1 | tee <name>.log` to any command (add `*.log` to `.gitignore` so the logs are not committed).

#### 1.8 What stays out of git

| Created by the setup | Location | In git? |
|---|---|---|
| Virtual environment | `.venv/` (repository) | No, listed in `.gitignore` |
| CIFAR-10 download | `data/` (repository) | No, listed in `.gitignore` |
| Experiment outputs | `experiment_vit/`, `darts_alphas/`, `retrain_results/` | No, listed in `.gitignore` |
| uv, its Python, caches, `env.sh` | `$BASE` (outside the repository) | No, outside the repository |

The exact package versions are not pinned yet: `requirements.txt` lists only names, so a fresh install takes the newest compatible versions. For a fully reproducible environment, the project can be moved to uv's project mode (`pyproject.toml` + a committed `uv.lock`).

#### 1.9 Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No space left on device` while installing | The cache is in a full home directory | Do step 1.1, then `rm -rf ~/.cache/uv` |
| `ModuleNotFoundError: No module named 'torch'` | The environment is not active, so `python` is the system Python | `source .venv/bin/activate` (the prompt shows `(.venv)`) |
| `CUDA: False`, or `torch` asks for CUDA 13 | PyTorch came from PyPI instead of the CUDA index | `uv pip install --reinstall torch torchvision --index-url <index from 1.4>` |
| `NoWritablePkgsDirError` (conda) | A `.condarc` points `pkgs_dirs` to a directory you cannot write | Not needed with uv; for conda, check `conda config --show-sources` |
| Warning: `unauthenticated requests to the HF Hub` | The ViT weights are downloaded without a Hugging Face token | Harmless; the download still works |

<details>
<summary>Alternative: conda</summary>

```bash
conda create -n vitnas python=3.10 -y
conda activate vitnas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

</details>

CIFAR-10 (~170 MB) and the pretrained ViT weights (~350 MB) are downloaded on first use.

### 2. Phase 1 — alphas (DARTS)

```bash
python run_darts_alphas.py --config_file config.yaml --epochs 1 --limit_data_value 500
```

The alphas are written to `vit_alphas_path` from the config (or `--output`). `--limit_data_value` is the total of train + val images (0 = the whole split). This short run only checks the pipeline; use more images/epochs for a meaningful ranking. The run is seeded (`--seed`), so the same command gives the same alphas on the same device.

A longer run, on the whole train/val split:

```bash
python run_darts_alphas.py --config_file config.yaml --epochs 10 --limit_data_value 0
```

#### Training stability

`N · softmax(alphas)` is unbounded: once the alphas of a block drift apart, a few heads or MLP neurons can be amplified tens of times, and together with a high learning rate on the pretrained MLP this can make training diverge after a few stable epochs (loss spikes, validation accuracy collapsing). The script guards against it:

| Flag | Default | Effect |
|---|---|---|
| `--lr_w` | `1e-4` | Learning rate of the MLP + classifier weights (1e-3 destroys the pretrained features) |
| `--lr_alpha` | `3e-4` | Learning rate of the alphas |
| `--grad_clip` | `5.0` | Max gradient norm of the weights and of the alphas, applied separately (`0` disables it) |
| `--warmup_epochs` | `1` | First epochs train only the weights, with the alphas fixed (capped at `epochs - 1`, so at least one epoch learns the alphas) |

After every epoch the model is scored on the **whole** validation split, and the alphas saved are those of the epoch with the best validation accuracy (among the epochs that updated the alphas), not the last one. The per-epoch loss/accuracy, the best epoch and the arguments are written to `<output>.history.json` (e.g. `darts_alphas/vit_base_cifar10.history.json`).

After every epoch that trains the alphas, the script also prints an alphas summary, so their drift can be followed while training runs: per block, the heads by importance, the range of the head weights, and the min/max/std of the MLP-neuron weights, followed by the largest head and MLP weight over all blocks. The same summary is printed for the saved (best) alphas at the end. All weights average 1: values near 1 mean almost uniform importances, and a max far above 1 (tens or more) means the weight has concentrated on a few heads or neurons.

The **full alphas of every epoch** (the 12 head weights and the 3072 MLP-neuron weights of every block, in the same format as the output file) are saved to `<output stem>_epochs/epoch_NN.json`, e.g. `darts_alphas/vit_base_cifar10_epochs/epoch_03.json`. Any of them can be passed to the search by pointing `vit_alphas_path` in the config at it. The output file itself is written as soon as a new best epoch appears, so an interrupted run still leaves the best alphas found so far.

`<output>.history.json` is rewritten after every epoch, so it can be read while the run is in progress (and survives an interrupted run). Each epoch entry holds `val_loss`, `val_acc`, `alphas_file` (that epoch's full alphas) and `alphas`: one record per block with `head_ranking`, `head_weights` (the 12 head weights), `head_min/max/std` and `mlp_min/max/std`. For example, the largest MLP weight per epoch:

```bash
python -c "import json; h=json.load(open('darts_alphas/vit_base_cifar10.history.json')); [print(e['epoch'], '%.2f%%' % e['val_acc'], 'max MLP=%.2f' % max(b['mlp_max'] for b in e['alphas'])) for e in h['epochs']]"
```

### 3. Phase 2 — search

```bash
python run_all_evolution.py --config_file config.yaml \
    --experiment_path experiment_vit/teste1 --algorithm nsga2 \
    --population_size 4 --num_generations 3 \
    --limit_data_value 500 --threads 1 --seed 42
```

To also prune the MLP, set `prune_mlp: true` in `config.yaml` (or a copy of it passed to `--config_file`).

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--algorithm` | `nsga2` | `nsga2` (crowding distance) or `nsga3` (reference directions) |
| `--population_size` | `20` | Individuals per generation (also the number of NSGA-III reference directions) |
| `--num_generations` | `50` | Generations, including the initial population |
| `--crossover_rate` | `0.9` | Probability that a pair of parents is recombined (SBX) |
| `--mutation_rate` | `1 / genes` | Per-gene mutation probability (polynomial mutation) |
| `--threads` | config | Candidates trained concurrently; one process each, one GPU per process on CUDA |
| `--data_path`, `--limit_data_value` | config | Override `config.yaml` when given |
| `--seed` | `42` | Seeds the GA and every candidate's training |

The evaluation picks CUDA, then Apple MPS, then CPU; keep `--threads 1` on MPS (a single shared GPU). Offspring that duplicate each other or the current population are discarded before training.

### 4. Results

The final Pareto front is printed at the end. After **every generation** the search rewrites two files in `<experiment_path>` with the same content: `pareto_history.pkl` (read by `retrain.py`) and `generations.json` (readable without Python). Each generation holds:

| Key | Content |
|---|---|
| `front` | Non-dominated set of every candidate evaluated so far (global archive) |
| `hypervolume` | Hypervolume of that front (should grow over the generations) |
| `evaluated` | Every candidate trained in this generation |
| `best` | The generation's best candidate: highest validation accuracy, fewest FLOPs on ties |
| `best_so_far` | The best candidate of the whole search up to this generation |

Every candidate record has `id` (`<generation>_<index>`), `chromosome` (gene indices into `percentages`), `percentages` (heads kept per block, in %), `best_accuracy` and `total_flops`. `log.txt` has the search log, with one line per generation for the hypervolume and one for the best candidate.

The best candidate (accuracy and FLOPs) of each generation:

```bash
python -c "import json; h=json.load(open('experiment_vit/teste1/generations.json')); [print(g, r['best']['id'], 'acc=%.4f' % r['best']['best_accuracy'], 'GFLOPs=%.2f' % (r['best']['total_flops']/1e9), '| best so far:', r['best_so_far']['id']) for g, r in h.items()]"
```

The Pareto front and hypervolume per generation:

```bash
python -c "import pickle; h=pickle.load(open('experiment_vit/teste1/pareto_history.pkl','rb')); [print(g, 'HV=%.4f' % r['hypervolume'], r['front']) for g, r in h.items()]"
```

Since the files are rewritten after each generation, an interrupted search keeps every finished generation.

The search accuracy is a proxy (few epochs, a subset, the validation set). Do not report it as the final result.

### 5. Phase 3 — retrain and final test evaluation

Pick candidates from the final Pareto front (or any evaluated candidate, e.g. a generation's `best`) by their `id` and retrain each one with the **same config file** used by the search:

```bash
python retrain.py --config_file config.yaml \
    --pareto_history experiment_vit/teste1/pareto_history.pkl --id 1_3 --epochs 10
```

The candidate trains its MLPs + classifier on the train split (`--limit_data_value 0` = all 45,000 images), the epoch with the best validation accuracy is kept, and only then is the test set loaded and evaluated once. `retrain_results/<id>/results.json` holds the chromosome, the validation curve, the **test accuracy**, FLOPs and parameters; `model_state.pt` holds the weights (rebuild the pruned architecture with the same chromosome and alphas file to load them). A chromosome can also be given directly with `--chromosome 3 5 7 ...`.

## License

MIT. See the LICENSE file.
