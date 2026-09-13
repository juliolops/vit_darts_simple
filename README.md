# ViT-NAS: Multi-Objective ViT Pruning on CIFAR-10

Searches for **pruned Vision Transformers** on **CIFAR-10**, trading **accuracy** (maximize) against **FLOPs** (minimize), in two phases:

1. **DARTS** learns one importance weight (*alpha*) per attention head of a pretrained `vit_base_patch16_224`.
2. **A multi-objective genetic algorithm** ([pymoo](https://pymoo.org) NSGA-II or NSGA-III) evolves one gene per transformer block: the **percentage of heads that block keeps**, from 20% to 90%. A gene of 40% keeps the 40% of that block's heads with the largest alpha. With `prune_mlp: true` in `config.yaml`, a second gene per block sets the **percentage of MLP hidden neurons** kept, ranked by the L2 norm of their `fc2` weights (24 genes in total).

| Chromosome | Genes | Pruned | Min. FLOPs (all genes at 20%) |
|---|---|---|---|
| `prune_mlp: false` | 12 | attention heads | ~24 G |
| `prune_mlp: true` | 24 | attention heads + MLP neurons | ~6.5 G |

The unpruned ViT-Base has 85.8M parameters and ~33.7G FLOPs; the MLP accounts for about two thirds of each block's compute.

Pruning is surgical: each block's `qkv`/`proj` (and `fc1`/`fc2`) layers are rebuilt with only the survivors, so a pruned candidate really is smaller. Each candidate then fine-tunes **its MLPs and the classifier head** for a few epochs on a class-balanced CIFAR-10 subset, while the **attention layers keep their pretrained weights frozen** (the same split used by the DARTS phase), and is scored on accuracy and FLOPs. The classifier and the MLPs have separate learning rates (`learning_rate`, `mlp_learning_rate` in `config.yaml`): with 90% of heads and neurons kept, 3 epochs on 450 images reach 87% validation accuracy with an MLP learning rate of 1e-4, against 20% with 1e-3 and 73% when training the classifier alone.

## Project structure

```
├── run_darts_alphas.py        # Phase 1: learn the head alphas -> darts_alphas/*.json
├── vit_transformer_search.py  # DARTS attention wrapper + training epoch
├── run_all_evolution.py       # Phase 2: multi-objective search
├── config.yaml                # Search space, ViT and training settings
├── algorithms/
│   └── nsga.py                # pymoo NSGA-II/III ask-tell loop, Pareto archive + hypervolume
└── core/
    ├── vit.py                 # Alphas I/O + attention-head and MLP-neuron pruning
    ├── data.py                # CIFAR-10 split, balanced subset, loaders
    ├── training.py            # Fine-tune MLPs + classifier, accuracy + FLOPs
    ├── evaluation.py          # Parallel evaluation of a population
    ├── config.py              # Config loading and validation
    └── utils.py               # Seeding, logging, device selection
```

## Running a quick test

### 1. Environment

```bash
conda create -n vitnas python=3.10 -y
conda activate vitnas
pip install torch torchvision
pip install -r requirements.txt
```

CIFAR-10 (~170 MB) and the pretrained ViT weights (~350 MB) are downloaded on first use.

### 2. Phase 1 — alphas (DARTS)

```bash
python run_darts_alphas.py --epochs 1 --limit_train 500 \
    --output darts_alphas/vit_base_cifar10.json
```

This short run only checks the pipeline; use more images/epochs (`--limit_train 0` = full train set) for a meaningful head ranking.

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

The final Pareto front is printed at the end. `<experiment_path>/pareto_history.pkl` holds, per generation, the non-dominated set of every candidate evaluated so far and its hypervolume; `log.txt` has the search log:

```bash
python -c "import pickle; h=pickle.load(open('experiment_vit/teste1/pareto_history.pkl','rb')); [print(g, 'HV=%.4f' % r['hypervolume'], r['front']) for g, r in h.items()]"
```

## License

MIT. See the LICENSE file.
