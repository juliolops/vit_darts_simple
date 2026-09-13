# ViT-NAS: Multi-Objective Attention-Head Pruning on CIFAR-10

Searches for **pruned Vision Transformers** on **CIFAR-10**, trading **accuracy** (maximize) against **FLOPs** (minimize), in two phases:

1. **DARTS** learns one importance weight (*alpha*) per attention head of a pretrained `vit_base_patch16_224`.
2. **A multi-objective genetic algorithm** evolves one gene per transformer block (12 genes): the **percentage of heads that block keeps**, from 20% to 90%. A gene of 40% keeps the 40% of that block's heads with the largest alpha.

Pruning is surgical: each block's `qkv`/`proj` layers are rebuilt with only the surviving heads, so a pruned candidate really is smaller. Each candidate is then fine-tuned with **only the classifier head trainable** for a few epochs on a class-balanced CIFAR-10 subset, and scored on accuracy and FLOPs.

## Project structure

```
├── run_darts_alphas.py        # Phase 1: learn the head alphas -> darts_alphas/*.json
├── vit_transformer_search.py  # DARTS attention wrapper + training epoch
├── run_all_evolution.py       # Phase 2: multi-objective search
├── config.yaml                # Search space, ViT and training settings
├── algorithms/
│   ├── nsga.py                # GA: rank tournament, crossover/mutation, NSGA-II selection, Pareto archive
│   └── pareto.py              # Dominance, non-dominated sort, crowding, hypervolume
└── core/
    ├── vit.py                 # Alphas I/O + attention-head pruning
    ├── data.py                # CIFAR-10 split, balanced subset, loaders
    ├── training.py            # Fine-tune the classifier, accuracy + FLOPs
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
    --experiment_path experiment_vit/teste1 \
    --population_size 4 --num_generations 3 \
    --limit_data_value 500 --threads 1 --seed 42
```

Flags: `--population_size`, `--num_generations`, `--crossover_rate` (0.9), `--mutation_rate` (0.05), `--seed` (42), `--log_level`. `--data_path`, `--limit_data_value` and `--threads` override `config.yaml` when given. The evaluation picks CUDA, then Apple MPS, then CPU; keep `--threads 1` on MPS.

### 4. Results

The final Pareto front is printed at the end. Per generation, `<experiment_path>/pareto_history.pkl` holds the Pareto archive and its hypervolume, and `log.txt` the search log:

```bash
python -c "import pickle; h=pickle.load(open('experiment_vit/teste1/pareto_history.pkl','rb')); [print(g, 'HV=%.4f' % r['hypervolume'], r[1]) for g, r in h.items()]"
```

## License

MIT. See the LICENSE file.
