# Running Experiments with the Launcher — A Practical Guide

This guide explains how to run MoQ-NAS experiments with the matrix launcher
(`launch.py`): how to write a matrix file, how the config files fit together,
how to manage GPUs (including using several GPUs for a single experiment), and
how to resume an interrupted batch so you can simply relaunch and have the
system pick up where it left off.

If you only need to launch **one** run, you can call `run_all_evolution.py`
directly (see [§7](#7-single-run-without-the-launcher)). For anything with
repeats, several algorithms, or several GPUs, use the launcher.

---

## 1. The two layers of configuration

There are two separate kinds of YAML files; do not confuse them.

| Layer | Directory | Defines |
|---|---|---|
| **Experiment config** | `experiment_configs/` | The search space (`function_dict`), algorithm hyperparameters, training settings (`max_epochs`, `optimizer`, `precision`, `eval_window_agg`), the **objectives**, and the metrics. One config = one experiment recipe. |
| **Dataset metadata** | `dataset_configs/` | Per-dataset metadata (`cifar10.yaml`, …) and the objective senses (`cfg_obj.json`, which says whether each objective is maximized or minimized). |
| **Experiment matrix** | `experiment_matrices/` | A *batch* description for the launcher: which configs/algorithms to run, how many repeats, which GPUs, and whether to resume. It points at the experiment configs above. |

A matrix never duplicates an experiment config — it **references** one per cell.

### Key experiment-config fields you will touch most

```yaml
train:
  max_epochs: 50
  epochs_to_eval: 5
  eval_window_agg: max          # max | mean | last — how best_accuracy is
                                # aggregated over the last epochs_to_eval epochs
  optimizer: AdamW
  precision: fp16               # fp32 | fp16 | bf16 (bf16 needs Ampere/Ada)
  workers_per_gpu: 5            # candidates trained concurrently PER visible GPU
                                # (or legacy `threads: N` = total workers). See §4
  multi_objective: true
  objectives: ['best_accuracy', 'total_flops']
  metrics:
    - name: Accuracy
    - name: HardwareMetrics     # produces total_params, total_flops, cuda_inference_time
    - name: ScalarizedFitness
```

- **`objectives`** are validated at startup against `dataset_configs/cfg_obj.json`
  and the configured `metrics`. A typo or an unproducible objective aborts the run
  with a clear message.
- Using deterministic objectives (`total_flops`, `total_params`) instead of the
  measured `cuda_inference_time` makes a whole run **bit-reproducible** — useful
  for comparisons and for testing resume.
- **`eval_window_agg`** only changes how the scalar proxy accuracy is computed; the
  saved model (`best_model.pth`) is always the best-val-accuracy epoch.
- **`workers_per_gpu`** sets per-GPU concurrency and memory pressure (the matrix's
  `gpus`/`gpus_per_run` is the other side of GPU usage). It can live in the config
  `train:` section or be set per batch from the matrix. Covered in detail in
  [§4](#4-managing-gpus).

---

## 2. Anatomy of a matrix file

```yaml
exp_root: experiment_cifar10_acc_flops   # root output directory
gpus: [0]                # GPU pool
gpus_per_run: 1          # GPUs assigned to each run (see §4)
repeats: 3               # how many times to repeat each experiment
seed_base: 42            # repeat i (1-based) uses seed = seed_base + i
resume: false            # or pass --resume on the command line (see §5)

defaults:                # arguments shared by every run -> --key value / --key (bool true)
  data_path: datasets/cifar10_data
  dataset: cifar10
  config_path_dataset: dataset_configs/cifar10.yaml
  log_level: INFO

experiments:
  - algo: moqnas
    config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml
    name: exp10
    overrides: {optimizer: AdamW}        # per-cell args, override defaults
    flags: [--multi_objective]           # literal flags appended verbatim
```

How values become CLI arguments (`_arg_tokens`):

- `key: value` → `--key value`
- `key: true` → `--key` (a store-true flag); `key: false` → omitted
- `key: [a, b]` → `--key a b`
- `flags: [--x, --y]` → appended literally (use this for store-true flags like
  `--multi_objective` or for the inverted `--no-...` rule flags)

Each cell expands to one `run_all_evolution.py` call; its output directory is
`<exp_root>/<algo>/<name>_repeat_<i>`. The launcher writes the exact command
(with `CUDA_VISIBLE_DEVICES`) into `launch_command.txt` in that directory.

**Always preview first** — `--dry-run` prints every expanded command and runs
nothing:

```bash
python launch.py experiment_matrices/your_matrix.yaml --dry-run
```

> Note: `population_size` and `num_generations` apply to the GA family
> (`ga`, `nsga2`, `nsga3`, `moead`). MO-QNAS reads its population from the
> config; `--num_generations` only overrides its `max_generations`.

---

## 3. Example: an accuracy + FLOPs experiment

This is the matrix shipped as `experiment_matrices/acc_flops.yaml`: the four
multi-objective algorithms on the `(best_accuracy, total_flops)` objective set,
3 repeats each, **one experiment at a time on two GPUs** (10 candidates per GPU).

```yaml
exp_root: experiment_cifar10_acc_flops
gpus: [0, 1]                  # two-GPU pool
gpus_per_run: 2              # 1 slot of 2 GPUs -> one experiment at a time, both GPUs
repeats: 3
seed_base: 42
defaults:
  data_path: datasets/cifar10_data
  dataset: cifar10
  config_path_dataset: dataset_configs/cifar10.yaml
  log_level: INFO
  multi_objective: true       # keep the config's multi-objective setting
  population_size: 20         # nsga2/nsga3/moead (ignored by moqnas)
  num_generations: 150
  workers_per_gpu: 10         # 10 candidates concurrent per GPU -> 20 in flight on 2 GPUs
experiments:
  - {algo: moqnas, config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: moqnas}
  - {algo: nsga2,  config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: nsga2}
  - {algo: nsga3,  config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: nsga3,
     overrides: {ref_divisions: 12}}
  - {algo: moead,  config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: moead,
     overrides: {ref_divisions: 12, moead_T: 20, moead_scalar: tchebycheff, moead_pneighbor: 0.9}}
```

The objective set comes from the config
(`config0_3_acc_flops.yaml` declares `objectives: [best_accuracy, total_flops]`).
To run a different objective set, point the cells at a different config — e.g.
`config0_3_flops.yaml` for `[best_accuracy, total_params, total_flops]`, or write
your own config with the `objectives` and `metrics` you need.

> `multi_objective: true` is required in `defaults` for multi-objective runs: the
> CLI flag defaults to false and would otherwise override the config's value.

Run it:

```bash
python launch.py experiment_matrices/acc_flops.yaml --dry-run   # preview
python launch.py experiment_matrices/acc_flops.yaml             # run
```

---

## 4. Managing GPUs

GPU usage has **two independent parts**. Understanding the split is the key to
using one or many GPUs per experiment:

1. **Which GPUs a run can see** — set by `CUDA_VISIBLE_DEVICES` (the launcher
   sets it per run via `gpus_per_run`; for a direct run you set it yourself).
2. **How the candidates of each generation are distributed across those visible
   GPUs** — done automatically by the evaluation scheduler in
   `core/evaluation.py`, controlled by `workers_per_gpu` (or the legacy
   `threads`) in the experiment config.

```
        matrix (launcher)                experiment config (train:)
   ┌───────────────────────────┐   ┌──────────────────────────────────┐
   │ gpus / gpus_per_run        │   │ workers_per_gpu  (or threads)    │
   │  → which GPUs are VISIBLE  │   │  → how many candidates run        │
   │    (CUDA_VISIBLE_DEVICES)  │   │    CONCURRENTLY per visible GPU   │
   └───────────────────────────┘   └──────────────────────────────────┘
```

> ⚠️ **`--gpu_list` does not select GPUs.** It only writes a line to the log.
> GPU selection is always via `CUDA_VISIBLE_DEVICES` (this was true in the
> original code too — the shell scripts set the env var). Use the launcher's
> `gpus_per_run`, or set `CUDA_VISIBLE_DEVICES` yourself for a direct run.

### 4.1 How the scheduler distributes candidates (the model)

Each generation has a population of candidate architectures to evaluate. The
scheduler:

- spawns `N` worker processes, where `N = workers_per_gpu × (number of visible
  GPUs)` (or `N = threads` total if you use the legacy key);
- pins each worker to a GPU by `gpu = worker_rank % (number of visible GPUs)`;
- feeds all candidates through a **single shared queue** (work-stealing): each
  worker trains one candidate at a time and, as soon as it finishes, pulls the
  next candidate from the queue.

So each visible GPU runs `workers_per_gpu` candidates **at the same time**, and
**as one finishes, the next one starts** — no GPU sits idle while another is
still busy. This is the same idea as the original round-robin distribution, but
with dynamic load balancing (no stragglers). Per-candidate seeding makes the
result independent of which worker/GPU trains which candidate, so results are
identical regardless of the number of GPUs.

### 4.2 `workers_per_gpu` vs `threads` — which to use

| Key | Meaning | Use it when |
|---|---|---|
| `workers_per_gpu: M` | `M` candidates concurrently **per visible GPU**; total workers = `M × n_gpus`. | You want a fixed concurrency *per GPU* that **scales with the hardware** — the same setup uses more GPUs automatically when they are available. **Recommended.** |
| `threads: T` (legacy) | `T` workers **in total**, split across the visible GPUs (`T / n_gpus` per GPU). | Back-compat with existing configs. The per-GPU concurrency changes if the number of GPUs changes. |

If both are present, `workers_per_gpu` wins. Existing configs ship with `threads`.

**Where to set them.** Either place works; the matrix value wins when present:

- In the **experiment config** under `train:` (applies wherever that config is used).
- In the **launcher matrix** (`defaults` or per-experiment `overrides`) — passed
  as `--workers_per_gpu` / `--threads`, so a batch is self-contained and you do
  not have to edit the config. This is the convenient option when the same config
  is shared across batches with different hardware.

**Worked example — "5 per GPU, refilled as they finish".** Population of 20, you
want each GPU to train 5 candidates at a time on 2 GPUs. Put everything in the
matrix:

```yaml
# launcher matrix
gpus: [0, 1]
gpus_per_run: 2        # the run sees both GPUs (CUDA_VISIBLE_DEVICES=0,1)
defaults:
  workers_per_gpu: 5   # -> --workers_per_gpu 5 (overrides the config's threads)
```

Result: `5 × 2 = 10` workers → 5 candidates concurrent on each GPU. The 20
candidates flow from the shared queue; the first 10 start, and each worker grabs
the next one the moment it finishes — exactly "as one architecture ends, another
enters". With 1 GPU the same matrix runs 5 at a time; with 4 GPUs, 5 per GPU = 20
at a time — without editing anything.

(With `threads`, you would write `threads: 10` and get 5 per GPU **only** while
exactly 2 GPUs are visible; change the GPU count and the per-GPU number changes.)

### 4.3 GPU memory / large datasets

`workers_per_gpu` is also your **memory-pressure control**: those concurrent
trainings share one GPU's VRAM. With a large dataset or large models, too many
concurrent candidates can exhaust memory:

- If the log shows CUDA out-of-memory (OOM) errors, **lower `workers_per_gpu`**
  (e.g. 5 → 3 → 2). Trade-off: less concurrency is safer but slower.
- There is a safety net: on an OOM the scheduler clears the CUDA cache and
  **retries the candidate once** before scoring it 0.0 (with an explicit error
  log). This absorbs transient memory spikes so a single OOM does not silently
  inject a zero-fitness candidate into the search — but it is not a substitute
  for tuning `workers_per_gpu` when OOM is systematic.

### 4.4 Running on one or many GPUs

**One experiment, several GPUs** — set `gpus_per_run` to how many GPUs each run
should use; the pool is split into slots of that size:

```yaml
gpus: [0, 1]
gpus_per_run: 2     # 1 slot of 2 GPUs; the run sees CUDA_VISIBLE_DEVICES=0,1
```

**Many experiments in parallel, one GPU each** (throughput-oriented):

```yaml
gpus: [0, 1, 2, 3]
gpus_per_run: 1     # (default) 4 slots -> up to 4 runs at once, one GPU each
```

**Both at once** — e.g. two experiments in parallel, two GPUs each:

```yaml
gpus: [0, 1, 2, 3]
gpus_per_run: 2     # 2 slots: [0,1] and [2,3]
```

A single GPU is never oversubscribed by two different runs: the launcher hands
each run its own slot. A failed run never stops its siblings; a summary is
printed at the end.

> When is multi-GPU-*per-run* worth it? Only when one run actually saturates a
> single GPU (large population, large models, full epochs). For many small
> candidates, running more experiments in parallel (`gpus_per_run: 1`) uses the
> hardware better, because per-run setup cost dominates over GPU compute. As a
> rule of thumb: pick `gpus_per_run` and `workers_per_gpu` so each GPU is busy
> but not OOM-ing, and use extra GPUs to run more experiments rather than to
> accelerate one small experiment.

### 4.5 Direct run (no launcher)

The same applies to a single `run_all_evolution.py` call — you set visibility
with `CUDA_VISIBLE_DEVICES` and per-GPU concurrency with the config:

```bash
CUDA_VISIBLE_DEVICES=0     python run_all_evolution.py ...   # 1 GPU
CUDA_VISIBLE_DEVICES=0,1,2 python run_all_evolution.py ...   # 3 GPUs, candidates spread over all 3
```

---

## 5. Resuming an interrupted batch

Every run — MO-QNAS and the whole GA family — writes a `checkpoint.pkl` at each
generation boundary. If a batch is interrupted (power loss, preemption, a manual
kill), **relaunch the same matrix with `--resume`**:

```bash
python launch.py experiment_matrices/acc_flops.yaml --resume
```

What happens:

1. The launcher prints a **pre-flight status** for every cell — its checkpoint's
   completed generation, or "no checkpoint (fresh)":

   ```
   [resume] checkpoint status per cell:
              gen 137  experiment_cifar10_acc_flops/moqnas/moqnas_repeat_1
   no checkpoint (fresh)  experiment_cifar10_acc_flops/nsga2/nsga2_repeat_1
   ...
   ```

2. Each cell is relaunched with `--resume`:
   - A **finished** run loads its final checkpoint and exits almost immediately
     (its generation loop has nothing left to do) — effectively skipped.
   - An **interrupted** run continues from its last saved generation, bit-identically
     to an uninterrupted run.
   - A **fresh** cell (no checkpoint) starts from generation 0.

So you do not track which experiments completed by hand: rerun the same launcher
command with `--resume` and the batch converges to completion. You can rerun it
as many times as needed.

You can also make resume the matrix default with `resume: true` at the top level;
the `--resume` flag overrides it. Without either, an existing checkpoint is
**ignored** and the run restarts from generation 0 — the safe default for a fresh
launch (so you never silently continue an old run by accident).

### Configuration guard

Resume refuses to continue if the run configuration differs from the checkpoint
(different objectives, population size, number of generations, precision, seed,
etc.) and aborts naming the differing field. Relaunch with the **same matrix** to
avoid this — that is exactly why reusing the matrix file is the intended workflow.

---

## 6. What each experiment directory contains

After (or during) a run, `<exp_root>/<algo>/<name>_repeat_<i>/` holds:

| File | Meaning |
|---|---|
| `launch_command.txt` | The exact command + `CUDA_VISIBLE_DEVICES` used (reproducibility). |
| `launcher.log` | The run's stdout/stderr (per-candidate metrics, generation summaries). |
| `checkpoint.pkl` | The resumable search state at the last generation boundary. |
| `pareto_history.pkl` | Per-generation Pareto front + hypervolume (multi-objective runs). |
| `eval_cache.pkl` | The evaluation cache (only when `--use_cache`/`use_cache: true`). |
| `log_params_evolution.txt` | The fully resolved parameters of the run. |

---

## 7. Single run without the launcher

For a one-off run, call the entry point directly. The launcher is just a wrapper
around this:

```bash
python run_all_evolution.py --algo nsga2 \
    --config_file experiment_configs/cifar_mo/config0_3_acc_flops.yaml \
    --experiment_path experiment_cifar10_acc_flops/nsga2/nsga2_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --multi_objective --population_size 20 --num_generations 150 \
    --seed 42 --log_level INFO

# Two GPUs for this single run (candidates balanced across both):
CUDA_VISIBLE_DEVICES=0,1 python run_all_evolution.py --algo nsga2 ... 

# Resume this run after an interruption:
python run_all_evolution.py --algo nsga2 ... --resume
```

`--use_cache` enables the evaluation cache (reuses metrics of architectures already
seen). The cache key includes the objectives, precision and `eval_window_agg`, so a
cached `max`/`fp16` result is never reused for a `mean`/`bf16` run.

---

## 8. Quick reference

```bash
# Preview the expanded commands
python launch.py experiment_matrices/M.yaml --dry-run

# Run a batch
python launch.py experiment_matrices/M.yaml

# Resume an interrupted batch (rerun as needed; finished cells are skipped)
python launch.py experiment_matrices/M.yaml --resume
```

| Matrix key | Purpose |
|---|---|
| `exp_root` | Root output directory. |
| `gpus` | GPU pool, e.g. `[0, 1]`. |
| `gpus_per_run` | GPUs per run (default 1). Pool is split into slots of this size. |
| `repeats` / `seed_base` | Repeats per experiment; repeat *i* uses seed `seed_base + i`. |
| `resume` | `true` to resume by default (overridden by `--resume`). |
| `defaults` | Args shared by every cell. |
| `experiments[].overrides` | Per-cell args (override `defaults`). |
| `experiments[].flags` | Literal flags appended verbatim (e.g. `--multi_objective`). |

GPU usage cheat-sheet (see [§4](#4-managing-gpus) for the full model):

| Want | Set |
|---|---|
| One experiment on N GPUs | matrix `gpus_per_run: N` (or `CUDA_VISIBLE_DEVICES` for a direct run) |
| Many experiments in parallel, 1 GPU each | matrix `gpus: [0,1,2,3]`, `gpus_per_run: 1` |
| K candidates concurrent per GPU (scales with #GPUs) | `workers_per_gpu: K` (config `train:` or matrix `defaults`) |
| Fixed total #workers across GPUs (legacy) | `threads: T` (config `train:` or matrix `defaults`) |
| Fewer OOMs on a large dataset | lower `workers_per_gpu` |
