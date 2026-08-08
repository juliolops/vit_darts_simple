# MoQ-NAS Improvement Strategy

This document is the deliverable of a code-level audit of the MoQ-NAS repository (multi-objective, quantum-inspired NAS over a 10,000-sample CIFAR-10 subset). It covers six improvement areas — evaluation caching, experiment launching, configurable objectives, FP16→BF16 migration, training parallelization, and checkpoint/resume — and for each one identifies the implicated files/classes/functions, states the structural problem, proposes the minimal viable strategy with alternatives and tradeoffs, and lays out a surgical step-by-step plan with observable verification criteria. No implementation code is included; each section is self-contained enough to be implemented independently, with cross-references where areas interact (cache↔precision, cache↔checkpointing, launcher↔resume).

Conventions used throughout:
- `file:line` references are against the current `refactor/update-2026-staged` branch (post-refactor: `algorithms/pareto` package exists, `utils/helpers.py` is a facade, configs live in `experiment_configs/` and `dataset_configs/`).
- `[ASSUMPTION: ...]` flags behavior inferred but not exhaustively traced; if wrong, the affected step says so in its Risks section.
- Reproducibility context that the strategies rely on: per-candidate deterministic seeding exists (`utils/seeding.py::seed_candidate`, called in `core/evaluation.py::run_individuals`), so per-candidate `best_accuracy`/`total_params` are bit-exact across runs and thread counts. `cuda_inference_time` is measured wall time and is **not** reproducible run-to-run; any strategy that compares end-to-end runs must compare accuracy/params, not time-dependent fronts.

---

## Area 1: Modular Evaluation Cache

### Current State Analysis

**Evaluation entry point.** `core/evaluation.py::EvalPopulation.__call__(decoded_params, decoded_nets, generation)` is the single component that actually trains candidates (forks `threads` worker processes, each calling `core/cnn/master.py::fitness`). It is cache-free by design and returns `{candidate_id: {metric_name: value}}`.

**The duplication.** Three near-identical caching layers sit *above* `EvalPopulation`, one per algorithm family:

| Site | Cache attr | Key | Value | Persistence |
|---|---|---|---|---|
| `algorithms/ga/base_ga.py` (`GA.__init__` ~:80-93; `_evaluate_*` ~:232-290) | `self.evaluated` + `self.eval_history` | `tuple(individual.tolist())` — **network chromosome only** | scalar fitness, averaged over up to 3 runs (`eval_history` collects raws) | `utils.io.load_cache`/`backup_cache` → `<exp>/cache_backup.pkl` |
| `algorithms/ga/nsga2.py` (`NSGA2.__init__` :38-39; eval at :88-105, registry at :112-150) | `self.eval_cache` | `tuple(pop[i])` (net chromosome) | full metric dict (multi-objective) | none for `eval_cache`; separate `unique_networks.pkl` registry |
| `algorithms/qnas/qnas2.py` (`QNAS`; `_eval_pop_with_cache` :531-580, `_eval_pop_without_cache` :414-480) | `self.evaluated` + `self.eval_history`-like averaging | `tuple(individual_net_array)` | scalar fitness, averaged over `num_runs` | `cache_backup.pkl` (load at :110, backup in `go_next_gen` qnas2:830) plus `unique_networks_db` → `unique_networks.pkl` |

`MOQNAS` (`algorithms/qnas/moqnas.py`) maintains `unique_networks_db` (:51-56) but routes evaluation through its own `multiobjective_fitness`; nsga3/moead inherit NSGA2's path.

**Structural problems.**
1. Three copies of "check key → reuse → else evaluate → store" with diverging semantics (scalar vs metric-dict values; averaging vs single-shot). The averaging-over-3-runs behavior exists in `base_ga` and `qnas2` but not `nsga2`.
2. The key is the **network chromosome alone**. It ignores: (a) the continuous hyperparameter chromosome (`base_ga.pop_params`, qnas2 `qpop_params`) — two candidates with the same layers but different evolved hyperparameters collide; (b) the training configuration (epochs, dataset, `limit_data_value`, optimizer, precision) — a cache file copied between experiments would silently poison results; (c) the seed policy.
3. The enable toggle (`--use_cache` in `run_all_evolution.py`, default False) is not honored uniformly: it is passed to GA/NSGA2/NSGA3/MOEAD/QNAS constructors but **not** to `MOQNAS` (see `run_all_evolution.py` moqnas branch — no `use_cache` kwarg).

`[ASSUMPTION: the decoded continuous hyperparameters actually influence trained fitness (they parameterize training/architecture); if some experiment configs evolve no continuous params (`params_ranges` empty), the current net-only key is accidentally correct for those runs.]`

### Proposed Strategy

Introduce **one** cache as a thin wrapper at the boundary every algorithm already shares: the `eval_func` callable handed to each algorithm by `run_all_evolution.py::_bootstrap`. A `CachedEvaluator` object wraps `EvalPopulation`, exposes the same `__call__(decoded_params, decoded_nets, generation)` signature, and filters the batch: cached candidates are answered from storage, the remainder is forwarded to the wrapped `EvalPopulation`, results are merged and stored. Algorithms keep their existing internal caches untouched initially; they are deleted in a final cleanup step only after parity is demonstrated. This is the Block-C/Block-D pattern already proven in this repo: create the canonical implementation, verify parity, then remove duplicates.

**Key design.** A composite, explicit key:
`(net_chromosome_tuple, hyperparam_tuple_rounded, eval_fingerprint)` where `eval_fingerprint` is a short hash of the evaluation-relevant config: dataset name, `limit_data_value`, `max_epochs`, optimizer, batch size, `precision` (see Area 4 — FP16 and BF16 entries must never collide), and the global seed policy. The fingerprint is computed once per run from `config.train_spec` and stored alongside the cache so a mismatched file is detected, not silently reused.

**Value design.** Always the full metric dict (the superset). Scalar consumers (GA/QNAS) read `objectives[0]` from it — this unifies the scalar/dict divergence. The averaging-over-N-runs policy becomes a cache option (`cache_avg_runs`, default 1) so existing GA/QNAS behavior is reproducible; note that post-0.6 determinism makes re-running the same candidate bit-identical, so averaging now only makes sense if per-run seeds are varied — surface this to the user rather than silently keeping it. **Simpler alternative considered:** keep per-algorithm caches and only fix the keys — rejected because it triples every future change (precision key, Area 4; resume reconstruction, Area 6).

**Storage backend.** File-based pickle dict per experiment (`<experiment_path>/eval_cache.pkl`), written atomically (temp + `os.replace`, the pattern already in `utils/io.py::_atomic_write_yaml`). Rationale: populations are ~20-50 candidates/generation and runs are ≤150 generations → at most a few thousand entries; SQLite adds a dependency surface for zero benefit at this scale; pure in-memory loses the cross-restart value that motivates the cache. If cross-experiment sharing is ever wanted, the fingerprint-in-key design already makes a shared file safe — do not build it now.

**Toggle.** Reuse the existing `--use_cache` CLI flag (already parsed, already in the shell scripts) — no new surface. Wire it so `_bootstrap` returns either `EvalPopulation` or `CachedEvaluator(EvalPopulation)`; algorithms receive whichever and stay oblivious. When disabled, behavior is byte-identical to today (the wrapper is simply not constructed).

### Step-by-Step Plan

```
Step 1: Characterize current cache hit semantics
Files affected: none (read-only)
Change scope: write a short report (in the PR description) of the three cache
  implementations' key/value/averaging differences, and confirm whether any
  config in experiment_configs/ evolves continuous params (params_ranges).
Verify: report lists the three sites with line numbers and states, per config
  file, whether hyperparams are part of the search space.

Step 2: Create core/eval_cache.py with CachedEvaluator
Files affected: core/eval_cache.py (new)
Change scope: wrapper class with __call__ matching EvalPopulation's signature;
  composite key as designed above; atomic pickle persistence; stats counters
  (hits/misses) logged per generation. No algorithm file is touched.
Verify: unit-style script: wrap a stub eval_func, call twice with the same
  batch → second call performs zero stub invocations and returns equal dicts;
  changing only the fingerprint forces re-evaluation.

Step 3: Wire the wrapper in run_all_evolution._bootstrap behind --use_cache
Files affected: run_all_evolution.py
Change scope: construct CachedEvaluator when args['use_cache'] is true; pass it
  in place of eval_pop to ALL engines including MOQNAS. Per-algorithm use_cache
  constructor args keep their current default behavior but are forced False
  when the wrapper is active (one cache, not two).
Verify: (a) smoke run nsga2 seed 42 pop 4 WITHOUT --use_cache is bit-exact vs
  .refactor_baseline/expB_run1.log; (b) WITH --use_cache, run twice with the
  same experiment_path: second run's log shows hit-count == population size
  and zero "Starting the Generation ... individuals" training time.

Step 4: Parity check against legacy caches, then remove them
Files affected: algorithms/ga/base_ga.py, algorithms/ga/nsga2.py,
  algorithms/qnas/qnas2.py
Change scope: delete the internal cache branches and the cache_backup.pkl
  load/save calls; keep the unique_networks registry (it is a result artifact,
  not a cache). Nothing else in selection/evolution logic changes.
Verify: 2-generation runs (pre-deletion vs post-deletion, both with
  --use_cache) produce bit-exact per-candidate accuracy/params; grep shows no
  remaining `self.evaluated[`/`self.eval_cache[` outside the new module.
```

### Risks and Open Questions

- **Averaging semantics change.** GA/QNAS users relying on 3-run averaging get single-shot values by default. Post-0.6 determinism makes 3 identical runs pointless, but if someone re-enables nondeterminism (e.g., disables seeding), averaging matters again. Mitigation: `cache_avg_runs` option + changelog note.
- **Key correctness depends on Step 1's finding** about continuous hyperparameters. If `decoded_params` contain non-hashable/derived entries, the hyperparam tuple must be canonicalized (sorted keys, rounded floats) — rounding granularity is an open question (proposal: 1e-6).
- **Cross-area:** the fingerprint must include the precision string from Area 4 and is consulted during Area 6 resume (see those sections). Area 5's value increases when parallel slots are memory-limited: cached candidates free slots.

---

## Area 2: Robust Experiment Launching

### Current State Analysis

Six launcher scripts exist at the repo root: `run_ea_1.sh`, `run_qnas_1.sh`, `run_moqnas_1.sh`, `run_fair_mo.sh`, `run_fairness_baseline.sh`, `run_retrain.sh`. The two main families (`run_ea_1.sh`, `run_moqnas_1.sh`) share ~80% of their bodies.

**What varies across scripts:** algorithm list (`algos` array), dataset + derived paths, config file list (`configs` array) and parallel `exps`/`cuda_devices` arrays, repeat count (`num_repeats`), GA hyperparameters (population, generations, rates, mutation strategy), QNAS toggles (`elite_mode`, `ref_dir_method`, no-op rule flags), and boolean toggles (`use_cache`, `early_stopping`, `elitism`, `data_augmentation`, `multi_objective`).

**Common boilerplate:** the `COMMON_ARGS` array assembly, the triple-nested loop (configs × algos × repeats), `exp_path` construction (`<exp_root>/<algo>/<exp>_repeat_<i>`), `mkdir -p`, and `CUDA_VISIBLE_DEVICES` prefixing.

**Structural problems.**
1. The bash boolean idiom `$($use_cache && echo --use_cache)` is fragile (word-splitting, silently drops flags if the variable is misspelled).
2. `run_moqnas_1.sh:78` passes `--gpu_list "${cuda_devices}"` — expanding an **array** name yields only its first element; a multi-GPU `cuda_devices` list is silently truncated. Latent bug.
3. Repeats run **sequentially inside one script**; "parallelism" is achieved by hand-editing copies of the script with different `cuda_devices` — that is exactly the proliferation to remove.
4. `--seed` (added in stage 0.5) is not plumbed into any script: repeats `i=1..N` currently differ only by accident of nondeterminism that the 0.6 work removed for accuracy/params. **Repeats need explicit per-repeat seeds to be meaningfully different.** This is the most urgent correctness issue in this area.
5. Adding a new objective combination (Area 3) today means copying a config file *and* a script.

### Proposed Strategy

**Recommended: experiment-matrix YAML + a small Python launcher** (`launch.py`, stdlib `argparse` + `yaml` + `subprocess`), generating one `run_all_evolution.py` invocation per (config × algo × repeat) cell and scheduling them over a declared GPU slot list.

Why this over the alternatives:
- *Parameterized shell script:* fixes nothing structural — bash arrays/booleans remain the failure mode (see problems 1-2), and parallel scheduling in bash is where scripts go to die.
- *Hydra/Sacred:* solves config composition and sweeps well, but imposes its own config layout, working-directory semantics, and a dependency, while the repo already has a config system (`core/config.py` + `experiment_configs/*.yaml`) that `run_all_evolution.py` consumes. Migrating that to Hydra is a broad refactor for marginal gain at this scale (single machine, ≤2 GPUs, one user). Not warranted now; the matrix-YAML format below is forward-compatible with a later Hydra migration if multi-node arrives.
- *Plain Python CLI without a matrix file:* already exists (`run_all_evolution.py`); the gap is the sweep/repeat/GPU-scheduling layer, not the per-run CLI.

**Matrix file shape (pseudocode, clarifies the design only):**

```yaml
defaults:        # everything COMMON_ARGS encodes today
  dataset: cifar10
  config_path_dataset: dataset_configs/cifar10.yaml
  limit_data_value: 10000
  log_level: INFO
gpus: [0, 1]     # slot pool; one process per slot at a time
repeats: 3
seed_base: 42    # repeat i runs with seed_base + i  → explicit, logged
experiments:
  - {algo: moqnas, config: experiment_configs/cifar_mo/config0_2.yaml, name: exp10}
  - {algo: nsga2,  config: experiment_configs/cifar_mo/config0_3.yaml, name: exp4,
     overrides: {population_size: 20, num_generations: 150}}
```

The launcher: expands the matrix, derives `experiment_path` exactly as the scripts do today (so existing analysis tooling keeps working), assigns each pending run to the next free GPU slot (`CUDA_VISIBLE_DEVICES=<slot>`), streams each run's stdout to `<experiment_path>/launcher.log`, records the fully-expanded command line into the experiment dir (reproducibility artifact), and supports `--dry-run` (print commands, run nothing). Resume integration: `--resume` and checkpoint path (Area 6) are just per-experiment keys in the matrix, so relaunching after preemption = rerun the same launcher command.

The existing `.sh` scripts are kept for one release as thin 3-line wrappers calling the launcher with an equivalent matrix file, then archived.

### Step-by-Step Plan

```
Step 1: Inventory and freeze current behavior
Files affected: none (read-only)
Change scope: extract from the 6 scripts the full set of varying parameters and
  produce matrix files reproducing each script's current cells verbatim
  (including the gpu_list bug NOT being reproduced — document it).
Verify: a table mapping every script variable to a matrix key; reviewers can
  diff it against the scripts.

Step 2: Implement launch.py (sequential mode first)
Files affected: launch.py (new), experiment_matrices/*.yaml (new)
Change scope: matrix parsing, command expansion, exp_path derivation, per-run
  command logging, --dry-run. No parallelism yet. run_all_evolution.py is NOT
  modified.
Verify: `python launch.py experiment_matrices/moqnas_smoke.yaml --dry-run`
  prints exactly the command the current run_moqnas_1.sh would execute
  (string-diff against a captured `set -x` trace of the script), with seeds
  added.

Step 3: Add GPU-slot parallel scheduling
Files affected: launch.py
Change scope: a slot pool sized by the `gpus` list; independent runs (different
  experiment_path) execute concurrently, one per slot; failures of one run do
  not kill siblings; summary table at the end.
Verify: a 4-cell smoke matrix on 2 slots finishes in ~half the sequential wall
  time; all 4 experiment dirs contain results; killing one child mid-run still
  yields 3 completed runs + 1 reported failure.

Step 4: Per-repeat seeds and reproducibility check
Files affected: launch.py, experiment_matrices/*
Change scope: seed_base + repeat-index policy; seed recorded in the saved
  command line and passed via --seed.
Verify: two launches of the same matrix produce per-candidate accuracy/params
  bit-exact per repeat (compare repeat_1 vs repeat_1), while repeat_1 vs
  repeat_2 differ (different seed) — both checked by grep-diff of candidate
  lines, as in .refactor_baseline/expB.sh.

Step 5: Convert the 6 scripts to wrappers; archive later
Files affected: run_*.sh
Change scope: each becomes `python launch.py experiment_matrices/<name>.yaml "$@"`.
  No deletion yet.
Verify: `bash -n` passes; running run_moqnas_1.sh executes the same cells as
  before (Step 2's diff re-run through the wrapper).
```

### Risks and Open Questions

- **Seed-per-repeat changes experimental semantics**: today's repeats differ only via timing nondeterminism. Decision needed from the experiment owner: is `seed_base + i` the intended definition of a "repeat"? (Recommended: yes; otherwise repeats measure only `cuda_inference_time` noise.)
- `CUDA_VISIBLE_DEVICES` slotting assumes one run saturates one GPU. With `threads: 20` configs, two runs sharing a GPU would contend; the launcher must never oversubscribe a slot. Multi-GPU *within* one run (Area 5) is orthogonal: such a run would declare it needs N slots.
- **Cross-area:** Area 6's `--resume` must be expressible per-cell; Area 3's objective sets live in the experiment config, so the matrix only points at config files — no launcher change needed when objectives change.

---

## Area 3: Configurable Multi-Objective Evaluation

### Current State Analysis

**Where objectives are declared.** Per-experiment config, `train:` section: `objectives: [best_accuracy, total_params, cuda_inference_time]` and a `metrics:` list (e.g. `Accuracy`, `HardwareMetrics`, `ValidationLossFitness`, `ScalarizedFitness`) — see `experiment_configs/cifar_mo/config0_3.yaml`. Objective **senses** are declared separately in `dataset_configs/cfg_obj.json` (`accuracy: maximize`, `inference/flops/params/energy: minimize`, plus fairness entries) and resolved by **substring matching** in `algorithms/ga/nsga2.py::__init__` (:53-70, `if key in active_obj`) and `algorithms/qnas/moqnas.py` (same pattern, :227+), producing `self.objective_senses`.

**Where objectives are computed.** The pluggable metrics system: `core/cnn/metrics/hardware.py::HardwareMetrics` already returns `cuda_inference_time` (µs, CUDA-event timing in `base_hardware.py::measure_inference_time`), `total_params`, **and `total_flops`** — FLOPs are already implemented via a manual forward-hook counter (`base_hardware.py::measure_flops`) covering `Conv2d`, `Conv1d`, `Linear`, returning 0 for unsupported layers. `core/cnn/trainer.py` instantiates metrics from the config (`master.py::create_metrics_from_config`), and `core/evaluation.py::EvalPopulation` extracts `train_params['objectives']` as the metric names it reports.

**Downstream consumers of the objective vector.** The `algorithms/pareto` package (post-Block-D) is fully parametrized: `dominates`/`fast_nondominated_sort`/`to_minimization`/`compute_hypervolume_mixed` take `objective_senses` and operate on (N, M) arrays for arbitrary M; `nsga3._build_reference_directions` and moead's variant derive direction counts from M automatically. The fairness path in `EvalPopulation` special-cases objectives containing the substring `'fairness'` (serial GPU evaluation).

**So the abstraction already exists.** Substituting FLOPs for inference time is, in principle, a config-only edit: `objectives: [best_accuracy, total_flops]` — `'flops' in 'total_flops'` matches the `cfg_obj.json` sense, `HardwareMetrics` already emits the value, and the Pareto operators are M-agnostic.

**The actual gaps:**
1. **No validation**: a typo (`total_flop`) or an objective with no matching sense fails late or silently (NSGA2 raises only if *no* key matches; `[ASSUMPTION: an objective matching multiple cfg_obj keys — e.g. a hypothetical name containing both 'params' and 'flops' — takes the first match; the loop order makes this nondeterministic across dict versions]`).
2. **`ScalarizedFitness` hardcodes the objective set**: its `params` block (`max_params: 114090`, `max_inference_time: 1000`) normalizes exactly params+time. Any experiment using `fitness_metric: scalar_multi_objective` with FLOPs needs a corresponding `max_flops` normalizer — currently absent.
3. **FLOPs hook coverage** excludes BN/pooling/activations; counts are therefore lower bounds. For *ranking* architectures this is acceptable (the error is consistent across candidates of the same operator family) but must be documented; if exact MACs matter, an external counter is needed.
4. The hypervolume reference point and plotting helpers (`utils/visualization.py`) assume the 3-column [acc, params, time] layout in some labels — cosmetic but misleading for other sets.

**FLOPs library choice.** The manual hook counter is already present, dependency-free, deterministic, and tested in production logs. `thop`/`ptflops` patch modules at runtime and are fragile with the dynamically-decoded layer list this repo builds (`core/cnn/model.py` from `fn_dict`); `fvcore` is the most accurate but pulls a heavyweight dependency. **Recommendation: keep the existing hook counter, extend its coverage list if needed** (it lives in one function), and only revisit if absolute MACs (not relative ranking) become a requirement.

### Proposed Strategy

Treat this as a **validation and de-hardcoding** task, not an abstraction-building task — the abstraction (config-declared objective names → metrics plugins → sense map → M-agnostic Pareto ops) already exists and survived the Block-D parity verification.

1. Add an explicit objective registry check at config-parse time (`core/config.py`): every name in `train.objectives` must resolve to exactly one `cfg_obj.json` sense (exact-match table preferred over substring; keep substring as fallback with a warning) and must be produced by at least one configured metric. Fail fast with the list of valid names.
2. Make `ScalarizedFitness` normalization generic: a `normalizers: {objective_name: max_value}` mapping in its `params`, replacing the two hardcoded keys. Existing configs keep working via a back-compat translation (`max_params` → `normalizers.total_params`).
3. Document (in the config comments and README) the three canonical sets — `[best_accuracy, total_params, cuda_inference_time]` (current), `[best_accuracy, total_params, total_flops]`, `[best_accuracy, total_flops]` — and ship one example config for each. No code may special-case any of them.
4. Cross-reference: the objective set string becomes part of the cache fingerprint (Area 1) and the checkpoint config block (Area 6).

**Alternative considered:** introducing an `Objective` class hierarchy (name, sense, compute fn). Rejected: it would duplicate what `metrics` + `cfg_obj.json` already encode and touch every algorithm constructor — a speculative abstraction with no current consumer.

### Step-by-Step Plan

```
Step 1: Trace and document the objective pipeline
Files affected: none (read-only)
Change scope: one diagram/table in the PR: config name → metric producer →
  sense resolution → consumers (EvalPopulation filter, objective_senses,
  ScalarizedFitness). Confirm the substring-multimatch assumption.
Verify: table reviewed; a deliberately misspelled objective demonstrates the
  current failure mode (capture the traceback as the "before" evidence).

Step 2: Objective validation at config parse
Files affected: core/config.py
Change scope: a _check_objectives step inside the existing _check_vars flow:
  exact-match against cfg_obj.json keys-extended-to-metric-names, error lists
  valid names. No behavior change for currently-valid configs.
Verify: current smoke config parses unchanged (bit-exact smoke vs
  expB_run1); a config with `total_flop` aborts before dataset download with
  the explicit message.

Step 3: Generalize ScalarizedFitness normalizers
Files affected: core/cnn/metrics/fitness.py, experiment_configs (comments only)
Change scope: normalizers mapping + back-compat for max_params /
  max_inference_time. Scalar value identical for old configs.
Verify: unit check: old-style params produce the same scalar as before on a
  fixed metric dict; new-style with total_flops produces finite scalars.

Step 4: Ship and verify the three objective-set example configs
Files affected: experiment_configs/cifar_mo/ (3 new yaml files)
Change scope: copies of config0_3.yaml differing ONLY in objectives (+
  normalizers where relevant). No source change.
Verify: 1-generation moqnas smoke per config completes; logs show the right
  metric columns; for the FLOPs sets, two runs with the same seed have
  bit-exact total_flops per candidate (FLOPs is deterministic, unlike time) —
  this also demonstrates the Pareto front itself becomes reproducible when
  the time objective is dropped, which is worth recording for Area 6 testing.

Step 5: Label cleanup in visualization helpers
Files affected: utils/visualization.py
Change scope: axis/label strings derived from objective names instead of the
  hardcoded [acc, params, time]; numeric logic untouched.
Verify: plot functions run on a synthetic 2-objective history without index
  errors.
```

### Risks and Open Questions

- **Hook-counter coverage**: if future operator families (attention, depthwise variants) enter `fn_dict`, the FLOPs lower-bound bias may stop being uniform across candidates, distorting ranking. Open question for the research owner: is relative ranking sufficient, or are absolute MACs needed for publication? The strategy holds for the former; the latter swaps Step 3's counter for `fvcore` behind the same metric name.
- The substring sense-matching ambiguity (`[ASSUMPTION]` above) is removed by Step 2's exact matching; if any *existing* experiment relied on substring tricks, Step 2's fallback warning will surface it rather than break it.
- **Cross-area:** dropping `cuda_inference_time` makes whole-run results bit-reproducible — Areas 5 and 6 should use a FLOPs-based config for their end-to-end verification to get bit-exact front comparisons.

---

## Area 4: Migration from FP16 to BF16

### Current State Analysis

**Every AMP/precision touchpoint found** (repo-wide grep for `float16|autocast|GradScaler|\.half\(\)|bfloat`):

1. `core/cnn/trainer.py:18` — `from torch.amp import GradScaler, autocast`.
2. `core/cnn/trainer.py:99` — `self.scaler = GradScaler(self.device.type, enabled=self.params.get('mixed_precision', False))`. **Instantiated unconditionally, gated only by `enabled=`.**
3. `core/cnn/trainer.py:150` — `with autocast(self.device.type, dtype=torch.float16, enabled=self.params.get('mixed_precision', False))`. **dtype hardcoded to `torch.float16`.**
4. `core/cnn/trainer.py:182-186` — `scaler.scale(loss).backward(); scaler.unscale_(optimizer); [clip]; scaler.step(); scaler.update()` — invoked on every step (a disabled scaler degrades to pass-through, but the call sites exist unconditionally).
5. `core/cnn/metrics/fairness.py:48-56` — `_autocast_kwargs()` **already auto-selects BF16** when `torch.cuda.is_bf16_supported()`, else FP16, unconditionally enabled on CUDA. This means today a `mixed_precision: true` run on an L40S trains in FP16 but evaluates fairness in BF16 — an inconsistency to fold into the single config point.
6. No `.half()` / `.to(torch.float16)` casts exist on models, inputs, or targets anywhere in live code (verified by grep; `old_files/` excluded).

**Configuration point.** A single boolean: `train.mixed_precision` in the experiment YAML (e.g. `config0_3.yaml:73`), read in the two trainer lines above and logged by `run_all_evolution._bootstrap`. So precision is *almost* centralized already — one boolean read in two places plus the independent fairness heuristic.

`[ASSUMPTION: inference-time measurement (base_hardware.measure_inference_time) runs outside autocast in FP32; if it actually inherits an autocast context, precision would also affect the cuda_inference_time objective — verify during Step 1.]`

### Proposed Strategy

Replace the boolean with a single string key `train.precision: fp32 | fp16 | bf16`, resolved **once** into a small precision policy (dtype for autocast, autocast enabled flag, scaler enabled flag) inside the trainer, and consumed by the three touchpoints (trainer autocast, trainer scaler, fairness autocast). Per the given rationale, BF16 removes the gradient-underflow noise that FP16 injects into short-budget heterogeneous training — i.e., it removes evaluation-score noise from the search signal.

Required properties (from the brief, mapped to mechanics):
- **Back-compat:** `mixed_precision: true` maps to `precision: fp16`, `false`/absent to `fp32`, with a deprecation log line. Existing configs and the seeded baselines remain valid and reproducible without edits.
- **Scaler gating:** the scaler is constructed with `enabled=(precision == 'fp16')` and — to make the no-scaler path explicit rather than relying on pass-through semantics — the backward/step block branches: fp16 path keeps scale/unscale/clip/step/update; bf16/fp32 path does plain `loss.backward(); clip; optimizer.step()`. The gradient-clipping behavior present today (post-unscale clipping) must be preserved identically in both branches.
- **Hardware check:** if `precision == 'bf16'` and (`device` is CUDA and not `torch.cuda.is_bf16_supported()`), raise `RuntimeError` naming the device and the requested precision at trainer construction — never silent fallback. (CPU autocast bf16 is allowed by torch; decide in Step 2 whether to permit it for debug runs — recommended: yes, with a warning.)
- **Fairness consistency:** `fairness.py::_autocast_kwargs` takes its dtype from the same policy instead of its own heuristic, so training and fairness evaluation always agree.
- **Cache interaction (Area 1):** the precision string is a mandatory component of the evaluation-cache fingerprint. An FP16 cache entry must never satisfy a BF16 request. If Area 1 lands first, this is one fingerprint field; if Area 4 lands first, note it in `core/eval_cache.py`'s spec. The same string goes into the Area 6 checkpoint config block.

**Reproducibility note (important for this repo's verification discipline):** changing precision changes trained accuracy values. All existing bit-exact references (`expB_run1`, `expC`) were produced under fp16. Verification of this area must therefore be *self-referential* (fp16 path unchanged → bit-exact vs old references; bf16 path → new references generated and committed to `.refactor_baseline/`).

### Step-by-Step Plan

```
Step 1: Audit confirmation + inference-time autocast check
Files affected: none (read-only)
Change scope: confirm the 6 touchpoints above are exhaustive (re-grep incl.
  retrain_model.py / retrain_parallel.py / scripts/), and trace whether
  measure_inference_time executes under autocast.
Verify: grep output attached to PR; a one-off debug print of
  torch.is_autocast_enabled() inside measure_inference_time during a smoke
  run answers the [ASSUMPTION].

Step 2: Introduce the precision policy in the trainer
Files affected: core/cnn/trainer.py, core/config.py (accept the new key +
  back-compat mapping), experiment config comments
Change scope: parse precision once in BaseTrainer.__init__; replace the two
  hardcoded reads; branch the backward block; add the bf16 hardware check.
  Nothing else in the training loop changes.
Verify: (a) fp16 regression: smoke run with mixed_precision: true config is
  bit-exact vs expB_run1 (proves the refactor of the backward block is
  inert for fp16); (b) precision: bf16 on the L40S completes a smoke run
  with zero GradScaler invocations (assert via a temporary counter or by
  constructing no scaler in that branch); (c) precision: bf16 with
  CUDA_VISIBLE_DEVICES pointed at unsupported hardware (or monkeypatched
  is_bf16_supported) raises the explicit RuntimeError.

Step 3: Unify fairness autocast with the policy
Files affected: core/cnn/metrics/fairness.py, the param plumbing in
  core/evaluation.py::evaluate_fairness_parallel_cuda (fairness_params)
Change scope: dtype comes from train_spec precision; the is_bf16_supported
  heuristic remains only as a guard, not a selector.
Verify: fairness smoke (run_fair_mo.sh config) logs the same dtype for
  training and fairness; fp16 fairness numbers match a pre-change run.

Step 4: Generate BF16 baselines and document
Files affected: .refactor_baseline/ (new logs), REFACTOR_GUIDE/README note
Change scope: seeded bf16 nsga2 + moqnas reference runs stored alongside the
  fp16 ones; doc table of which baseline corresponds to which precision.
Verify: two consecutive bf16 runs with seed 42 are bit-exact in per-candidate
  accuracy/params (bf16 does not break the 0.6 determinism guarantees);
  fp16 vs bf16 accuracies differ (expected, recorded, not "fixed").
```

### Risks and Open Questions

- **Does bf16 preserve bit-determinism?** Expected yes (deterministic kernels + same seeding), but Step 4's double-run check is the gate. If bf16 kernels are nondeterministic on this stack, the strategy still stands but Area-6 testing must use FLOPs-objective configs (see Area 3 cross-ref).
- **Accuracy drift:** BF16's 7 mantissa bits can shift short-budget accuracies by more than FP16's underflow noise removed — the search *signal* improves per the rationale, but absolute numbers move. Flag to the research owner that cross-precision comparisons of historical experiments are invalid; the cache fingerprint enforces this mechanically.
- **Cross-area:** fingerprint field (Area 1), checkpoint config block (Area 6) — both must treat `precision` as identity-defining, not metadata.

---

## Area 5: Training Parallelization Opportunities

### Current State Analysis

**Execution model today.** Population evaluation is already process-parallel: `core/evaluation.py::EvalPopulation.__call__` (:71-119) partitions the population round-robin into `train_params['threads']` batches (:76-82), forks one `torch.multiprocessing.Process` per batch (:90-99), assigns each worker a GPU by `thread_id % num_gpus` (:172-177), and collects results via a shared `mp.Queue`. Production configs use `threads: 20` on a single GPU — i.e., up to 20 candidate trainings time-share one device. Within each worker, candidates run **sequentially** through `master.fitness`. Fairness evaluation has its own second-stage parallel pool (`evaluate_fairness_parallel_cuda`, spawn context, `processes_per_gpu=10`).

**Independence structure.** Candidate trainings within one generation are mutually independent — and, since stage 0.6, *provably* order-independent: `seed_candidate(global_seed, generation, candidate_id)` + per-candidate DataLoader-generator reseeding make each candidate's result a pure function of `(seed, generation, candidate_id, architecture)`, verified bit-exact across `threads ∈ {2,4,20}`. The hard dependency is the **generation barrier**: selection (NSGA-II environmental selection, MOEA/D neighborhood update, MO-QNAS quantum update) needs the full fitness matrix of generation g before generation g+1 can be sampled. MOEA/D is nominally steady-state but this implementation evaluates children in batch (`moead.py` child evaluation loop), so the barrier holds for all algorithms here.

**Primary bottleneck.** Training time per architecture under GPU **time-sharing**, with GPU memory as the secondary constraint. Evidence from this repo's own logs: 4 candidates in parallel on one GPU ≈ 80 s/generation; 20 candidates time-sharing one device do not finish 5× slower than 4 — SM contention and memory pressure dominate (`[ASSUMPTION: threads: 20 was tuned empirically for the production GPU; no OOM-retry logic exists — a worker OOM surfaces as the RuntimeError catch at evaluation.py:199 and scores the candidate 0.0, silently biasing the search]`). Search-algorithm synchronization cost is negligible (CPU-side numpy on ≤50×3 arrays).

**Structural problems with the current parallelism.**
1. **Static pre-partitioning → stragglers.** Batches are fixed up-front (`individual_per_thread`); a worker that drew small/fast architectures idles while another grinds through deep ones. With heterogeneous NAS candidates this is the norm, not the exception.
2. The candidate-scores-0.0-on-OOM behavior couples parallelism level to *search correctness*: raising concurrency raises OOM probability which silently injects zero-fitness candidates.
3. One `GenericDataLoader` is built per worker (CIFAR-10 subset replicated per process) — memory overhead linear in `threads`.

### Proposed Strategy

Three changes, in order of value-per-risk; all preserve the generation barrier and therefore the algorithms' semantics exactly (the 0.6 determinism guarantee is precisely what makes them safe — results do not depend on which worker runs what, in which order):

1. **Work-stealing queue instead of static batches** (replace the round-robin partition with a shared task queue from which the N workers pull). Eliminates stragglers with ~30 lines of change confined to `EvalPopulation.__call__`/`run_individuals`. Determinism is unaffected by construction. This is the minimum change with the largest wall-time win.
2. **Explicit concurrency-vs-memory control + OOM honesty:** make `threads` per-GPU (`workers_per_gpu`), and on a worker OOM, retry the candidate once on an idle slot (after `torch.cuda.empty_cache()`) before scoring 0.0; log loudly either way. This decouples "more parallelism" from "more silent zeros".
3. **Population-level multi-GPU is already wired** (`thread_id % num_gpus`); after (1), verify it actually balances on a 2-GPU box and document `--gpu_list`/launcher slot interaction (Area 2: a run that declares 2 GPUs occupies 2 launcher slots).

**Evaluated and deliberately rejected:**
- *Async evaluation across the generation boundary* (job queue feeding the optimizer continuously): changes algorithm semantics (steady-state vs generational), invalidating every baseline and the Block-D parity work. Not justified by the bottleneck analysis.
- *Per-architecture DDP/data-parallel:* candidate nets are small (≤1.2 M params in logs) on 10 k samples; multi-GPU per candidate is overhead-dominated. Useful only for a future full-training retrain phase (`retrain_parallel.py` already covers that use case separately).
- *Ray/Dask style executors:* a dependency to do what `mp.Queue` + existing worker code already does at this scale.

**Cache interaction (Area 1).** A cache hit removes a task from the queue before workers see it, so effective parallelism is spent only on novel candidates — most valuable exactly when `workers_per_gpu` must be lowered for memory reasons (e.g., BF16/FP32 runs after Area 4, larger archs). The queue refactor should consult the cache wrapper *before* enqueueing, which falls out naturally if Area 1's `CachedEvaluator` wraps `EvalPopulation` (hits never reach `__call__`'s queue).

### Step-by-Step Plan

```
Step 1: Instrument the current scheduler
Files affected: core/evaluation.py (logging only)
Change scope: per-worker start/finish timestamps and per-candidate wall time
  in the generation summary log. No control-flow change.
Verify: a threads:4/pop:8 smoke run's log shows per-worker idle time —
  baseline straggler evidence to compare Step 2 against; run remains
  bit-exact vs expB_run1.

Step 2: Replace static batches with a task queue
Files affected: core/evaluation.py
Change scope: __call__ builds one task queue + N workers that pull until
  empty; run_individuals keeps its body (seeding, loader reseed, fitness
  call) per task. Result queue, fairness stage, and API are unchanged.
Verify: bit-exact per-candidate accuracy/params vs expB_run1 for
  threads ∈ {2,4} AND wall-time: rerun Step 1's instrumented case — max
  worker idle time drops (record numbers in the PR).

Step 3: workers_per_gpu semantics + OOM retry
Files affected: core/evaluation.py, core/config.py (key rename with
  back-compat: threads → workers_per_gpu)
Change scope: worker count = workers_per_gpu × len(visible GPUs); OOM →
  one retry then 0.0 with an ERROR-level log including candidate id.
Verify: forced-OOM test (tiny CUDA mem cap via env or an oversized dummy
  candidate) shows retry then explicit failure log; normal smoke
  bit-exact as always.

Step 4: Two-GPU balance validation
Files affected: none (experiment)
Change scope: run a pop-20 generation on gpu_list "0,1" with the queue
  scheduler.
Verify: per-GPU candidate counts within ±2; wall time < single-GPU run's
  by ≥1.6×; results bit-exact vs the single-GPU run (determinism across
  topology, the 0.6 property extended to multi-GPU).
```

### Risks and Open Questions

- **CUDA + fork:** workers are forked (`mp.Process` default on Linux) and currently initialize CUDA post-fork successfully; the queue refactor must not move any CUDA initialization before the fork. If a future torch version breaks this, the spawn context used by the fairness stage is the fallback (slower startup — loaders rebuilt).
- **Bit-exactness across GPU models is NOT claimed** — only across scheduling/topology on identical hardware. Step 4's cross-check assumes both GPUs are the same model (true for the current box; flag otherwise).
- The OOM-retry policy changes failure semantics from "always 0.0" to "retry once": strictly closer to the user's intent, but it must be documented since a flaky-OOM candidate could now succeed where the old code zeroed it.
- **Cross-area:** Area 1 (hits skip the queue), Area 2 (GPU slot accounting), Area 4 (precision changes memory footprint → workers_per_gpu retuning).

---

## Area 6: Experiment Checkpointing and Resumption

### Current State Analysis

**Where the state lives** (mapping the brief's notation to code):

| Brief | Code | Location |
|---|---|---|
| Qpop (PMF tensor, NQ×L×M) | `self.qpop_net.probabilities` (+ `qpop_net.current_pop`); continuous side: `self.qpop_params.{lower, upper, current_pop}` | built in `qnas2.QNAS.initialize_qnas` (:224-256 via `population.py::QPopulationNetwork/QPopulationParams`); inherited by `MOQNAS` |
| C̃g / Cg (classical pops) | `self.classical_nets`, `self.fits`, `self.raw_fits` (moqnas), `current_pop` mirrors in the qpop objects | `moqnas.evolve` loop (:658+) |
| Eg (elite set guiding the next quantum update) | derived inside `update_quantum` → `update_strategies.py`; **stateful part:** `self._q_ema` (`update_strategies.py:98-101`) — the EMA of elite distributions accumulated across ALL previous updates | `algorithms/qnas/helpers/update_strategies.py` |
| Ag (nondominated archive) | `self.pareto_global_population / _fitnesses / _params / _ids` + `self.fronts_history` | `moqnas.py` (:210-214), persisted per-gen to `pareto_history.pkl` |
| g | `self.current_gen` | qnas2/moqnas |
| Λ, λk (reference directions) | built at init from `ref_dir_method` / `elite_mode=moead_topk` config; `[ASSUMPTION: directions are deterministic given config (das-dennis) and are NOT mutated during the run — dirichlet method would be RNG-dependent and must be checkpointed if used]` | `population.py` / `update_strategies.py` |
| τu, τxo | `update_quantum_gen`, `crossover_frequency` — **pure functions of `current_gen`** (`current_gen % update_quantum_gen == 0` at qnas2:813; `current_gen % crossover_frequency != 0` at moqnas:477). Not independently stateful; restoring `current_gen` restores the cadence. | qnas2:813, moqnas:477 |

**Hidden stateful pieces beyond the brief's list** (these are what a naive implementation would miss):
- `self._q_ema` (above) — restoring Qpop without it makes the first post-resume update blend against a reset EMA. This is exactly the Eg-coupling failure mode described in the brief, materialized as one concrete variable.
- Update-schedule internals in `population.py`: `self._U_total` (:251/:296), the update counter `u` driving cosine schedules of `update_quantum_rate`/`max_update` (:496-519). `[ASSUMPTION: u is derived from current_gen // update_quantum_gen rather than independently incremented — must be confirmed; if independently incremented, it joins the checkpoint.]`
- `self.random = np.random.rand()` (qnas2:387) — the per-generation update intensity, drawn from **global numpy RNG**.
- Early-stopping memory: `best_so_far`, `last_best_so_far`, patience counters.

**The generation boundary / safe checkpoint location.** `MOQNAS.go_next_gen` (moqnas:592-655) runs, in order: `update_quantum(current_gen)` (:619, the PMF shift), archive/pareto bookkeeping, `save_data()` (:655), cleanup, then increments the counter. The **only safe checkpoint point** is the end of `go_next_gen`, after `save_data()` and after the increment decision is known — everything for g is final and g+1 has consumed no randomness yet. (For qnas2's own `evolve` the analogous point is its `go_next_gen` at qnas2:818+, which already calls `backup_cache` + `save_data`.)

**Existing serialization (checkpointing is NOT absent — it is incomplete).**
`qnas2.save_data` (:720-742) appends, keyed by generation, into `data_file` (`<exp>/...pkl`): fitnesses, raw_fitnesses, `qpop_params.lower/upper/current_pop`, **`qpop_net.probabilities`** (the PMFs!), `num_net_nodes`, `net_pop`. `load_qnas_data` (:753-778) restores all of that and `current_gen`. The entry-point side already has a resume phase: `--continue_path` → `phase='continue_evolution'` (`run_all_evolution._bootstrap`, `core/config.py::_get_continue_params`, `utils/experiment.py::check_files`). **What's missing:** (1) no RNG state of any source; (2) no `_q_ema`/schedule internals; (3) no archive Ag in `data_file` (it lives in the separate `pareto_history.pkl`, which stores fronts per gen — `pareto_global_params/population` are reconstructible from it only partially `[ASSUMPTION: fronts_history records ids+objective values but not chromosomes; chromosome recovery requires unique_networks.pkl or the archive itself must be checkpointed]`); (4) no config fingerprint/validation; (5) the per-generation append grows the file unboundedly; (6) no `--resume` discipline (continue_path semantics differ: it points at an old experiment dir and `[ASSUMPTION: the moqnas evolve path does not actually call load_qnas_data — the continue_evolution wiring appears complete only for config reloading; must be verified in Step 1]`).

**RNG sources in use** (all three families present, as the brief anticipates):
- **numpy global** — the search's backbone: observation sampling, crossover masks (`helpers/operators.py:25-134`), elite/archive sampling (moqnas:423-491), update intensity (qnas2:387), moead pruning. THE critical state.
- **Python `random`** — reseeded globally at startup (`set_global_seeds`) and per-candidate (`seed_candidate`); no search-side draws found, but capturing `random.getstate()` is one line — include it.
- **torch** — reseeded deterministically per candidate from `(global_seed, generation, candidate_id)` (stage 0.6), so its state is *derivable*, not accumulated; capture anyway for defense in depth (CPU + CUDA states).

**Are Cg's objective vectors retained?** Yes, twice: `save_data` stores `fitnesses`/`raw_fitnesses` per generation, and `unique_networks.pkl` (qnas2:477, moqnas:51) maps network → evaluation record. Therefore the evaluation cache (Area 1) **can** reconstruct elite objective vectors on resume without retraining, provided the cache fingerprint matches.

### Proposed Strategy

Extend the existing `save_data`/`load_qnas_data` mechanism into a complete, validated checkpoint — do not build a parallel system. One file `<experiment_path>/checkpoint.pkl`, written atomically (temp + `os.replace`) at the end of `go_next_gen`, **overwriting** by default; optional `checkpoint_keep_every: N` writes an additional immutable `checkpoint_gen{g}.pkl`. The legacy per-gen `data_file` append remains untouched for analysis tooling (it is a log, not a checkpoint).

**Checkpoint contents** (single dict, versioned with a `format_version` field):
1. `g = current_gen`, `total_eval`, early-stopping state (`best_so_far`, `last_best_so_far`, counters).
2. Qpop: `qpop_net.probabilities`, `qpop_net.current_pop`, `num_net_nodes`; `qpop_params.lower/upper/current_pop`.
3. Elite-update state: `_q_ema` and the schedule counter(s) identified in Step 1 (the Eg-coupling state).
4. Classical state: `classical_nets`, `fits`, `raw_fits` (Eg itself is recomputed from these + `_q_ema` by the next update; storing inputs beats storing a derived set).
5. Archive Ag: `pareto_global_population/_fitnesses/_params/_ids` (+ `fronts_history` reference is already on disk).
6. RNG: `np.random.get_state()`, `random.getstate()`, `torch.get_rng_state()` + `torch.cuda.get_rng_state_all()`.
7. Config block for mismatch detection: `NQ` (num_quantum_ind), `L` (max_num_nodes), `M` (len(fn_list)), `update_quantum_gen`, `crossover_frequency`, objective list, `precision` (Area 4), global seed, `ref_dir_method`, `elite_mode`, dataset fingerprint fields shared with Area 1.

**Resume protocol.** New `--resume` flag in `run_all_evolution.py` (explicitly distinct from the legacy `--continue_path`): without it, an existing checkpoint is ignored and the run starts at g=0 (safe-by-default, per the brief's direct-execution rationale — only a log line announces the ignored checkpoint). With it: load config block → compare field-by-field against the current run's values → on any mismatch abort listing exactly the differing fields → only then load state, restore RNG **last** (after any construction code that might consume randomness), and enter the loop at g+1. Reference directions Λ are rebuilt from config and compared against a stored hash to confirm consistency (covers the dirichlet `[ASSUMPTION]`).

**Cache interaction (Area 1).** Order: checkpoint state loads **first** (it is authoritative for search state); the cache is consulted only *after*, and only for one job — if any architecture in the restored elite/classical sets lacks objective vectors (e.g., a partially-written legacy state), its vectors are taken from the cache instead of retraining. A missing cache entry for such an architecture is a hard error under `--resume --strict` (default) or triggers a re-evaluation of exactly that architecture under `--resume --reeval-missing`; silent zero-filling is never allowed. With the checkpoint format above this path should never fire (fits are stored), so the cache remains an optimization, not a dependency — exactly the "complementary but distinct" relationship the brief requires.

**Launcher interaction (Area 2).** `resume: true|false` and `checkpoint_path` (default `<experiment_path>/checkpoint.pkl`) become per-cell matrix keys; relaunching after preemption is rerunning the same launcher command with `resume: true` — no argument archaeology.

**Verification gold standard** (enabled by Area 3's observation): with a FLOPs-based objective set (no measured-time objective), an uninterrupted N-generation run is bit-reproducible; therefore *interrupt-at-g + resume → must equal the uninterrupted run bit-for-bit* (populations, PMFs, archive, fronts). This is a far stronger acceptance test than "it runs" and is cheap to automate.

### Step-by-Step Plan

```
Step 1: State audit — confirm the three [ASSUMPTION]s
Files affected: none (read-only)
Change scope: trace (a) whether population.py's update counter u is derived
  from current_gen or self-incremented; (b) whether moqnas's evolve ever calls
  load_qnas_data under continue_evolution; (c) what fronts_history stores
  (chromosomes or only ids/objectives) and whether ref dirs can be RNG-built
  (dirichlet path).
Verify: a written state inventory table (variable → owner class → derived?
  → in checkpoint?) reviewed against this section; each assumption marked
  confirmed/refuted with file:line evidence.

Step 2: Implement checkpoint write at the generation boundary
Files affected: algorithms/qnas/qnas2.py (go_next_gen), algorithms/qnas/
  moqnas.py (go_next_gen), small new helper algorithms/qnas/checkpoint.py
Change scope: serialize the contents list above; atomic write; overwrite
  default + keep-every-N option read from config. No load path yet; no
  behavior change for runs (write-only addition).
Verify: a 3-generation moqnas smoke produces checkpoint.pkl whose keys match
  the spec; re-running the same smoke yields a byte-identical checkpoint
  (FLOPs-objective config, fixed seed) — proving the serialized state is
  itself deterministic.

Step 3: Implement --resume with config validation
Files affected: run_all_evolution.py, algorithms/qnas/checkpoint.py,
  core/config.py (expose the config block fields)
Change scope: flag parsing, mismatch check (abort listing differing fields),
  state restore order (config → objects → RNG last), loop entry at g+1.
  Default path (no flag) provably untouched.
Verify: (a) mismatch test: checkpoint from a 3-objective run + resume with a
  2-objective config aborts naming `objectives`; (b) no-flag test: with a
  checkpoint present, a run without --resume starts at generation 0 and logs
  the ignored checkpoint.

Step 4: Bit-equivalence acceptance test
Files affected: .refactor_baseline/ (test script, in the spirit of expB.sh)
Change scope: script runs (A) uninterrupted 6-gen moqnas with FLOPs
  objectives, (B) same config killed after gen 3 then resumed to gen 6;
  compares per-candidate metrics, final PMFs, archive contents.
Verify: A == B bit-exact (numpy array_equal on PMFs and fitness matrices;
  identical pareto_history). This single check certifies Qpop, Eg-coupling
  (_q_ema), RNG capture, and cadence restoration simultaneously — any
  missed state makes generations 4-6 diverge.

Step 5: Cache + launcher wiring
Files affected: core/eval_cache.py (Area 1), launch.py + matrices (Area 2),
  run_all_evolution.py (strict/reeval-missing flags)
Change scope: the missing-vector policy described above; resume/checkpoint
  keys in the matrix schema.
Verify: delete one elite's cache entry from a checkpointed experiment →
  --resume (strict) aborts naming the architecture; --resume
  --reeval-missing retrains exactly one candidate (log shows 1 evaluation)
  and Step 4's equivalence still holds afterward.
```

### Risks and Open Questions

- **The Step-1 assumptions are load-bearing.** If `u` is self-incremented or moqnas already half-loads legacy state on continue, the checkpoint contents list changes; the plan deliberately front-loads that audit.
- **Pickle fragility across versions:** numpy/torch RNG state pickles are stable in practice but the checkpoint carries `format_version` + library versions so a mismatch fails loudly. Long-term storage of checkpoints is not a goal (they are overwritten); only the keep-every-N artifacts persist.
- **Time-based objectives break the gold-standard test, not the feature:** with `cuda_inference_time` in the objective set, resumed runs legitimately diverge from uninterrupted ones after the first post-resume selection. The acceptance test pins this down by using FLOPs objectives; production users with time objectives get correct-but-not-bit-identical resumption, and the docs must say so explicitly.
- **Cross-area:** Area 1 (vector reconstruction policy, shared fingerprint), Area 2 (resume as a config key), Area 3 (FLOPs configs enable the bit-equivalence test), Area 4 (precision is a mandatory config-block field — resuming an fp16 checkpoint under bf16 must abort).

---

## Implementation Order Recommendation

Dependencies imply a natural sequence: **Area 3 → Area 4 → Area 1 → Area 6 → Area 5 → Area 2** is *not* required in full — areas are independently implementable as written — but two orderings pay for themselves: (1) Area 3's FLOPs-objective example config should land before Area 6's Step 4, because it turns the resume acceptance test into a bit-exact comparison; (2) Area 1's fingerprint should be defined (even if the cache ships later) before Area 4 and Area 6 freeze their config-block schemas, so the three features share one notion of "evaluation identity": `(dataset, limit, epochs, optimizer, batch size, precision, objective set, seed policy)`.
