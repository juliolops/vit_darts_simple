#!/usr/bin/env python3
"""Pareto-aware retrain launcher for multi-objective MoQ-NAS experiments.

Reads a YAML retrain matrix, applies configurable Pareto selection rules
(weighted scalarization or explicit IDs) to each (experiment × repeat) cell,
and dispatches retrain_parallel.py for each cell.

Scope: multi-objective algorithms only (MOQNAS, NSGA-II, NSGA-III, MOEA/D).
Single-objective algorithms continue using the existing .sh workflow.

Usage:
    python launch_retrain.py retrain_matrices/cifar_mo.yaml
    python launch_retrain.py retrain_matrices/cifar_mo.yaml --dry-run
"""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
import warnings

import yaml

RETRAIN_SCRIPT = 'retrain_parallel.py'
CFG_OBJ_PATH = 'dataset_configs/cfg_obj.json'


# ---------------------------------------------------------------------------
# Pareto front loading
# ---------------------------------------------------------------------------

def load_pareto_front(repeat_dir: str) -> list[dict]:
    """Load final Pareto front records from pareto_history.pkl in repeat_dir.

    Returns a list of dicts, each with 'id' (str) and one float per objective.
    Filters to candidates that exist in archive/.
    """
    path = os.path.join(repeat_dir, 'pareto_history.pkl')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"pareto_history.pkl not found: {path}")
    with open(path, 'rb') as f:
        history = pickle.load(f)
    if not history:
        raise ValueError(f"pareto_history.pkl is empty: {path}")
    last_gen = max(history.keys())
    records = history[last_gen].get(1, [])
    if not records:
        raise ValueError(
            f"No rank-1 Pareto front at generation {last_gen} in {repeat_dir}"
        )
    # Normalize numpy scalars to Python native types
    front = []
    for rec in records:
        entry = {}
        for k, v in rec.items():
            entry[k] = str(v) if k == 'id' else float(v)
        front.append(entry)

    # Filter to candidates that exist on disk
    archive_dir = os.path.join(repeat_dir, 'archive')
    if os.path.isdir(archive_dir):
        existing = set(os.listdir(archive_dir))
        front = [c for c in front if c['id'] in existing]
    if not front:
        raise ValueError(
            f"No Pareto front candidates found in archive: {archive_dir}"
        )
    return front


# ---------------------------------------------------------------------------
# Objective direction resolution
# ---------------------------------------------------------------------------

def _load_cfg_obj_fragments(cfg_obj_path: str) -> dict[str, str]:
    """Load {fragment: 'max'|'min'} from cfg_obj.json.

    Fragment matching is by substring: 'flops' matches 'total_flops'.
    """
    if not os.path.isfile(cfg_obj_path):
        raise FileNotFoundError(
            f"Objective direction file not found: {cfg_obj_path}\n"
            "Cannot determine maximize/minimize for selection weights."
        )
    with open(cfg_obj_path) as f:
        data = json.load(f)
    return {
        k: ('max' if v['goal'] == 'maximize' else 'min')
        for k, v in data['objectives'].items()
    }


def resolve_directions(objective_names: list[str], cfg_obj_path: str) -> dict[str, str]:
    """Map each objective name to 'max' or 'min' via substring matching."""
    fragments = _load_cfg_obj_fragments(cfg_obj_path)
    directions = {}
    for name in objective_names:
        matched = next(
            (direction for fragment, direction in fragments.items() if fragment in name),
            None
        )
        if matched is None:
            warnings.warn(
                f"Objective '{name}' not matched in {cfg_obj_path}. Defaulting to 'max'.",
                stacklevel=3,
            )
            matched = 'max'
        directions[name] = matched
    return directions


# ---------------------------------------------------------------------------
# Pure selection functions (no I/O, no subprocess — independently testable)
# ---------------------------------------------------------------------------

def select_by_weighted_score(
    front: list[dict],
    weights: dict[str, float],
    directions: dict[str, str],
    n: int,
) -> list[int]:
    """Select top-n candidates using weighted scalarization over the Pareto front.

    Score for candidate i:
        score_i = Σ_j  w_j × contribution_j(val_ij)
    where:
        contribution_j = normalize_j(val)         for maximize objectives
        contribution_j = 1 - normalize_j(val)     for minimize objectives
        normalize_j(val) = (val - min_j) / (max_j - min_j)

    Args:
        front: List of candidate dicts with 'id' and per-objective float values.
        weights: {objective_name: weight}. Auto-normalized to sum=1.0 if needed.
        directions: {objective_name: 'max'|'min'} for every objective in weights.
        n: Number of candidates to return.

    Returns:
        Indices into front sorted by score descending (stable: ties → lower index).

    Edge cases:
        - len(front) == 1: returns [0] regardless of weights.
        - Objective in weights missing from front: raises ValueError.
        - Weights not summing to 1: normalized + warning.
        - Zero-range objective (all equal): contribution = 0.5 + warning.
        - n > len(front): all indices returned + warning.
    """
    if len(front) == 1:
        return [0]

    # Validate objectives present in front
    available = [k for k in front[0] if k != 'id']
    for obj in weights:
        if obj not in front[0]:
            raise ValueError(
                f"Objective '{obj}' from weights not found in Pareto front. "
                f"Available: {available}"
            )
        if obj not in directions:
            raise ValueError(
                f"No direction ('max'/'min') provided for objective '{obj}'."
            )

    # Normalize weights
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        warnings.warn(
            f"Weights sum to {total:.4f} ≠ 1.0. Normalizing automatically. "
            f"Original weights: {weights}",
            stacklevel=3,
        )
        weights = {k: v / total for k, v in weights.items()}

    # Per-objective range
    obj_min = {obj: min(c[obj] for c in front) for obj in weights}
    obj_max = {obj: max(c[obj] for c in front) for obj in weights}

    # Score each candidate
    scores = []
    for i, cand in enumerate(front):
        score = 0.0
        for obj, w in weights.items():
            lo, hi = obj_min[obj], obj_max[obj]
            if math.isclose(lo, hi, rel_tol=1e-9, abs_tol=1e-9):
                warnings.warn(
                    f"Objective '{obj}' has zero range on front "
                    f"(all values = {lo}). Setting contribution = 0.5.",
                    stacklevel=3,
                )
                contrib = 0.5
            else:
                norm = (cand[obj] - lo) / (hi - lo)
                contrib = norm if directions[obj] == 'max' else (1.0 - norm)
            score += w * contrib
        scores.append((score, i))

    # Stable descending sort; ties broken by original index (lowest wins)
    scores.sort(key=lambda x: (-x[0], x[1]))
    indices = [i for _, i in scores]

    if n > len(front):
        warnings.warn(
            f"Requested n={n} > front size {len(front)}. Returning all.",
            stacklevel=3,
        )
        return indices
    return indices[:n]


def select_by_ids(front: list[dict], ids: list[str]) -> list[int]:
    """Select candidates by explicit IDs, preserving the order of `ids`.

    Args:
        front: List of candidate dicts with 'id' field.
        ids: Ordered list of candidate IDs to select.

    Returns:
        Indices into front in the same order as ids.

    Raises:
        ValueError: If any requested ID is absent from the front.
    """
    id_to_idx = {c['id']: i for i, c in enumerate(front)}
    result = []
    for cid in ids:
        if cid not in id_to_idx:
            raise ValueError(
                f"ID '{cid}' not found in Pareto front. "
                f"Available: {list(id_to_idx)}"
            )
        result.append(id_to_idx[cid])
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def pick_candidates(repeat_dir: str, rule: dict, cfg_obj_path: str) -> list[str]:
    """Apply a selection rule to the Pareto front and return selected IDs."""
    front = load_pareto_front(repeat_dir)

    min_acc = rule.get('min_accuracy')
    if min_acc is not None:
        filtered = [c for c in front if c.get('best_accuracy', 0.0) >= float(min_acc)]
        if not filtered:
            raise ValueError(
                f"No candidates with best_accuracy >= {min_acc} in {repeat_dir}. "
                f"Front range: {min(c.get('best_accuracy',0) for c in front):.1f}–"
                f"{max(c.get('best_accuracy',0) for c in front):.1f}%"
            )
        front = filtered

    rule_type = rule.get('type')

    if rule_type == 'weighted':
        weights = {str(k): float(v) for k, v in rule.get('weights', {}).items()}
        n = int(rule.get('n', 1))
        if not weights:
            raise ValueError("Selection 'weighted' requires a non-empty 'weights' dict.")
        directions = resolve_directions(list(weights.keys()), cfg_obj_path)
        indices = select_by_weighted_score(front, weights, directions, n)
        return [front[i]['id'] for i in indices]

    elif rule_type == 'ids':
        ids = [str(x) for x in rule.get('ids', [])]
        if not ids:
            raise ValueError("Selection 'ids' requires a non-empty 'ids' list.")
        indices = select_by_ids(front, ids)
        return [front[i]['id'] for i in indices]

    else:
        raise ValueError(
            f"Unknown selection type: '{rule_type}'. Supported: 'weighted', 'ids'."
        )


def _build_argv(repeat_dir: str, candidate_ids: list[str], merged: dict) -> list[str]:
    """Build retrain_parallel.py CLI argv for one cell."""
    argv = [sys.executable, RETRAIN_SCRIPT,
            '--experiment_path', repeat_dir,
            '--ids', *candidate_ids]

    # Keyword arguments passed through to retrain_parallel.py
    kv_args = [
        'data_path', 'dataset', 'max_epochs', 'epochs_to_eval', 'num_repetitions',
        'lr_scheduler', 'optimizer', 'batch_size', 'eval_batch_size', 'log_level',
        'num_workers', 'save_checkpoints_epochs', 'patience_retrain', 'delta_fraction',
        'max_parallel_workers', 'network_config',
    ]
    for key in kv_args:
        if key in merged:
            argv += [f'--{key}', str(merged[key])]

    # Boolean flags
    for flag in ('data_augmentation', 'limit_data', 'keep_metrics'):
        if merged.get(flag):
            argv.append(f'--{flag}')

    return argv


def expand(matrix: dict):
    """Yield one cell dict per (experiment, repeat) pair."""
    defaults = matrix.get('defaults', {}) or {}
    exp_root = matrix['exp_root']
    cfg_obj_path = matrix.get('cfg_obj_path', CFG_OBJ_PATH)
    global_rule = matrix.get('selection')

    for exp in matrix['experiments']:
        algo = exp['algo']
        name = exp['name']
        repeats = int(exp.get('repeats', matrix.get('repeats', 1)))
        rule = exp.get('selection') or global_rule
        overrides = exp.get('overrides', {}) or {}
        merged = {**defaults, **overrides}

        if rule is None:
            raise ValueError(
                f"No 'selection' block for experiment '{name}' (algo={algo}). "
                "Add a per-experiment or top-level 'selection' key."
            )

        for i in range(1, repeats + 1):
            repeat_dir = os.path.join(exp_root, algo, f'{name}_repeat_{i}')
            yield {
                'algo': algo,
                'name': name,
                'repeat': i,
                'repeat_dir': repeat_dir,
                'rule': rule,
                'merged': merged,
                'cfg_obj_path': cfg_obj_path,
            }


def _build_slots(matrix: dict) -> list[list]:
    """Group GPU pool into slots of gpus_per_run GPUs each."""
    gpus = matrix.get('gpus', [0]) or [0]
    gpr = max(1, int(matrix.get('gpus_per_run', 1)))
    return [gpus[i:i + gpr] for i in range(0, len(gpus), gpr)]


def _write_selection_log(log_path: str, cell: dict) -> None:
    """Append selection audit entry to the launch log."""
    with open(log_path, 'a') as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launch_retrain.py\n")
        f.write(f"  algo={cell['algo']}  name={cell['name']}  repeat={cell['repeat']}\n")
        f.write(f"  rule={cell['rule']}\n")
        f.write(f"  selected_ids={cell['candidate_ids']}\n")
        f.write(f"  cmd={' '.join(cell['argv'])}\n")


def run_matrix(matrix: dict, dry_run: bool = False) -> int:
    slots = _build_slots(matrix)
    cfg_obj_path = matrix.get('cfg_obj_path', CFG_OBJ_PATH)

    # Resolve candidates for every cell before launching anything
    resolved = []
    for cell in expand(matrix):
        try:
            candidate_ids = pick_candidates(cell['repeat_dir'], cell['rule'], cfg_obj_path)
        except Exception as exc:
            print(
                f"[ERROR] {cell['algo']}/{cell['name']}_repeat_{cell['repeat']}: {exc}",
                file=sys.stderr,
            )
            continue

        argv = _build_argv(cell['repeat_dir'], candidate_ids, cell['merged'])
        resolved.append({**cell, 'candidate_ids': candidate_ids, 'argv': argv})

    if not resolved:
        print("[ERROR] No cells could be resolved. Aborting.", file=sys.stderr)
        return 1

    if dry_run:
        for cell in resolved:
            visible = ','.join(str(g) for g in slots[0])
            label = f"{cell['algo']}/{cell['name']}_repeat_{cell['repeat']}"
            print(f"# {label}  selected={cell['candidate_ids']}  rule={cell['rule']}")
            print(f"CUDA_VISIBLE_DEVICES={visible} " + ' '.join(cell['argv']))
            print()
        print(
            f"# {len(resolved)} cell(s), {len(slots)} slot(s) "
            f"of {len(slots[0])} GPU(s) each.",
            file=sys.stderr,
        )
        return 0

    # Schedule cells over GPU slots (same pattern as launch.py)
    pending = list(resolved)
    running = {}
    results = []
    t0 = time.perf_counter()

    while pending or running:
        free = [i for i in range(len(slots)) if i not in running]
        while free and pending:
            si = free.pop(0)
            cell = pending.pop(0)
            visible = ','.join(str(g) for g in slots[si])
            cell['gpu'] = visible

            os.makedirs(cell['repeat_dir'], exist_ok=True)
            log_path = os.path.join(cell['repeat_dir'], 'retrain_launcher.log')
            _write_selection_log(log_path, cell)

            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible)
            log_file = open(log_path, 'a')
            proc = subprocess.Popen(
                cell['argv'], env=env, stdout=log_file, stderr=subprocess.STDOUT
            )
            cell['_log'] = log_file
            running[si] = (cell, proc)
            print(
                f"[launch] GPU{visible} "
                f"{cell['algo']}/{cell['name']}_repeat_{cell['repeat']} "
                f"ids={cell['candidate_ids']}"
            )

        time.sleep(1.0)
        for si, (cell, proc) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                cell['_log'].close()
                results.append((cell['repeat_dir'], rc))
                status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
                print(
                    f"[done]   GPU{cell['gpu']} {status}: "
                    f"{cell['algo']}/{cell['name']}_repeat_{cell['repeat']}"
                )
                del running[si]

    wall = time.perf_counter() - t0
    failed = [p for p, rc in results if rc != 0]
    print(
        f"\n=== {len(results)} run(s) in {wall:.0f}s; "
        f"{len(results) - len(failed)} ok, {len(failed)} failed ==="
    )
    for p in failed:
        print(f"  FAILED: {p} (see {os.path.join(p, 'retrain_launcher.log')})")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(
        description='MoQ-NAS Pareto-aware retrain launcher (multi-objective only).'
    )
    ap.add_argument('matrix', help='Path to a retrain matrix YAML file.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print expanded commands without running anything.')
    ap.add_argument('--cfg-obj', default=None,
                    help=f'Override path to cfg_obj.json (default: {CFG_OBJ_PATH}).')
    args = ap.parse_args()

    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)
    if args.cfg_obj:
        matrix['cfg_obj_path'] = args.cfg_obj

    sys.exit(run_matrix(matrix, dry_run=args.dry_run))


if __name__ == '__main__':
    main()
