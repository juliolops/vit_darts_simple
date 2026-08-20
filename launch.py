#!/usr/bin/env python3
"""Experiment launcher for MoQ-NAS evolution runs (Area 2).

Replaces the hand-edited per-experiment ``run_*.sh`` scripts with one
launcher driven by an experiment-matrix YAML. The launcher expands the
matrix into one ``run_all_evolution.py`` invocation per
(experiment x repeat) cell, schedules the cells over a declared GPU-slot
pool (one process per slot at a time), gives every repeat an explicit
seed (``seed_base + repeat_index`` — the scripts never passed --seed, so
their "repeats" differed only by timing noise), records the fully
expanded command into each experiment directory, and supports
``--dry-run``.

Matrix format (see experiment_matrices/*.yaml):

    defaults:            # args common to every cell -> --key value / --key (bool true)
      dataset: cifar10
      config_path_dataset: dataset_configs/cifar10_vit.yaml
      limit_data_value: 10000
      log_level: INFO
    gpus: [0]            # GPU pool
    gpus_per_run: 1      # GPUs per run; pool is grouped into slots of this size
    repeats: 3
    seed_base: 42        # repeat i (1-based) runs with seed_base + i
    resume: false        # OR pass --resume on the launcher to resume the batch
    exp_root: experiment_cifar10_qfamily
    experiments:
      - algo: moqnas
        config: experiment_configs/vit/config_vit_heads.yaml
        name: exp10
        overrides: {optimizer: AdamW}     # per-cell args (override defaults)
        flags: [--multi_objective]        # literal flags appended verbatim

Each cell's experiment_path is ``<exp_root>/<algo>/<name>_repeat_<i>``,
matching the directory layout the shell scripts produced so existing
analysis tooling keeps working.
"""
import argparse
import os
import subprocess
import sys
import time

import yaml

RUN_SCRIPT = 'run_all_evolution.py'


def _arg_tokens(key, value):
    """Translate one matrix key/value into run_all_evolution CLI tokens."""
    if isinstance(value, bool):
        return [f'--{key}'] if value else []          # store_true flag
    if isinstance(value, (list, tuple)):
        return [f'--{key}', *[str(v) for v in value]]
    return [f'--{key}', str(value)]


def expand(matrix: dict, resume: bool = False):
    """Yield one cell dict per (experiment, repeat).

    Each cell: {experiment_path, seed, gpu(None yet), argv(list)}.

    ``resume`` (matrix key ``resume: true`` OR the launcher ``--resume``
    flag) appends ``--resume`` to cells that already have a
    ``<experiment_path>/checkpoint.pkl``, so relaunching an interrupted
    batch is rerunning the same matrix: each interrupted run picks up
    from its own checkpoint while cells that never started launch fresh
    (run_all_evolution.py raises FileNotFoundError if --resume is passed
    without an existing checkpoint).
    """
    defaults = matrix.get('defaults', {}) or {}
    repeats = int(matrix.get('repeats', 1))
    seed_base = int(matrix.get('seed_base', 42))
    exp_root = matrix['exp_root']
    resume = bool(resume or matrix.get('resume', False))

    for exp in matrix['experiments']:
        algo = exp['algo']
        config = exp['config']
        name = exp['name']
        overrides = exp.get('overrides', {}) or {}
        flags = list(exp.get('flags', []) or [])
        for i in range(1, repeats + 1):
            exp_path = os.path.join(exp_root, algo, f'{name}_repeat_{i}')
            seed = seed_base + i
            merged = {**defaults, **overrides}
            argv = [sys.executable, RUN_SCRIPT,
                    '--algo', algo,
                    '--config_file', config,
                    '--experiment_path', exp_path,
                    '--seed', str(seed)]
            for k, v in merged.items():
                argv += _arg_tokens(k, v)
            argv += flags
            if resume and _checkpoint_gen(exp_path) is not None:
                argv.append('--resume')
            yield {'experiment_path': exp_path, 'seed': seed, 'gpu': None, 'argv': argv}


def _record_command(cell):
    os.makedirs(cell['experiment_path'], exist_ok=True)
    with open(os.path.join(cell['experiment_path'], 'launch_command.txt'), 'w') as f:
        env = f"CUDA_VISIBLE_DEVICES={cell['gpu']} " if cell['gpu'] is not None else ''
        f.write(env + ' '.join(cell['argv']) + '\n')


def _checkpoint_gen(experiment_path):
    """Completed generation in a cell's checkpoint, or None if absent."""
    path = os.path.join(experiment_path, 'checkpoint.pkl')
    if not os.path.exists(path):
        return None
    try:
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f).get('completed_gen')
    except Exception:
        return None


def _build_slots(matrix):
    """Group the GPU pool into slots of ``gpus_per_run`` GPUs each.

    With ``gpus: [0,1,2,3]`` and ``gpus_per_run: 2`` the slots are
    ``[[0,1],[2,3]]``: two runs execute concurrently, each one seeing two
    GPUs (CUDA_VISIBLE_DEVICES="0,1"/"2,3") over which its candidates are
    balanced. Default ``gpus_per_run: 1`` keeps one GPU per run.
    """
    gpus = matrix.get('gpus', [0]) or [0]
    gpr = max(1, int(matrix.get('gpus_per_run', 1)))
    return [gpus[i:i + gpr] for i in range(0, len(gpus), gpr)]


def run_matrix(matrix: dict, dry_run: bool = False, resume: bool = False):
    slots = _build_slots(matrix)
    resume = bool(resume or matrix.get('resume', False))
    cells = list(expand(matrix, resume=resume))

    if dry_run:
        for c in cells:
            print(' '.join(c['argv']))
        print(f"\n# {len(cells)} cell(s), {len(slots)} slot(s) "
              f"of {len(slots[0])} GPU(s) each.", file=sys.stderr)
        return 0

    if resume:
        # Pre-flight: show which cells already have progress. Finished runs
        # load their final checkpoint and exit almost immediately; interrupted
        # runs continue from their last generation.
        print("[resume] checkpoint status per cell:")
        for c in cells:
            g = _checkpoint_gen(c['experiment_path'])
            print(f"  {'gen ' + str(g) if g is not None else 'no checkpoint (fresh)':>22}"
                  f"  {c['experiment_path']}")

    pending = list(cells)
    running = {}   # slot_index -> (cell, Popen)
    results = []   # (experiment_path, returncode)
    t0 = time.perf_counter()

    while pending or running:
        # Fill every free slot.
        free = [i for i in range(len(slots)) if i not in running]
        while free and pending:
            si = free.pop(0)
            cell = pending.pop(0)
            visible = ','.join(str(g) for g in slots[si])
            cell['gpu'] = visible
            _record_command(cell)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible)
            log = open(os.path.join(cell['experiment_path'], 'launcher.log'), 'w')
            proc = subprocess.Popen(cell['argv'], env=env, stdout=log, stderr=subprocess.STDOUT)
            cell['_log'] = log
            running[si] = (cell, proc)
            print(f"[launch] GPU{visible} seed={cell['seed']} -> {cell['experiment_path']}")

        # Reap any finished slot; one failure never kills siblings.
        time.sleep(1.0)
        for si, (cell, proc) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                cell['_log'].close()
                results.append((cell['experiment_path'], rc))
                status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
                print(f"[done]   GPU{cell['gpu']} {status}: {cell['experiment_path']}")
                del running[si]

    wall = time.perf_counter() - t0
    failed = [p for p, rc in results if rc != 0]
    print(f"\n=== {len(results)} run(s) in {wall:.0f}s; "
          f"{len(results) - len(failed)} ok, {len(failed)} failed ===")
    for p in failed:
        print(f"  FAILED: {p} (see {os.path.join(p, 'launcher.log')})")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description='MoQ-NAS experiment-matrix launcher.')
    ap.add_argument('matrix', help='Path to an experiment-matrix YAML file.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print the expanded commands without running anything.')
    ap.add_argument('--resume', action='store_true',
                    help='Append --resume to every run (each picks up from its own '
                         'checkpoint.pkl). Overrides the matrix `resume` key. Use this '
                         'to relaunch an interrupted batch with the same matrix.')
    args = ap.parse_args()
    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)
    sys.exit(run_matrix(matrix, dry_run=args.dry_run, resume=args.resume))


if __name__ == '__main__':
    main()
