"""Engine-agnostic checkpoint/resume for the evolutionary search engines.

One checkpoint file per experiment (``<experiment_path>/checkpoint.pkl``),
written atomically at every generation boundary — the end of
``go_next_gen``, after the archive/update and ``save_data`` are complete
and before the generation counter advances — and overwritten by default
(``checkpoint_keep_every: N`` in the train config additionally keeps an
immutable ``checkpoint_gen{g}.pkl`` every N generations).

The module captures only what is **common** to every engine:

- the generation counter and shared bookkeeping (``total_eval``,
  best-so-far, early-stopping counter);
- ALL RNG states — numpy global (the search backbone), Python ``random``
  and torch CPU/CUDA. torch states are stored as numpy arrays because a
  raw ``torch.Tensor`` pickles with non-deterministic metadata (identical
  content, different bytes), which would make otherwise-equal checkpoints
  byte-differ;
- a config block validated field-by-field on resume; any mismatch aborts
  naming the differing fields.

Everything **engine-specific** is delegated to three methods that each
engine implements (see ``algorithms/qnas`` for the quantum populations and
``algorithms/ga`` for the GA-family populations/archives):

- ``_checkpoint_config_block()`` -> dict of identity-defining config;
- ``_checkpoint_state()`` -> dict of the engine's resumable state;
- ``_restore_state(state)`` -> restore it.

Resume requires the explicit ``--resume`` flag; without it an existing
checkpoint is ignored and the run starts from generation 0.
"""
import os
import pickle
import random
import shutil
import tempfile
import time

import numpy as np
import torch

FORMAT_VERSION = 2


def checkpoint_path(engine) -> str:
    return os.path.join(engine.experiment_path, 'checkpoint.pkl')


def _capture_rng() -> dict:
    return {
        'numpy': np.random.get_state(),
        'python': random.getstate(),
        'torch_cpu': torch.get_rng_state().numpy(),
        'torch_cuda': ([s.numpy() for s in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_initialized() else None),
    }


def _restore_rng(rng: dict) -> None:
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    torch.set_rng_state(torch.tensor(np.asarray(rng['torch_cpu']), dtype=torch.uint8))
    if rng['torch_cuda'] is not None and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(
            [torch.tensor(np.asarray(s), dtype=torch.uint8) for s in rng['torch_cuda']])


def save_checkpoint(engine) -> None:
    """Serialize the full search state at the current generation boundary."""
    elapsed = (
        getattr(engine, '_elapsed_so_far', 0.0)
        + (time.time() - getattr(engine, '_session_start', time.time()))
    )
    state = {
        'format_version': FORMAT_VERSION,
        'completed_gen': int(engine.current_gen),
        'elapsed_seconds': elapsed,
        'common': {
            'total_eval': getattr(engine, 'total_eval', 0),
            'best_so_far': getattr(engine, 'best_so_far', None),
            'last_best_so_far': getattr(engine, 'last_best_so_far', None),
            'best_so_far_id': getattr(engine, 'best_so_far_id', None),
            'early_stopping_counter': getattr(engine, 'early_stopping_counter', 0),
        },
        'rng': _capture_rng(),
        'config': engine._checkpoint_config_block(),
        'engine_state': engine._checkpoint_state(),
    }

    path = checkpoint_path(engine)
    parent = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(prefix='.checkpoint.', suffix='.tmp', dir=parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    keep_every = int(getattr(engine, 'checkpoint_keep_every', 0) or 0)
    if keep_every and engine.current_gen % keep_every == 0:
        shutil.copy2(path, os.path.join(
            engine.experiment_path, f'checkpoint_gen{engine.current_gen}.pkl'))


def load_checkpoint(engine) -> int:
    """Validate and restore a checkpoint into ``engine``.

    Returns the completed generation g; the caller resumes at g+1.

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists for the experiment.
    RuntimeError
        If the format version is unknown, or the checkpoint's config block
        does not match the current run (the message lists every differing
        field).
    """
    path = checkpoint_path(engine)
    if not os.path.exists(path):
        raise FileNotFoundError(f"--resume requested but no checkpoint at {path}")
    with open(path, 'rb') as f:
        state = pickle.load(f)

    if state.get('format_version') != FORMAT_VERSION:
        raise RuntimeError(
            f"Checkpoint format {state.get('format_version')} != supported {FORMAT_VERSION}")

    current = engine._checkpoint_config_block()
    saved = state['config']
    mismatches = [(k, saved.get(k), current.get(k))
                  for k in sorted(set(saved) | set(current))
                  if saved.get(k) != current.get(k)]
    if mismatches:
        detail = "; ".join(f"{k}: checkpoint={s!r} vs run={c!r}" for k, s, c in mismatches)
        raise RuntimeError(
            f"Checkpoint/run configuration mismatch — refusing to resume. {detail}")

    c = state['common']
    engine.total_eval = c['total_eval']
    engine.best_so_far = c['best_so_far']
    engine.last_best_so_far = c['last_best_so_far']
    engine.best_so_far_id = c['best_so_far_id']
    engine.early_stopping_counter = c['early_stopping_counter']
    engine.current_gen = state['completed_gen']
    engine._elapsed_so_far = state.get('elapsed_seconds', 0.0)

    engine._restore_state(state['engine_state'])

    # RNG restored LAST so nothing above consumes randomness afterwards.
    _restore_rng(state['rng'])

    engine._resumed = True
    return state['completed_gen']
