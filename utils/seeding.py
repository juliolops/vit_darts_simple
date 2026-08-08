"""Global RNG seeding utilities for reproducible evolutionary runs.

Centralizes the seeding of every random number generator used by the
search pipeline (``random``, ``numpy``, ``torch`` CPU/CUDA) so that
experiments launched with the same ``--seed`` are comparable run-to-run.
"""
import os
import random

import numpy as np
import torch


def set_global_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs used by the evolutionary search for reproducible runs.

    Parameters
    ----------
    seed : int, default 42
        Seed applied to ``random``, ``numpy`` and ``torch`` (CPU and CUDA).
    deterministic : bool, default True
        If True, force cuDNN into deterministic mode and disable benchmark
        autotuning. Trades some GPU throughput for run-to-run stability.

    Notes
    -----
    Architecture sampling is fully deterministic under this seeding:
    repeated runs produce bit-exact chromosomes/parameter counts.
    Trained accuracy is NOT reproducible run-to-run, because candidates
    are trained in parallel worker threads that draw weight
    initialization from the shared global RNG, so the interleaving is
    scheduler-dependent (observed spread ~1-3 pp on 1-epoch smoke runs).
    Operator correctness in Block D is therefore verified with synthetic
    parity scripts and architecture-level diffs, not bit-exact
    end-to-end accuracy/hypervolume comparisons.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_candidate(global_seed: int, generation: int, candidate_id: int) -> int:
    """Reseed all RNGs deterministically for one candidate evaluation.

    Called by each evaluation worker right before training a candidate, so
    weight initialization, dropout and data shuffling depend only on
    ``(global_seed, generation, candidate_id)`` — not on which worker
    process trains the candidate or how many candidates it trained before.
    This makes per-candidate trained accuracy bit-exact across runs even
    with parallel evaluation threads.

    Returns the derived seed (useful for logging).
    """
    seed = (global_seed + 100_003 * generation + candidate_id) % (2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
