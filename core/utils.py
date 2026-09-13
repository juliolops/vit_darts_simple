"""Seeding and logging helpers."""
import logging
import os
import random

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` so the GA is reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_candidate(global_seed: int, generation: int, candidate_id: int) -> int:
    """Reseed all RNGs for one candidate, so its training depends only on
    ``(global_seed, generation, candidate_id)`` and not on the worker that runs it."""
    seed = (global_seed + 100_003 * generation + candidate_id) % (2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed


def init_log(log_level: str, name: str, file_path: str = None) -> logging.Logger:
    """Logger writing to ``file_path`` (or stdout) at ``log_level`` ('NONE' | 'INFO' | 'DEBUG')."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.FileHandler(file_path) if file_path else logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    if log_level in ('INFO', 'DEBUG'):
        logger.setLevel(getattr(logging, log_level))
    return logger


def resolve_device(worker_rank: int = 0) -> str:
    """CUDA if present (one GPU per worker, round-robin), else Apple MPS, else CPU."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu_idx = worker_rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_idx)
        return f'cuda:{gpu_idx}'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
