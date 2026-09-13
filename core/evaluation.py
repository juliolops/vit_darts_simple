"""Evaluate a population in parallel worker processes.

A single task queue feeds ``threads`` workers that pull candidates until it is
drained. Each worker picks its device (CUDA -> MPS -> CPU), builds the data
loaders once and reseeds everything per candidate, so a candidate's result does
not depend on which worker trains it.
"""
import os
import time
from typing import Dict, List

import torch
import torch.multiprocessing as mp

from core.training import evaluate_candidate
from core.utils import init_log, resolve_device, seed_candidate

OBJECTIVES = ('best_accuracy', 'total_flops')


def _worker(worker_rank, generation, params, log_level, task_queue, result_queue):
    from core.data import build_loaders

    logger = init_log(log_level, name=__name__)
    try:
        device = resolve_device(worker_rank)
    except Exception as e:
        logger.error(f"GPU initialization failed for worker {worker_rank}: {e}. Using CPU.")
        device = 'cpu'
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

    train_loader, val_loader = build_loaders(params, device)

    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, percentages = task
        cand_seed = seed_candidate(params['seed'], generation, idx)
        train_loader.generator.manual_seed(cand_seed)
        try:
            result = evaluate_candidate(percentages, params, device, train_loader, val_loader)
        except Exception as e:
            logger.error(f"Error training candidate {generation}_{idx}: {e}; scoring 0.0")
            result = {k: 0.0 for k in OBJECTIVES}
        logger.info(f"Worker {worker_rank} - candidate {idx}: "
                    + ", ".join(f"{k}={result[k]:.3f}" for k in OBJECTIVES))
        result_queue.put((idx, result))


class EvalPopulation:
    """Callable mapping a list of decoded chromosomes to ``{index: objectives}``."""

    def __init__(self, params: dict, log_level: str = 'INFO'):
        self.params = params
        self.log_level = log_level
        self.logger = init_log(log_level, name=__name__)

    def __call__(self, population: List[List[int]], generation: int) -> Dict[int, dict]:
        pop_size = len(population)
        n_workers = max(1, min(int(self.params['threads']), pop_size))

        task_queue, result_queue = mp.Queue(), mp.Queue()
        for idx, percentages in enumerate(population):
            task_queue.put((idx, percentages))
        for _ in range(n_workers):
            task_queue.put(None)

        self.logger.info(f"Starting generation {generation} with {pop_size} individuals "
                         f"({n_workers} workers)")
        t0 = time.perf_counter()
        processes = [mp.Process(target=_worker,
                                args=(rank, generation, self.params, self.log_level,
                                      task_queue, result_queue))
                     for rank in range(n_workers)]
        for p in processes:
            p.start()
        results = dict(result_queue.get() for _ in range(pop_size))
        for p in processes:
            p.join()

        mins, secs = divmod(time.perf_counter() - t0, 60)
        self.logger.info(f"Generation {generation} evaluated in {int(mins)}m {int(secs)}s")
        return results
