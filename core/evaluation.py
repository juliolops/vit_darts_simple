# -*- coding: utf-8 -*-
""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""

import os
import torch
import time
import copy
from typing import Dict, Any, List
import torch.multiprocessing as mp
from .cnn import input, master
from utils.helpers import init_log, setup_dataset_info
from utils.seeding import seed_candidate

worker_data_loader = None

# Worst-case penalty values for fairness objectives when a candidate fails
# (timeout, OOM, etc.) — must be the least-preferred value for each goal:
#   minimize objectives → large positive (0.0 would appear as "perfect fairness")
#   maximize objectives → 0.0
_FAIRNESS_PENALTY = {
    'fairness_spd':      1.0,   # minimize: 0=perfect fairness, 1=worst disparity
    'fairness_mean_tpr': 0.0,   # maximize: 0=worst TPR
    'fairness_score':    0.0,   # maximize: 0=worst score
}

class EvalPopulation(object):
    """
    Evaluate a population using a two-stage process.
    1. Primary objectives are evaluated in parallel.
    2. Expensive post-processing objectives (e.g., Fairness) are evaluated serially.
    """
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO'):
        self.fn_dict = fn_dict
        self.logger = init_log(log_level, name=__name__)
        self.global_seed = int(params.get('seed', 42))
        self.loader = input.GenericDataLoader(params=params)
        self.train_params = setup_dataset_info(params)
        if 'objectives' not in self.train_params:
            raise KeyError("train_params must contain 'objectives' for evaluation.")
        
        objectives = self.train_params.get('objectives', [])
        self.train_params['fitness_metric'] = objectives[0] if isinstance(objectives, list) and objectives else 'best_accuracy'
        
        all_objectives = list(self.train_params['objectives'])
        
        # Initialize lists
        self.fairness_metric_names = []
        self.primary_metric_names = []
        self.parallel_train_params = copy.deepcopy(self.train_params) # Params for subprocesses

        # Check if FairnessMetric is defined in the detailed metrics configuration
        fairness_metric_config = next((m for m in self.train_params.get('metrics', []) if m['name'] == 'FairnessMetric'), None)
        self.fairness_key_map = {
                'fairness_spd': 'spd_sum',
                'fairness_mean_tpr': 'mean_tpr',
                'fairness_score': 'fairness_score'
            }
        if fairness_metric_config:
            # 1. DYNAMIC SEPARATION
            # Filter objectives containing 'fairness' into the serial list
            self.fairness_metric_names = [obj for obj in all_objectives if 'fairness' in obj]
            
            # Everything else is a primary metric (run in parallel)
            self.primary_metric_names = [obj for obj in all_objectives if 'fairness' not in obj]

            # 2. CRITICAL: Exclude FairnessMetric from parallel workers
            self.parallel_train_params['metrics'] = [
                m for m in self.train_params.get('metrics', []) if m['name'] != 'FairnessMetric'
            ]
            self.logger.info(f"Primary metrics for PARALLEL evaluation: {self.primary_metric_names}")
            self.logger.info(f"Fairness metrics for SERIAL evaluation: {self.fairness_metric_names}")
        else:
            self.primary_metric_names = all_objectives
            self.logger.info(f"All metrics will be evaluated in PARALLEL: {self.primary_metric_names}")
        
        # Combined list for tracking purposes
        self.metric_names = self.primary_metric_names + self.fairness_metric_names

    def _num_workers(self, pop_size: int) -> int:
        """Number of evaluation worker processes for this generation.

        ``workers_per_gpu`` (preferred) sets the per-GPU concurrency and is
        multiplied by the visible GPU count; the legacy ``threads`` key is
        used as-is when present (a total worker count, not per-GPU). Never
        spawn more workers than candidates.

        GPU count is read from CUDA_VISIBLE_DEVICES rather than calling any
        torch.cuda function, because is_available()/device_count() register a
        pthread_atfork handler in the NVIDIA driver; once registered in the
        parent, all forked workers get _cuda_isInBadFork=True and cannot use
        the GPU.
        """
        per_gpu = self.train_params.get('workers_per_gpu')
        if per_gpu is not None:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible is None or cuda_visible.strip() in ("", "NoDeviceFiles", "-1"):
                n_gpus = 1  # unset or CPU-only: one "slot"
            else:
                n_gpus = len([x for x in cuda_visible.split(",") if x.strip()])
            workers = max(1, int(per_gpu)) * max(1, n_gpus)
        else:
            workers = max(1, int(self.train_params.get('threads', 1)))
        return max(1, min(workers, pop_size))

    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
        pop_size = len(decoded_nets)
        n_workers = self._num_workers(pop_size)

        # Work-stealing scheduler: a single task queue feeds N workers that
        # pull until drained, so a worker that drew fast candidates keeps
        # helping instead of idling (the static round-robin partition left
        # stragglers). Per-candidate seeding keys training on candidate_id,
        # not on which worker runs it, so results stay bit-exact.
        task_queue: mp.Queue = mp.Queue()
        result_queue: mp.Queue = mp.Queue()
        stats_queue: mp.Queue = mp.Queue()
        for idx in range(pop_size):
            task_queue.put((idx, decoded_nets[idx], decoded_params[idx]))
        for _ in range(n_workers):
            task_queue.put(None)  # one sentinel per worker

        if torch.cuda.is_initialized():
            self.logger.warning(
                "CUDA is initialized in the parent process; forked workers will fail to use "
                "the GPU. Check for parent-side CUDA calls (e.g. tensors/RNG state on CUDA).")

        processes = []
        print("\n")
        self.logger.info(f"Starting the Generation {generation} with {pop_size} individuals "
                         f"({n_workers} workers)")
        evol_time_start = time.perf_counter()

        for worker_rank in range(n_workers):
            p = mp.Process(
                target=self.run_individuals,
                args=(generation, self.parallel_train_params, self.fn_dict,
                      worker_rank, task_queue, result_queue, stats_queue)
            )
            p.start()
            processes.append(p)

        results: Dict[int, Dict[str, Any]] = {}
        model_specs = [None] * pop_size
        for _ in range(pop_size):
            idx, res_dict, model_path = result_queue.get()
            results[idx] = res_dict

            if model_path and os.path.exists(model_path):
                # Save spec for later fairness eval
                model_specs[idx] = {
                    "model_path": model_path,
                    "decoded_net": decoded_nets[idx],
                    "decoded_params": decoded_params[idx],
                    "info_dir": os.path.dirname(model_path)
                }
            else:
                model_specs[idx] = None
        for p in processes:
            p.join()

        # Aggregate per-worker stats for straggler/idle visibility.
        busy = {}
        for _ in range(n_workers):
            wr, n_cand, busy_s = stats_queue.get()
            busy[wr] = (n_cand, busy_s)
        total_wall = time.perf_counter() - evol_time_start
        max_idle = max((total_wall - b[1]) for b in busy.values()) if busy else 0.0
        self.logger.info(
            "Parallel evaluation complete. workers=%d, candidates/worker=%s, "
            "max worker idle=%.1fs of %.1fs wall.",
            n_workers, {wr: busy[wr][0] for wr in sorted(busy)}, max_idle, total_wall)

        # --- FAIRNESS EVALUATION BLOCK ---
        if self.fairness_metric_names:
            # Run parallel evaluation
            fairness_results = self.evaluate_fairness_parallel_cuda(model_specs)
            self.logger.info("Merging fairness results...")
            
            for idx, f_metrics in fairness_results.items():
                if idx in results:
                    # 1. Update the candidate's result with all raw data (useful for logs/storage)
                    results[idx].update(f_metrics)
                    
                    log_parts = []
                    # 2. Extract specifically the objectives requested in the config
                    for obj_name in self.fairness_metric_names:
                        # Use the map to find the internal key (e.g., 'fairness_spd' -> 'spd_sum')
                        internal_key = self.fairness_key_map.get(obj_name)
                        
                        if not internal_key:
                            internal_key = obj_name

                        val = f_metrics.get(internal_key)
                        
                        if val is not None:
                            results[idx][obj_name] = val
                            log_parts.append(f"{obj_name}: {val:.4f}")
                        else:
                            penalty = _FAIRNESS_PENALTY.get(obj_name, 0.0)
                            self.logger.warning(
                                f"Metric '{internal_key}' not found for candidate {idx}; "
                                f"assigning penalty {penalty}")
                            results[idx][obj_name] = penalty
                            log_parts.append(f"{obj_name}: penalty={penalty}")

                    log_msg = ", ".join(log_parts)
                    self.logger.info(f"Candidate {idx} - {log_msg}")
        else:
            # remove the model files if no fairness evaluation is done
            for spec in model_specs:
                if spec and os.path.exists(spec["model_path"]):
                    try:
                        os.remove(spec["model_path"])
                    except Exception as e:
                        self.logger.warning(f"Could not remove model file {spec['model_path']}: {e}")

        evol_end_time = time.perf_counter()
        mins, secs = divmod(evol_end_time - evol_time_start, 60)
        self.logger.info(f"Total time elapsed for generation {generation}: {int(mins)}m {int(secs)}s")

        return results

    def run_individuals(self, generation, train_params, fn_dict, worker_rank,
                        task_queue: mp.Queue, result_queue: mp.Queue, stats_queue: mp.Queue):
        """Worker loop: pull candidates from ``task_queue`` until a sentinel.

        Each worker builds its loaders once and reseeds them per candidate
        from the candidate's deterministic seed, so the result of a candidate
        is independent of which worker ran it or how many it ran before.
        """
        global worker_data_loader
        gpu_device = 'cpu'
        try:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    gpu_idx = worker_rank % num_gpus
                    torch.cuda.set_device(gpu_idx)
                    gpu_device = f'cuda:{gpu_idx}'
        except Exception as e:
            self.logger.error(f"CUDA initialization failed for worker {worker_rank}: {e}. Falling back to CPU.")
            gpu_device = 'cpu'

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

        train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)

        n_done = 0
        busy = 0.0
        while True:
            task = task_queue.get()
            if task is None:  # sentinel: no more work for this worker
                break
            original_idx, decoded_net, decoded_params = task
            candidate_id = decoded_params.get('candidate_id', original_idx)
            id_str = f"{generation}_{candidate_id}"
            model_path = None
            cand_seed_id = candidate_id if isinstance(candidate_id, int) else original_idx
            t0 = time.perf_counter()
            # A CUDA OOM gets ONE retry (after clearing the cache) before the
            # candidate is scored 0.0, so transient memory pressure under high
            # worker concurrency does not silently inject zero-fitness
            # candidates into the search. Each attempt reseeds from the
            # candidate's deterministic seed; non-OOM errors fail immediately.
            results_dict = None
            for attempt in range(2):
                # Per-candidate deterministic seeding: accuracy must not depend
                # on which worker trains the candidate or in what order.
                cand_seed = seed_candidate(self.global_seed, generation, cand_seed_id)
                # The loader is shared by all candidates of this process; reseed
                # its generator so shuffle order/augmentation don't depend on
                # how many candidates were trained before in the same process.
                if getattr(train_loader, 'generator', None) is not None:
                    train_loader.generator.manual_seed(cand_seed)
                try:
                    results_dict, model_path = master.fitness(id_str,
                                {**train_params, 'device': gpu_device},
                                fn_dict,
                                decoded_net,
                                decoded_params,
                                train_loader,
                                val_loader)
                    break
                except RuntimeError as e:
                    is_oom = 'out of memory' in str(e).lower()
                    if is_oom and attempt == 0:
                        self.logger.error(
                            f"CUDA OOM training model {id_str} (worker {worker_rank}); "
                            f"clearing cache and retrying once.")
                        if str(gpu_device).startswith('cuda'):
                            torch.cuda.empty_cache()
                        continue
                    self.logger.error(
                        f"{'CUDA OOM' if is_oom else 'RuntimeError'} training model "
                        f"{id_str} after {attempt + 1} attempt(s); scoring 0.0: {e}")
                    results_dict = {k: 0.0 for k in self.metric_names}
                    break
                except Exception as e:
                    self.logger.error(f"Error training model {id_str}: {e}")
                    results_dict = {k: 0.0 for k in self.metric_names}
                    break

            busy += time.perf_counter() - t0
            # Filter results (primary metrics only)
            filtered_results = {k: results_dict.get(k, 0.0) for k in self.primary_metric_names}
            result_queue.put((original_idx, filtered_results, model_path))
            n_done += 1

            if not results_dict:
                self.logger.warning(f"Worker {worker_rank} – candidate {original_idx}: No results returned.")
            else:
                metrics_log = ", ".join(f"{k}={results_dict.get(k, 0.0):.3f}" for k in self.primary_metric_names)
                self.logger.info(f"Worker {worker_rank} – candidate {original_idx}: {metrics_log}")

        stats_queue.put((worker_rank, n_done, busy))
    
    def evaluate_fairness_parallel_cuda(self, model_specs: list, processes_per_gpu: int = 10) -> Dict[int, Dict[str, float]]:
        import torch.multiprocessing as mp
        from core.fairness.fairness_worker import (
            device_count_probe_runner,
            fairness_queue_runner,
        )

        metric_config = next((m for m in self.train_params.get('metrics', [])
                            if m['name'] == 'FairnessMetric'), None)
        
        # If no config or no metrics, return zeros
        if not metric_config or not self.fairness_metric_names:
            return {i: {name: _FAIRNESS_PENALTY.get(name, 0.0) for name in self.fairness_metric_names} for i in range(len(model_specs))}
        
        fairness_params = metric_config.get('params', {}) or {}
        # Fairness evaluation follows the run's precision policy (Area 4);
        # a metric-level 'precision' in the config may still override it.
        fairness_params.setdefault('precision', self.train_params.get('precision', 'fp32'))
        input_shape = self.train_params.get('input_shape', (3, 224, 224))
        fairness_params['img_size'] = input_shape[2]

        # Probe GPU count
        ctx = mp.get_context("spawn")
        probe_q = ctx.Queue()
        probe_p = ctx.Process(target=device_count_probe_runner, args=(probe_q,))
        probe_p.start()
        num_gpus = probe_q.get()
        probe_p.join()

        indices = [i for i, spec in enumerate(model_specs) if spec]
        if not indices:
            return {i: {name: _FAIRNESS_PENALTY.get(name, 0.0) for name in self.fairness_metric_names} for i in range(len(model_specs))}

        slots = max(0, num_gpus * max(1, int(processes_per_gpu)))
        if slots == 0:
            merged = {}
            for i in range(len(model_specs)):
                merged.setdefault(i, {name: _FAIRNESS_PENALTY.get(name, 0.0) for name in self.fairness_metric_names})
            return merged

        shards = [[] for _ in range(slots)]
        for k, idx in enumerate(indices):
            shards[k % slots].append((idx, model_specs[idx]))

        res_q = ctx.Queue()
        procs = []
        for slot_id, shard in enumerate(shards):
            if not shard:
                continue
            dev_idx = slot_id % num_gpus
            p = ctx.Process(
                target=fairness_queue_runner,
                args=(
                    res_q,
                    shard,
                    self.parallel_train_params,
                    self.fn_dict,
                    self.fairness_metric_names, # PASSING LIST HERE
                    fairness_params,
                    dev_idx,
                ),
            )
            p.start()
            procs.append(p)

        merged: Dict[int, Dict[str, float]] = {}
        for _ in range(len(procs)):
            part = res_q.get()
            merged.update(part)

        for p in procs:
            p.join()

        # Fill any missing entries (failed candidates get worst-case penalty)
        for i in range(len(model_specs)):
            merged.setdefault(i, {name: _FAIRNESS_PENALTY.get(name, 0.0) for name in self.fairness_metric_names})

        return merged