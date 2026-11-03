"""NSGA-II / NSGA-III (pymoo) over per-block pruning percentages: accuracy up, FLOPs down.

Each individual has one integer gene per transformer block indexing
``percentages``. pymoo generates the population (integer random sampling),
recombines it (SBX + polynomial mutation, rounded back to integers), and
selects survivors by Pareto rank plus crowding distance (NSGA-II) or
reference-direction niching (NSGA-III). Evaluation stays ours: every
generation is handed to ``eval_func`` as one batch, which trains the
candidates in parallel worker processes.

After each generation ``<experiment_path>/pareto_history.pkl`` is rewritten with,
per generation: the global archive of every non-dominated candidate seen so far
(``front``) and its hypervolume, every candidate evaluated in that generation
(``evaluated``), the generation's best candidate (``best``: highest accuracy,
fewest FLOPs on ties) and the best candidate of the whole search so far
(``best_so_far``). The same data, readable without Python, goes to
``<experiment_path>/generations.json``.

The full search state (the pymoo algorithm with its population, the archive,
the history and the random number generators) is checkpointed after each
generation to ``<experiment_path>/checkpoint.pkl``, so an interrupted search
resumes from the next generation with ``resume=True`` (``--resume``) and gives
the same result as an uninterrupted run.
"""
import json
import os
import pickle
import random

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.problems.static import StaticProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from core.utils import init_log

OBJECTIVES = ('best_accuracy', 'total_flops')


def to_minimization(results) -> np.ndarray:
    """``[{best_accuracy, total_flops}, ...]`` -> (N, 2) array to minimize."""
    return np.array([[-r['best_accuracy'], r['total_flops']] for r in results], dtype=float)


def best_candidate(candidates):
    """Highest accuracy; on ties, the fewest FLOPs."""
    return max(candidates, key=lambda c: (c['best_accuracy'], -c['total_flops']))


def hypervolume(F: np.ndarray) -> float:
    """Hypervolume of a minimization front w.r.t. a point just worse than its worst values."""
    return float(HV(ref_point=F.max(axis=0) + 1e-6)(F))


def build_algorithm(name, population_size, crossover_rate, mutation_rate):
    """pymoo NSGA-II or NSGA-III with integer-gene operators."""
    operators = dict(
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=crossover_rate, eta=3.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, prob_var=mutation_rate, eta=3.0, vtype=float,
                    repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    if name == 'nsga2':
        return NSGA2(pop_size=population_size, **operators)
    if name == 'nsga3':
        # One reference direction per individual on the 2-objective simplex.
        ref_dirs = get_reference_directions('das-dennis', len(OBJECTIVES),
                                            n_partitions=population_size - 1)
        return NSGA3(ref_dirs=ref_dirs, pop_size=population_size, **operators)
    raise ValueError(f"Unknown algorithm {name!r}; expected 'nsga2' or 'nsga3'.")


CHECKPOINT = 'checkpoint.pkl'


def _atomic_pickle(obj, path):
    """Write ``obj`` to ``path`` so that a crash mid-write never leaves a broken file."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(experiment_path):
    """The saved search state, or ``None`` when the experiment has no checkpoint."""
    path = os.path.join(experiment_path, CHECKPOINT)
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_search(algorithm_name, eval_func, experiment_path, percentages, population_size,
               num_generations, num_genes, crossover_rate, mutation_rate, seed,
               log_file, log_level, resume=False, run_info=None):
    """Run the search and return the final archive as ``[(id, chromosome, objectives)]``.

    With ``resume=True`` and a checkpoint in ``experiment_path``, the search continues
    from the generation after the last one completed. ``run_info`` (config file,
    dataset, FLOPs counter...) is stored in the checkpoint and must match on resume,
    so a search is never continued with different settings. ``num_generations`` may
    be raised on resume to extend a finished search.
    """
    logger = init_log(log_level, name=__name__, file_path=log_file)
    settings = {'algorithm': algorithm_name, 'population_size': population_size,
                'num_genes': num_genes, 'percentages': list(percentages),
                'crossover_rate': crossover_rate, 'mutation_rate': mutation_rate,
                'seed': seed, **(run_info or {})}

    state = load_checkpoint(experiment_path) if resume else None
    if state is not None:
        saved = state['settings']
        changed = {k: (saved.get(k), v) for k, v in settings.items() if saved.get(k) != v}
        if changed:
            details = ', '.join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in changed.items())
            raise ValueError(f"Cannot resume {experiment_path}: settings differ from the "
                             f"checkpoint ({details}). Use the same command, or a new "
                             f"--experiment_path for a different search.")
        algorithm = state['algorithm']
        from pymoo.termination import get_termination
        algorithm.termination = get_termination('n_gen', num_generations)
        problem = algorithm.problem
        archive_ids, archive_X, archive_F = state['archive']
        history, best_so_far = state['history'], state['best_so_far']
        generation = state['generation']
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        logger.info(f"Resuming {experiment_path} at generation {generation} "
                    f"({generation} of {num_generations} already done)")
    else:
        problem = Problem(n_var=num_genes, n_obj=len(OBJECTIVES),
                          xl=0, xu=len(percentages) - 1, vtype=int)
        algorithm = build_algorithm(algorithm_name, population_size, crossover_rate,
                                    mutation_rate)
        algorithm.setup(problem, termination=('n_gen', num_generations), seed=seed)
        archive_ids, archive_X, archive_F = [], np.empty((0, num_genes), int), np.empty((0, 2))
        history = {}
        best_so_far = None
        generation = 0

    while algorithm.has_next():
        pop = algorithm.ask()
        X = pop.get('X').astype(int)

        results = eval_func([[percentages[g] for g in x] for x in X], generation=generation)
        F = to_minimization([results[i] for i in range(len(X))])
        Evaluator().eval(StaticProblem(problem, F=F), pop)
        algorithm.tell(infills=pop)

        # Global archive: non-dominated set of everything evaluated so far.
        archive_ids += [f"{generation}_{i}" for i in range(len(X))]
        archive_X = np.vstack([archive_X, X])
        archive_F = np.vstack([archive_F, F])
        front = NonDominatedSorting().do(archive_F, only_non_dominated_front=True)
        archive_ids = [archive_ids[i] for i in front]
        archive_X, archive_F = archive_X[front], archive_F[front]

        def point(cid, x, f):
            genes = [int(g) for g in x]
            return {'id': cid, 'chromosome': genes,
                    'percentages': [percentages[g] for g in genes],
                    'best_accuracy': float(-f[0]), 'total_flops': float(f[1])}

        evaluated = [point(f"{generation}_{i}", x, f) for i, (x, f) in enumerate(zip(X, F))]
        best = best_candidate(evaluated)
        if best_so_far is None or best_candidate([best_so_far, best]) is best:
            best_so_far = best

        history[generation] = {
            'front': [point(cid, x, f) for cid, x, f in zip(archive_ids, archive_X, archive_F)],
            'hypervolume': hypervolume(archive_F),
            'evaluated': evaluated,
            'best': best,
            'best_so_far': best_so_far,
        }
        with open(os.path.join(experiment_path, 'pareto_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        with open(os.path.join(experiment_path, 'generations.json'), 'w') as f:
            json.dump({str(g): rec for g, rec in history.items()}, f, indent=2)
        logger.info(f"Gen {generation} -> evaluated {len(X)}, Hypervolume = "
                    f"{history[generation]['hypervolume']:.4f}, Global Front Size = {len(front)}")
        logger.info(f"Gen {generation} -> best: {best['id']} acc={best['best_accuracy']:.4f} "
                    f"flops={best['total_flops']:.4g} percentages={best['percentages']} | "
                    f"best so far: {best_so_far['id']} acc={best_so_far['best_accuracy']:.4f}")
        generation += 1

        # Everything needed to continue from the next generation.
        _atomic_pickle({'settings': settings, 'generation': generation,
                        'algorithm': algorithm,
                        'archive': (archive_ids, archive_X, archive_F),
                        'history': history, 'best_so_far': best_so_far,
                        'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()},
                       os.path.join(experiment_path, CHECKPOINT))

    return [(cid, x.tolist(), {'best_accuracy': float(-f[0]), 'total_flops': float(f[1])})
            for cid, x, f in zip(archive_ids, archive_X, archive_F)]
