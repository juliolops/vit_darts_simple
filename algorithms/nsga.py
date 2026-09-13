"""NSGA-II / NSGA-III (pymoo) over per-block pruning percentages: accuracy up, FLOPs down.

Each individual has one integer gene per transformer block indexing
``percentages``. pymoo generates the population (integer random sampling),
recombines it (SBX + polynomial mutation, rounded back to integers), and
selects survivors by Pareto rank plus crowding distance (NSGA-II) or
reference-direction niching (NSGA-III). Evaluation stays ours: every
generation is handed to ``eval_func`` as one batch, which trains the
candidates in parallel worker processes.

A global archive of every non-dominated candidate seen so far is saved after
each generation to ``<experiment_path>/pareto_history.pkl``.
"""
import os
import pickle

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


def run_search(algorithm_name, eval_func, experiment_path, percentages, population_size,
               num_generations, num_genes, crossover_rate, mutation_rate, seed,
               log_file, log_level):
    """Run the search and return the final archive as ``[(id, chromosome, objectives)]``."""
    logger = init_log(log_level, name=__name__, file_path=log_file)

    problem = Problem(n_var=num_genes, n_obj=len(OBJECTIVES),
                      xl=0, xu=len(percentages) - 1, vtype=int)
    algorithm = build_algorithm(algorithm_name, population_size, crossover_rate, mutation_rate)
    algorithm.setup(problem, termination=('n_gen', num_generations), seed=seed)

    archive_ids, archive_X, archive_F = [], np.empty((0, num_genes), int), np.empty((0, 2))
    history = {}
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

        history[generation] = {
            'front': [{'id': cid, 'chromosome': x.tolist(),
                       'best_accuracy': float(-f[0]), 'total_flops': float(f[1])}
                      for cid, x, f in zip(archive_ids, archive_X, archive_F)],
            'hypervolume': hypervolume(archive_F),
        }
        with open(os.path.join(experiment_path, 'pareto_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        logger.info(f"Gen {generation} -> evaluated {len(X)}, Hypervolume = "
                    f"{history[generation]['hypervolume']:.4f}, Global Front Size = {len(front)}")
        generation += 1

    return [(cid, x.tolist(), {'best_accuracy': float(-f[0]), 'total_flops': float(f[1])})
            for cid, x, f in zip(archive_ids, archive_X, archive_F)]
