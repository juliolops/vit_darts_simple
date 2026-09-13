"""Pareto operators for fitness matrices with mixed max/min objectives."""
import numpy as np
from pymoo.indicators.hv import Hypervolume


def dominates(a, b, objective_senses) -> bool:
    """True if ``a`` is no worse than ``b`` in every objective and strictly better in one."""
    obj_a = np.array(a, copy=True)
    obj_b = np.array(b, copy=True)
    for i, sense in enumerate(objective_senses):
        if sense == 'max':
            obj_a[i] = -obj_a[i]
            obj_b[i] = -obj_b[i]
    return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)


def fast_nondominated_sort(fits, objective_senses):
    """Split the rows of ``fits`` (N, M) into fronts; ``fronts[0]`` is the Pareto front."""
    N = len(fits)
    dominated = [set() for _ in range(N)]
    dom_count = np.zeros(N, dtype=int)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if dominates(fits[p], fits[q], objective_senses):
                dominated[p].add(q)
            elif dominates(fits[q], fits[p], objective_senses):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(fits, front):
    """Crowding distance of each index in ``front``; boundary points get infinity."""
    f = fits[front]
    F, M = f.shape
    dist = np.zeros(F)
    if F <= 2:
        return np.array([np.inf] * F)
    sorted_idx = np.argsort(f, axis=0)
    dist[sorted_idx[0, :]] = np.inf
    dist[sorted_idx[-1, :]] = np.inf
    min_vals = f[sorted_idx[0, :], np.arange(M)]
    max_vals = f[sorted_idx[-1, :], np.arange(M)]
    denom = max_vals - min_vals
    for j in range(M):
        if denom[j] == 0:
            continue
        prev = f[sorted_idx[:-2, j], j]
        nxt = f[sorted_idx[2:, j], j]
        dist[sorted_idx[1:-1, j]] += (nxt - prev) / denom[j]
    return dist


def compute_hypervolume(front, objective_senses) -> float:
    """Hypervolume of ``front`` w.r.t. a point just worse than its worst value per objective."""
    if front is None or len(front) == 0:
        return 0.0
    f = np.array(front, dtype=float, copy=True)
    for i, sense in enumerate(objective_senses):
        if sense == 'max':
            f[:, i] = -f[:, i]
    ref_point = np.max(f, axis=0) + 1e-6
    return float(Hypervolume(ref_point=ref_point).do(f))
