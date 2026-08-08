"""Diversity-preservation operators for the multi-objective algorithms.

``crowding_distance`` extracted from the canonical implementation in
``algorithms/ga/nsga2.py`` (verified identical to the copy in
``algorithms/qnas/moqnas.py``). The body is verbatim — the method never
used ``self`` — and parity is checked by the Block-D synthetic scripts
before the duplicates are deleted (stages D.6/D.7).
"""
import numpy as np


def crowding_distance(fits, front):
    """
    Compute the crowding distance for individuals in a given front.

    Uses a vectorized approach over all objectives to measure solution density,
    assigning infinite distance to boundary points.

    Args:
        fits (np.ndarray): Fitness array of shape (N, M).
        front (list[int]): Indices of individuals in the front.

    Returns:
        np.ndarray: Crowding distances for each index in `front`.
    """
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
