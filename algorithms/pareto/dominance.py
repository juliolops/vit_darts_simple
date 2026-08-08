"""Pareto-dominance operators shared by the multi-objective algorithms.

``dominates`` and ``fast_nondominated_sort`` extracted from the canonical
implementations in ``algorithms/ga/nsga2.py`` (identical to the copies in
``algorithms/qnas/moqnas.py``), parametrized by ``objective_senses``
instead of reading ``self.objective_senses``. Bodies are otherwise
verbatim; numeric behavior is checked by the Block-D synthetic parity
scripts before any duplicate is deleted (stages D.6/D.7).
"""
import numpy as np


def dominates(a, b, objective_senses):
    """Determine Pareto domination between two fitness tuples.

    Parameters
    ----------
    a, b : array_like
        Fitness vectors of equal length (one value per objective).
    objective_senses : list of str
        Per-objective optimization sense, ``'max'`` or ``'min'``;
        maximization objectives are sign-flipped so the comparison is done
        in minimization form.

    Returns
    -------
    bool
        True if ``a`` Pareto-dominates ``b`` (no worse in every objective
        and strictly better in at least one).
    """
    # Create copies of the fitness values to avoid modifying the originals
    obj_a = np.array(a, copy=True)
    obj_b = np.array(b, copy=True)

    # Flip the sign for maximization objectives to convert them to minimization
    for i, sense in enumerate(objective_senses):
        if sense == 'max':
            obj_a[i] = -obj_a[i]
            obj_b[i] = -obj_b[i]

    return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)


def fast_nondominated_sort(fits, objective_senses):
    """Perform non-dominated sorting on a set of fitness vectors.

    Parameters
    ----------
    fits : np.ndarray
        Array of shape (N, M) for N individuals and M objectives.
    objective_senses : list of str
        Per-objective sense (``'max'``/``'min'``) forwarded to
        :func:`dominates`.

    Returns
    -------
    list of list of int
        A list of fronts, each a list of individual indices.
        ``fronts[0]`` is the first Pareto front, etc.
    """
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
