"""Reference-direction utilities shared by NSGA-III and MOEA/D.

``simplex_lattice`` and ``to_minimization`` extracted from
``algorithms/ga/nsga3.py`` (the moead copies differ only cosmetically:
docstrings, joined statements and a parameter name; the executable logic
is identical), parametrized instead of reading ``self``.

``_build_reference_directions`` is deliberately NOT consolidated here:
the nsga3 version never prunes the lattice (extra directions are kept
for better spread), while the moead version prunes down to the requested
population size with a random ``np.random.choice``. The two behaviors
diverge, so each algorithm keeps its own implementation.
"""
import numpy as np


def simplex_lattice(M, p):
    """Integer compositions of ``p`` into ``M`` parts, scaled by ``p``.

    Das & Dennis-style lattice on the unit simplex.

    Parameters
    ----------
    M : int
        Number of objectives (parts of the composition).
    p : int
        Number of divisions along each objective axis.

    Returns
    -------
    np.ndarray
        Array of shape (K, M) with rows summing to 1.0, where
        ``K = C(p + M - 1, M - 1)``.
    """
    out = []
    buf = [0]*M
    def rec(depth, left):
        if depth == M-1:
            buf[depth] = left
            out.append(buf.copy())
            return
        for v in range(left+1):
            buf[depth] = v
            rec(depth+1, left-v)
    rec(0, p)
    arr = np.array(out, dtype=float)
    return arr / max(p, 1)


def to_minimization(fits, objective_senses):
    """Convert a fitness matrix to minimization form.

    Parameters
    ----------
    fits : array_like
        Array of shape (N, M) with raw fitness values.
    objective_senses : list of str
        Per-objective sense, ``'max'`` or ``'min'``; maximization columns
        are sign-flipped.

    Returns
    -------
    np.ndarray
        A float copy of ``fits`` where every ``'max'`` column is negated.
    """
    f = np.array(fits, dtype=float, copy=True)
    for i, sense in enumerate(objective_senses):
        if sense == 'max':
            f[:, i] = -f[:, i]
    return f
