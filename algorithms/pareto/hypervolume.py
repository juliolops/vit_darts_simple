"""Hypervolume computation for mixed max/min objective fronts.

Canonical ``compute_hypervolume_mixed`` extracted from the class
implementations in ``algorithms/ga/nsga2.py`` / ``algorithms/qnas/moqnas.py``
(code-identical; only their docstrings differed), parametrized by
``objective_senses`` instead of reading ``self.objective_senses``. The
deprecated standalone in ``utils/visualization.py`` is NOT the source:
it hardcodes the column-0 flip and does not flip a user-supplied
``ref_point``.
"""
import numpy as np
from pymoo.indicators.hv import Hypervolume


def compute_hypervolume_mixed(front_raw: np.ndarray, objective_senses, ref_point=None) -> float:
    """Compute the hypervolume of a Pareto front with mixed objective senses.

    Parameters
    ----------
    front_raw : np.ndarray
        Array of shape (N, M) with raw fitness values, one column per
        objective in its original sense (e.g. accuracy to maximize,
        parameter count to minimize).
    objective_senses : list of str
        Per-objective sense, ``'max'`` or ``'min'``. Maximization columns
        are sign-flipped to convert the front to minimization form.
    ref_point : array_like, optional
        Reference point expressed in the ORIGINAL mixed-objective space;
        its maximization components are sign-flipped consistently with the
        front. If None, a safe point slightly worse than the worst
        observed value per objective is used.

    Returns
    -------
    float
        The hypervolume (pymoo ``Hypervolume`` on the minimization form).
        0.0 for an empty/None front.

    Notes
    -----
    Unlike the deprecated standalone in ``utils.visualization`` (which
    hardcodes flipping column 0 and passes ``ref_point`` through
    unflipped), this canonical version flips every ``'max'`` objective in
    BOTH the front and the reference point, so it works for any number
    and ordering of objectives.
    """
    if front_raw is None or len(front_raw) == 0:
        return 0.0

    f = np.array(front_raw, dtype=float, copy=True)

    # Flip the sign for maximization objectives
    for i, sense in enumerate(objective_senses):
        if sense == 'max':
            f[:, i] = -f[:, i]

    # Choose a safe reference point (must be worse than all points for minimization)
    if ref_point is None:
        rp = np.max(f, axis=0) + 1e-6
    else:
        rp = np.asarray(ref_point, dtype=float)
        # Flip the sign for maximization objectives in the reference point as well
        for i, sense in enumerate(objective_senses):
            if sense == 'max':
                rp[i] = -rp[i]

    return float(Hypervolume(ref_point=rp).do(f))
