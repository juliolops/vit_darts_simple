"""Shared Pareto operators for the multi-objective algorithms (Block D).

Public API (canonical, parametrized versions of the operators that were
duplicated across nsga2/moqnas/nsga3/moead):

- ``dominates(a, b, objective_senses)``
- ``fast_nondominated_sort(fits, objective_senses)``
- ``crowding_distance(fits, front)``
- ``compute_hypervolume_mixed(front_raw, objective_senses, ref_point=None)``
- ``simplex_lattice(M, p)``
- ``to_minimization(fits, objective_senses)``

``build_reference_directions`` is intentionally NOT part of this package:
the nsga3 and moead implementations diverge (no pruning vs random
pruning), so each algorithm keeps its own.
"""
from .dominance      import dominates, fast_nondominated_sort
from .diversity      import crowding_distance
from .hypervolume    import compute_hypervolume_mixed
from .reference_dirs import simplex_lattice, to_minimization
