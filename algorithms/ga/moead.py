# algorithms/ga/moead.py
import numpy as np
from math import comb
from .nsga2 import NSGA2
from algorithms.pareto import simplex_lattice, to_minimization

_EPS = 1e-12

class MOEAD(NSGA2):
    """
    MOEA/D: Multi-objective Evolutionary Algorithm based on Decomposition.

    Reuses NSGA2's multi-objective evaluation, logging, and Pareto-archive,
    but selection is via neighborhood replacement under a scalarizing function.
    """

    def __init__(self,
                eval_func,
                experiment_path,
                objectives,
                log_file,
                log_level,
                data_file,
                use_cache=False,
                divisions=None,            # Das & Dennis lattice p (auto if None)
                T=20,                      # neighborhood size
                scalar_method="tchebycheff",  # "tchebycheff" or "weighted_sum"
                prob_neighbor_mating=0.9):
        super().__init__(eval_func, experiment_path, objectives,
                        log_file, log_level, data_file, use_cache)
        self.divisions = divisions
        self.T = T
        self.scalar_method = scalar_method
        self.prob_neighbor_mating = prob_neighbor_mating

        self.weights = None     # (N, M)
        self.neighbors = None   # (N, T)
        self.z = None           # ideal point (M,)
        self._M = len(objectives) if objectives else None

    # ---------- reference directions & neighbors ----------
    def _build_reference_directions(self, M, N, divisions=None):
        p = int(divisions) if divisions is not None else 1
        while comb(p + M - 1, M - 1) < N:
            p += 1
        dirs = simplex_lattice(M, p)  # (K, M), rows sum to 1
        if dirs.shape[0] >= N:
            idx = np.random.choice(dirs.shape[0], size=N, replace=False)
            dirs = dirs[idx]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < _EPS] = 1.0
        return dirs / norms

    def _init_weights_and_neighbors(self):
        N = self.population_size
        M = self._M
        self.weights = self._build_reference_directions(M, N, self.divisions)  # (N, M)
        dist = np.linalg.norm(self.weights[:, None, :] - self.weights[None, :, :], axis=2)
        self.neighbors = np.argsort(dist, axis=1)[:, :min(self.T, N)]

    # ---------- scalarization ----------
    def _scalarize(self, f_vec, w, z):
        if self.scalar_method == "weighted_sum":
            return float(np.dot(w, f_vec))
        # default: Tchebycheff
        return float(np.max(w * np.abs(f_vec - z)))

    # ---------- lifecycle ----------
    def initialize_ga(self, *args, **kwargs):
        super().initialize_ga(*args, **kwargs)
        if self._M is None:
            self._M = len(self.objectives)
        self._init_weights_and_neighbors()

        # evaluate once to set ideal point
        self.current_gen = 0
        fits = self.evaluate_population()            # (N, M)
        fmin = to_minimization(fits, self.objective_senses)
        self.z = np.min(fmin, axis=0)

    def generate_offspring(self):
        N = self.population_size

        new_population = self.population.copy()
        new_params = self.pop_params.copy()

        child_disc, child_cont, owner = [], [], []

        for i in range(N):
            pool = self.neighbors[i] if np.random.rand() < self.prob_neighbor_mating else np.arange(N)
            p1, p2 = np.random.choice(pool, size=2, replace=False)
            d1, d2 = self.population[p1], self.population[p2]
            c1, c2 = self.pop_params[p1], self.pop_params[p2]

            if np.random.rand() < self.crossover_rate:
                mask = np.random.rand(*d1.shape) < 0.5
                disc = np.where(mask, d1, d2)
                alpha = np.random.rand()
                cont = alpha * c1 + (1 - alpha) * c2
            else:
                disc = d1.copy()
                cont = c1.copy()

            disc = self.mutate(disc)
            cont = self._mutate_cont(np.array(cont))

            child_disc.append(disc); child_cont.append(cont); owner.append(i)

        # batch-evaluate children
        old_pop, old_params = self.population, self.pop_params
        self.population = np.array(child_disc)
        self.pop_params = np.array(child_cont)
        child_fits = self.evaluate_population()       # (K, M)
        f_children_min = to_minimization(child_fits, self.objective_senses)
        self.population, self.pop_params = old_pop, old_params

        # update ideal
        self.z = np.minimum(self.z, np.min(f_children_min, axis=0))

        # neighborhood replacement
        for k, i in enumerate(owner):
            fy_min = f_children_min[k]
            for j in self.neighbors[i]:
                wj = self.weights[j]
                fj_min = to_minimization(self.fitnesses[j:j+1, :], self.objective_senses)[0]
                if self._scalarize(fy_min, wj, self.z) + 1e-12 < self._scalarize(fj_min, wj, self.z):
                    new_population[j] = child_disc[k]
                    new_params[j] = child_cont[k]
                    self.fitnesses[j, :] = child_fits[k, :]

        self.population = new_population
        self.pop_params = new_params

    # ---- Checkpoint hooks: add the ideal point z, weights and neighbors ----

    def _checkpoint_config_block(self) -> dict:
        b = super()._checkpoint_config_block()
        b.update({
            'ref_divisions': self.divisions,
            'moead_T': self.T,
            'scalar_method': self.scalar_method,
            'prob_neighbor_mating': self.prob_neighbor_mating,
        })
        return b

    def _checkpoint_state(self) -> dict:
        s = super()._checkpoint_state()
        # z (ideal point) accumulates the per-objective minimum across all
        # generations — losing it would reset the decomposition on resume.
        s.update({'z': self.z, 'weights': self.weights, 'neighbors': self.neighbors})
        return s

    def _restore_state(self, s: dict) -> None:
        super()._restore_state(s)
        self.z = s['z']
        self.weights = s['weights']
        self.neighbors = s['neighbors']

    def evolve(self):
        self.logger.info("Starting MOEA/D evolution")
        if getattr(self, '_resumed', False):
            # initialize_ga already rebuilt weights/neighbors/z from a fresh
            # gen-0 eval; load_checkpoint then overwrote them (and population)
            # with the restored values. Continue at g+1.
            self.logger.info("Resuming MOEA/D after completed generation %d.", self.current_gen)
            self.current_gen += 1
        while self.current_gen < self.num_generations:
            self.evaluate_population()      # refresh fitnesses for current pop
            self.generate_offspring()
            self.population_ids = [f"{self.current_gen}_{i}" for i in range(self.population.shape[0])]
            self.go_next_gen()
            if self.early_stopping and self.check_early_stopping():
                break
        return self.pareto_global_population, self.pareto_global_fitnesses
