# nsga3.py
import numpy as np
from math import comb
from .nsga2 import NSGA2
from algorithms.pareto import fast_nondominated_sort, simplex_lattice, to_minimization

_EPS = 1e-12

class NSGA3(NSGA2):
    """
    NSGA-III: many-objective evolutionary algorithm with reference directions.

    Inherits most behavior from your NSGA2 (evaluation cache, sorting, history,
    evolve loop, logging, etc.). Only overrides:
        - environmental_selection (Niching w/ reference directions)
        - generate_offspring (tournament on rank only; tie -> random)
    """

    def __init__(self, eval_func, experiment_path, objectives, log_file, log_level, data_file,
                use_cache=False, ref_divisions=None, maximize_first=True):
        super().__init__(eval_func, experiment_path, objectives, log_file, log_level, data_file, use_cache)
        self.maximize_first = maximize_first
        self.ref_divisions = ref_divisions  # if None -> auto based on pop size and M
        self._ref_dirs = None  # computed on first use

    # ---------- Public API differences ----------

    # ---- Checkpoint hooks: add the reference directions to the GA/NSGA2 state ----

    def _checkpoint_config_block(self) -> dict:
        b = super()._checkpoint_config_block()
        b['ref_divisions'] = self.ref_divisions
        return b

    def _checkpoint_state(self) -> dict:
        s = super()._checkpoint_state()
        s['_ref_dirs'] = self._ref_dirs
        return s

    def _restore_state(self, s: dict) -> None:
        super()._restore_state(s)
        self._ref_dirs = s['_ref_dirs']

    def generate_offspring(self):
        """
        NSGA-III usually uses rank-only tournament; ties broken at random.
        """
        pop_size = self.population_size
        gene_len = self.population.shape[1]
        new_pop = np.empty((pop_size, gene_len), dtype=self.population.dtype)

        fronts = fast_nondominated_sort(self.fitnesses, self.objective_senses)
        rank = np.empty(len(self.population), dtype=int)
        for r, fr in enumerate(fronts):
            for idx in fr:
                rank[idx] = r

        i = 0
        while i < pop_size:
            p1 = self._tournament_rank_only(rank)
            p2 = self._tournament_rank_only(rank)
            if np.random.rand() < self.crossover_rate:
                c1, c2 = self.crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop[i] = self.mutate(c1)
            if i + 1 < pop_size:
                new_pop[i+1] = self.mutate(c2)
            i += 2
        self.population = new_pop

    def environmental_selection(self, pop, fits):
        """
        NSGA-III environmental selection:
            1) Fill fronts as in NSGA-II.
            2) If last front partially fits: do NSGA-III niching using reference directions.
        """
        M = fits.shape[1]
        if M <= 2:
            # NSGA-III reduces to NSGA-II niching for bi-objective; reuse your implementation.
            return super().environmental_selection(pop, fits)

        pop_size = self.population_size
        fronts = fast_nondominated_sort(fits, self.objective_senses)

        selected_idx = []
        for fr in fronts:
            if len(selected_idx) + len(fr) <= pop_size:
                selected_idx.extend(fr)
            else:
                # Need to select remaining from this (last) front using niching
                rem = pop_size - len(selected_idx)
                last_front = fr

                # Build candidate set L = selected ∪ last_front for normalization/association
                L_idx = selected_idx + last_front
                F = to_minimization(fits[L_idx], self.objective_senses)
                # reference directions prepared for M objectives
                if self._ref_dirs is None:
                    self._ref_dirs = self._build_reference_directions(M, pop_size, self.ref_divisions)

                # Normalize wrt L (ideal point + intercepts)
                F_norm = self._normalize(F)  # shape (|L|, M)

                # Association
                asg_idx, perp_dist = self._associate_to_ref_dirs(F_norm, self._ref_dirs)

                # Niche counts: among already selected (first |selected_idx| rows in L)
                niche_count = self._niche_count(asg_idx[:len(selected_idx)], n_refs=len(self._ref_dirs))

                # Candidates for selection are the last front portion in L
                cand_mask = np.zeros(len(L_idx), dtype=bool)
                cand_mask[len(selected_idx):] = True
                chosen_from_last = self._niching_select(
                    rem=rem,
                    asg_idx=asg_idx,
                    perp_dist=perp_dist,
                    cand_mask=cand_mask,
                    niche_count=niche_count
                )

                selected_idx.extend([last_front[i] for i in chosen_from_last])
                break

        selected_idx = np.array(selected_idx, dtype=int)
        new_pop = pop[selected_idx]
        new_fits = fits[selected_idx]
        return new_pop, new_fits, selected_idx.tolist()

    # ---------- Helpers: tournaments & objective orientation ----------

    def _tournament_rank_only(self, rank):
        i, j = np.random.choice(len(rank), 2, replace=False)
        if rank[i] < rank[j]:
            return self.population[i]
        if rank[j] < rank[i]:
            return self.population[j]
        # tie -> random
        return self.population[i] if np.random.rand() < 0.5 else self.population[j]

    # ---------- NSGA-III core: ref dirs, normalization, association, niching ----------

    def _build_reference_directions(self, M, pop_size, divisions=None):
        """
        Generate reference directions on the unit simplex (Das & Dennis),
        building the lattice via algorithms.pareto.simplex_lattice.
        If divisions is None, pick the smallest p with C(p+M-1, M-1) >= pop_size.
        No-prune policy: extra directions are kept for better spread (unlike
        moead's variant, which prunes randomly — reason this method is not
        consolidated into algorithms.pareto).
        """
        p = divisions
        if p is None:
            p = 1
            while comb(p + M - 1, M - 1) < pop_size:
                p += 1
        dirs = simplex_lattice(M, p)  # shape (K, M)
        # In rare cases K >> pop_size; that's okay (better spread). No random pruning here.
        # Normalize directions to unit length for distance computations.
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < _EPS] = 1.0
        return dirs / norms

    def _normalize(self, F):
        """
        Normalize F (minimization) via ideal point + intercepts (Deb & Jain 2014).
        """
        zmin = np.min(F, axis=0)
        Ft = F - zmin  # translate to ideal point

        # 1) Extreme points using ASF
        M = F.shape[1]
        asf_ext_idx = []
        for j in range(M):
            w = np.full(M, _EPS)
            w[j] = 1.0
            asf = np.max(Ft / w, axis=1)
            asf_ext_idx.append(int(np.argmin(asf)))
        A = Ft[asf_ext_idx, :]  # shape (M, M)

        # 2) Intercepts via solving A * a = 1
        intercepts = None
        try:
            ones = np.ones(M)
            a = np.linalg.solve(A, ones)
            intercepts = 1.0 / a
        except np.linalg.LinAlgError:
            pass

        # Fallbacks if singular or bad intercepts
        if intercepts is None or np.any(np.isnan(intercepts)) or np.any(intercepts <= _EPS):
            intercepts = np.max(Ft, axis=0)
        intercepts[intercepts < _EPS] = 1.0

        F_norm = Ft / intercepts
        F_norm[np.isnan(F_norm)] = 0.0
        return F_norm

    def _associate_to_ref_dirs(self, F_norm, ref_dirs):
        """
        For each point, find the closest reference direction by perpendicular distance.
        Returns:
            nearest_ref_idx: (N,) int
            perp_dist:       (N,) float
        """
        # Normalize point vectors
        norms = np.linalg.norm(F_norm, axis=1, keepdims=True)
        norms[norms < _EPS] = 1.0
        V = F_norm / norms

        # Cosine to each reference
        # (N, M) @ (M, K) -> (N, K)
        cos = V @ ref_dirs.T
        cos = np.clip(cos, -1.0, 1.0)

        # Perpendicular distance = ||v|| * sqrt(1 - cos^2)
        perp = norms[:, 0:1] * np.sqrt(1.0 - cos**2)

        nearest = np.argmin(perp, axis=1)
        d = perp[np.arange(len(F_norm)), nearest]
        return nearest, d

    def _niche_count(self, assoc_selected, n_refs):
        """
        Count how many already-selected individuals are associated to each reference dir.
        """
        counts = np.zeros(n_refs, dtype=int)
        for r in assoc_selected:
            counts[r] += 1
        return counts

    def _niching_select(self, rem, asg_idx, perp_dist, cand_mask, niche_count):
        """
        Niching procedure to pick 'rem' individuals among candidates (cand_mask True).
        Strategy:
            - repeatedly choose the reference with the smallest niche_count
            - from its candidates: if niche_count==0 -> pick smallest perpendicular distance
                                else -> pick a random candidate
        Returns list of indices *within the last front*, i.e., relative positions into that front.
        """
        chosen_local = []
        n_refs = len(niche_count)

        # Indices within L for candidates only
        cand_L_idx = np.where(cand_mask)[0]
        # Map ref -> list of L indices of its candidates
        groups = {r: [] for r in range(n_refs)}
        for li in cand_L_idx:
            groups[asg_idx[li]].append(li)

        # Keep track which candidates remain per ref
        for r in range(n_refs):
            # sort by distance for deterministic selection when niche empty
            groups[r].sort(key=lambda li: perp_dist[li])

        # While we still need to pick
        while rem > 0:
            # References with at least one remaining candidate
            nonempty = [r for r in range(n_refs) if len(groups[r]) > 0]
            if not nonempty:
                break  # nothing left; should not usually happen

            # Pick ref with minimum niche count (ties random)
            min_count = min(niche_count[r] for r in nonempty)
            candidates_refs = [r for r in nonempty if niche_count[r] == min_count]
            r_pick = np.random.choice(candidates_refs)

            # Choose individual from that ref
            if niche_count[r_pick] == 0:
                # pick the closest (smallest perp distance)
                li = groups[r_pick].pop(0)
            else:
                # pick random among that ref's remaining
                k = np.random.randint(len(groups[r_pick]))
                li = groups[r_pick].pop(k)

            niche_count[r_pick] += 1
            rem -= 1

            # Convert this L-index back to local index inside *last front*
            # L = [selected ...] + [last_front ...]; last front starts at offset 'S'
            S = np.sum(~cand_mask)  # number of already-selected in L
            chosen_local.append(li - S)

        return chosen_local
