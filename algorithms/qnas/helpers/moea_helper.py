# In algorithms/qnas/moea_helper.py

import numpy as np
from typing import List, Dict

from .configs import MOEAConfig 

class MOEADHelper:
    """Encapsulates all logic for MOEA/D updates and objective handling."""
    
    def __init__(self, num_objectives: int, objective_sense: List[str], 
                num_quantum_ind: int, moea_config: MOEAConfig):
        """
        Initializes the helper and generates reference directions.

        Args:
            num_objectives (int): The number of objectives (M).
            objective_sense (List[str]): List of "min" or "max" for each objective.
            num_quantum_ind (int): Number of quantum individuals (Nq).
            moea_config (MOEAConfig): The configuration object with MOEA/D params.
        """
        self.num_objectives = num_objectives
        self.objective_sense = objective_sense
        self.num_quantum_ind = num_quantum_ind
        self.config = moea_config

        # Store these for the update logic
        self.moead_q_low = self.config.moead_q_low
        self.moead_q_high = self.config.moead_q_high
        self.topP_mult = self.config.topP_mult

        # Generate and assign reference directions
        self._ref_dirs = self._make_ref_directions(
            M=self.num_objectives, 
            D=self.num_quantum_ind, 
            method=self.config.ref_dir_method
        )
        
        # Assign each quantum individual to a reference direction
        num_dirs = self._ref_dirs.shape[0]
        self._ind_to_dir = np.array(
            [i % num_dirs for i in range(self.num_quantum_ind)], 
            dtype=int
        )

    # --- Methods Moved from QPopulationNetwork ---

    def _das_dennis(self, M: int, H: int) -> np.ndarray:
        """Generates uniformly spaced reference directions on a simplex. (Das-Dennis method)

        Args:
            M (int): Number of objectives.
            H (int): Number of divisions along each objective axis.

        Returns:
            np.ndarray: An array of reference direction vectors.
        """
        if H == 0:
            return np.array([[1.0]]) if M == 1 else np.empty((0, M), dtype=float)

        def gen_partitions(n_rem, k_rem):
            if k_rem == 1:
                yield (n_rem,)
                return
            for i in range(n_rem + 1):
                for p in gen_partitions(n_rem - i, k_rem - 1):
                    yield (i,) + p

        partitions = list(gen_partitions(H, M))
        return np.array(partitions, dtype=float) / float(H)

    def _make_ref_directions(self, M: int, D: int, method: str = "das-dennis") -> np.ndarray:
        """Creates reference direction vectors for MOEA/D.

        Tries to use the systematic Das-Dennis method first, falling back to a
        Dirichlet distribution if a suitable number of points cannot be generated.
        
        Explicitly prioritizes extreme points to ensure boundary coverage.
        """
        rng = np.random.default_rng(12345)
        
        # 1. Fallback / Random Method
        if method == 'dirichlet':
            if M > D:
                return rng.dirichlet(alpha=np.ones(M), size=D)
            extreme_dirs = np.eye(M, dtype=float)
            num_random_dirs = D - M
            if num_random_dirs > 0:
                random_dirs = rng.dirichlet(alpha=np.ones(M), size=num_random_dirs)
                all_dirs = np.vstack((extreme_dirs, random_dirs))
            else:
                all_dirs = extreme_dirs[:D]
            rng.shuffle(all_dirs)
            return all_dirs

        # 2. Das-Dennis Method
        for H in range(1, 10):
            dirs = self._das_dennis(M, H)
            
            # If we found a partition size that generates enough points...
            if dirs.shape[0] >= D:
                
                # Identify extreme points (those with 1.0 on any axis)
                is_extreme = np.any(dirs >= (1.0 - 1e-6), axis=1)
                
                extremes = dirs[is_extreme]
                others = dirs[~is_extreme]
                
                # Case A: We have more extremes than D (rare, but possible if D < M)
                if len(extremes) >= D:
                    # Just take the first D extremes (or shuffle them)
                    return extremes[:D]
                
                # Case B: We need all extremes + some intermediates
                selected = [extremes]
                needed = D - len(extremes)
                
                if needed > 0:
                    # Shuffle the 'others' to avoid bias (since Das-Dennis is ordered)
                    # We use the fixed seed rng defined above
                    rng.shuffle(others)
                    selected.append(others[:needed])
                    
                final_dirs = np.vstack(selected)
                return final_dirs
        return rng.dirichlet(alpha=np.ones(M), size=D)

    def _normalize_objectives_01(self, objs: np.ndarray) -> np.ndarray:
        """Normalizes objective values to a [0, 1] range, oriented for maximization.

        Minimization objectives are inverted (1 - normalized_value).

        Args:
            objs (np.ndarray): Raw objective values of shape (E, M).

        Returns:
            np.ndarray: Normalized values of shape (E, M), where higher is better.
        """
        E, M = objs.shape
        g = np.empty_like(objs, dtype=float)
        for m in range(M):
            col = objs[:, m].astype(float)
            lo, hi = np.min(col), np.max(col)
            if hi - lo < 1e-12:
                norm = np.zeros_like(col)
            else:
                norm = (col - lo) / (hi - lo)
            g[:, m] = (1.0 - norm) if (self.objective_sense[m] == "min") else norm
        return g

    def _score_weighted_sum(self, g: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Calculates a weighted sum score for each individual.

        Args:
            g (np.ndarray): Maximization-oriented normalized objectives (E, M).
            lam (np.ndarray): A weight vector (M,).

        Returns:
            np.ndarray: A score for each individual (E,).
        """
        return g.dot(lam)

    def _quantile_thresholds(self, g: np.ndarray) -> dict:
        """Calculates quantile-based thresholds for each objective.

        Args:
            g (np.ndarray): Maximization-oriented normalized objectives.
            q_low (float, optional): The lower quantile. Defaults to 0.3.
            q_high (float, optional): The upper quantile. Defaults to 0.90.

        Returns:
            dict: A dictionary with "min_ok" and "max_ok" threshold vectors.
        """
        min_ok = np.quantile(g, self.moead_q_low, axis=0)
        max_ok = np.quantile(g, self.moead_q_high, axis=0)
        return {"min_ok": min_ok, "max_ok": max_ok}

    def _mask_constraints(self, g: np.ndarray, cons: dict, hard: dict | None = None) -> np.ndarray:
        """Creates a boolean mask to filter individuals based on constraints.

        Args:
            g (np.ndarray): Maximization-oriented normalized objectives.
            cons (dict): Dictionary of constraints, e.g., from `_quantile_thresholds`.
            hard (dict | None, optional): Additional hard constraints. Defaults to None.

        Returns:
            np.ndarray: A boolean array indicating which individuals pass the filter.
        """
        E, M = g.shape
        mask = np.ones(E, dtype=bool)
        if cons and ("min_ok" in cons):
            min_ok = cons["min_ok"]
            if hard and ("min_idx" in hard):
                for m in hard["min_idx"]:
                    mask &= (g[:, m] >= float(min_ok[m]))
            else:
                for m in range(M):
                    mask &= (g[:, m] >= float(min_ok[m]))
        return mask

    def _select_topk_stratified(self, scores: np.ndarray, K: int) -> np.ndarray:
        """Selects K individuals using stratified sampling.

        It first selects the top P individuals (P > K), splits them into K
        groups, and picks the best from each group.

        Args:
            scores (np.ndarray): The scores for each individual.
            K (int): The final number of individuals to select.
            P_mult (int, optional): The multiplier to determine the initial pool size P.

        Returns:
            np.ndarray: The indices of the K selected individuals.
        """
        E = scores.shape[0]
        K = int(min(max(1, K), E))
        P = int(min(self.topP_mult * K, E))
        order = np.argsort(-scores)
        topP = order[:P]
        splits = np.array_split(topP, K)
        picks = [seg[0] for seg in splits if seg.size > 0]
        return np.array(picks[:K], dtype=int)

    def build_q_moead_topk_rows(self, pool_choices: np.ndarray, pool_objs: np.ndarray,
                                rows: np.ndarray, cols: np.ndarray, F: int, K: int) -> np.ndarray:
        """Builds target distributions using the MOEA/D-TopK strategy.

        For each quantum individual to be updated, it uses its assigned reference
        direction to select the best `K` individuals from the elite pool and
        builds a target distribution from them.

        Args:
            pool_choices (np.ndarray): Chromosomes of the elite pool (E_pool, L).
            pool_objs (np.ndarray): Objective values of the elite pool (E_pool, M).
            rows (np.ndarray): Indices of quantum individuals to update.
            cols (np.ndarray): Indices of genes to update.
            F (int): Number of functions.
            K (int): Number of elites to select.

        Returns:
            np.ndarray: Target distributions `q_rows` of shape (K_sel, F).
        """
        E_pool, L = pool_choices.shape
        K_sel = rows.size
        q_rows = np.zeros((K_sel, F), dtype=float)

        g = self._normalize_objectives_01(pool_objs)
        cons = self._quantile_thresholds(g)

        for r in range(K_sel):
            i = int(rows[r]); j = int(cols[r])
            lam = self._ref_dirs[self._ind_to_dir[i]]

            mask = self._mask_constraints(g, cons, hard=None)
            idx = np.where(mask)[0]
            if idx.size < K:
                idx = np.arange(E_pool)

            s = self._score_weighted_sum(g[idx, :], lam)
            pick_rel = self._select_topk_stratified(s, K=K)
            pick = idx[pick_rel]

            ops = pool_choices[pick, j].astype(int)
            counts = np.bincount(ops, minlength=F).astype(float)
            ssum = counts.sum()
            q_rows[r, :] = counts / ssum if ssum > 0 else (1.0 / F)
        return q_rows

    def get_directions_for_logging(self) -> dict:
        """Returns a dictionary of quantum individuals and their directions for logging."""
        log_data = {}
        if self._ref_dirs is not None and self._ind_to_dir is not None:
            for i in range(self.num_quantum_ind):
                dir_index = self._ind_to_dir[i]
                direction_vector = self._ref_dirs[dir_index]
                direction_str = ", ".join([f"{val:.3f}" for val in direction_vector])
                log_data[i] = f"Direction -> [{direction_str}]"
        return log_data