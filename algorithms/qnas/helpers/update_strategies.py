# In algorithms/qnas/update_strategies.py

import numpy as np
from abc import ABC, abstractmethod

# Import the helpers and configs we've been creating
from .configs import EliteUpdateConfig
from .moea_helper import MOEADHelper 


class EliteUpdateStrategy(ABC):
    """
    Abstract base class for all quantum elite update strategies.
    Defines the interface for building a target distribution.
    """
    def __init__(self, elite_config: EliteUpdateConfig, chromosome, **kwargs):
        self.config = elite_config
        self.chromosome = chromosome
        self.k_elites = self.config.k_elites
        self.rank_weighting = self.config.rank_weighting
        self.ema_beta = self.config.ema_beta
        self._q_ema = None # For GlobalK EMA

    @abstractmethod
    def build_target_distributions(self, current_pop: np.ndarray, 
                                current_pop_objs: np.ndarray | None,
                                rows: np.ndarray, cols: np.ndarray,
                                moea_helper: MOEADHelper | None) -> np.ndarray:
        """
        Builds the target probability distributions (q_rows) for the
        (individual, gene) pairs specified by rows and cols.

        Args:
            current_pop (np.ndarray): The elite classical population.
            current_pop_objs (np.ndarray | None): Objectives of the elite pop.
            rows (np.ndarray): The quantum individual indices to update.
            cols (np.ndarray): The gene indices to update.
            moea_helper (MOEADHelper | None): The MOEA/D helper, if available.

        Returns:
            np.ndarray: The target distributions 'q_rows' of shape (len(rows), F).
        """
        pass

    def _elite_weights(self, E: int) -> np.ndarray:
        """Generates weights for elites, optionally based on rank.

        Args:
            E (int): The number of elites.

        Returns:
            np.ndarray: A normalized weight vector of size E.
        """
        if not self.rank_weighting:
            return np.ones(E, dtype=float) / max(E, 1)
        ranks = np.arange(E, dtype=float) + 1.0
        w = 1.0 / ranks
        return w / w.sum()


class SingleStrategy(EliteUpdateStrategy):
    """
    'single' elite_mode: The target distribution is a one-hot vector
    pointing to the choice of the corresponding elite individual.
    """
    def build_target_distributions(self, current_pop: np.ndarray, 
                                current_pop_objs: np.ndarray | None,
                                rows: np.ndarray, cols: np.ndarray,
                                moea_helper: MOEADHelper | None) -> np.ndarray:
        
        F = self.chromosome.num_functions
        E = min(rows.size, current_pop.shape[0]) # Match num individuals to update
        best_classic = current_pop[:E]
        
        # This mode assumes a 1-to-1 mapping of q-ind to elite-ind
        winners_single = best_classic[rows, cols]
        
        q_rows = np.zeros((rows.size, F), dtype=float)
        q_rows[np.arange(rows.size), winners_single] = 1.0
        return q_rows


class GlobalKStrategy(EliteUpdateStrategy):
    """
    'global_k' elite_mode: The target is a single global distribution
    built from a weighted histogram of the Top-K elites.
    """
    def _build_q_global(self, elites_choices: np.ndarray, F: int) -> np.ndarray:
        """Builds a global target distribution `q` from elite individuals."""
        E, L = elites_choices.shape
        w = self._elite_weights(E)
        counts = np.zeros((L, F), dtype=float)
        for e in range(E):
            counts[np.arange(L), elites_choices[e]] += w[e]
        
        q = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)

        if self.ema_beta and self.ema_beta > 0.0:
            if self._q_ema is None or self._q_ema.shape != q.shape:
                self._q_ema = q.copy()
            self._q_ema = self.ema_beta * self._q_ema + (1 - self.ema_beta) * q
            q = self._q_ema
        return q

    def build_target_distributions(self, current_pop: np.ndarray, 
                                current_pop_objs: np.ndarray | None,
                                rows: np.ndarray, cols: np.ndarray,
                                moea_helper: MOEADHelper | None) -> np.ndarray:
        
        F = self.chromosome.num_functions
        E = min(self.k_elites, current_pop.shape[0])
        topk = current_pop[:E]
        
        q_full = self._build_q_global(topk, F)
        q_rows = q_full[cols, :] # All individuals get the same target for gene 'j'
        return q_rows


class BootstrapKStrategy(EliteUpdateStrategy):
    """
    'bootstrap_k' elite_mode: Target is built by sampling (with replacement)
    K elites from a larger pool for each *specific* (ind, gene) pair.
    """
    def _build_q_bootstrap_rows(self, pool_choices: np.ndarray, rows: np.ndarray,
                                cols: np.ndarray, F: int, k: int,
                                weights: np.ndarray = None,
                                rng: np.random.Generator = None) -> np.ndarray:
        """Builds target distributions by bootstrapping from an elite pool."""
        if rng is None:
            rng = np.random.default_rng()

        E_pool, L = pool_choices.shape
        K_sel = rows.size
        q_rows = np.zeros((K_sel, F), dtype=float)
        if weights is None:
            weights = np.ones(E_pool, dtype=float) / max(E_pool, 1)

        for i in range(K_sel):
            c = cols[i]
            sel = rng.choice(E_pool, size=k, replace=True, p=weights)
            ops = pool_choices[sel, c]
            for op in ops:
                q_rows[i, op] += 1.0
            
            s = q_rows[i].sum()
            q_rows[i] = (q_rows[i] / s) if s > 0 else (1.0 / F)
        return q_rows

    def build_target_distributions(self, current_pop: np.ndarray, 
                                current_pop_objs: np.ndarray | None,
                                rows: np.ndarray, cols: np.ndarray,
                                moea_helper: MOEADHelper | None) -> np.ndarray:
        
        F = self.chromosome.num_functions
        pool_factor = self.config.pool_factor
        
        E_pool = min(current_pop.shape[0], max(self.k_elites, pool_factor * self.k_elites))
        pool = current_pop[:E_pool]
        weights = self._elite_weights(E_pool)
        
        q_rows = self._build_q_bootstrap_rows(pool, rows, cols, F, k=self.k_elites, weights=weights)
        return q_rows


class MOEADTopKStrategy(EliteUpdateStrategy):
    """
    'moead_topk' elite_mode: Target is built using the MOEA/D helper
    to find the Top-K individuals *relative to that individual's
    assigned reference direction*.
    """
    def build_target_distributions(self, current_pop: np.ndarray, 
                                current_pop_objs: np.ndarray | None,
                                rows: np.ndarray, cols: np.ndarray,
                                moea_helper: MOEADHelper | None) -> np.ndarray:

        if moea_helper is None:
            raise RuntimeError(
                "elite_mode='moead_topk' requires 'set_objective_directions' "
                "to be called first, but moea_helper is None."
            )
        if current_pop_objs is None:
            raise RuntimeError(
                "elite_mode='moead_topk' requires current_pop_objs, but it is None."
            )

        F = self.chromosome.num_functions
        E_pool = current_pop.shape[0]
        pool_choices = current_pop[:E_pool]
        pool_objs = current_pop_objs[:E_pool, :]
        
        q_rows = moea_helper.build_q_moead_topk_rows(
            pool_choices=pool_choices, pool_objs=pool_objs,
            rows=rows, cols=cols, F=F, K=self.k_elites
        )
        return q_rows


def create_update_strategy(config: EliteUpdateConfig, chromosome, **kwargs) -> EliteUpdateStrategy:
    """
    Factory function to create the correct update strategy.
    """
    mode = config.elite_mode
    if mode == "single":
        return SingleStrategy(config, chromosome, **kwargs)
    if mode == "global_k":
        return GlobalKStrategy(config, chromosome, **kwargs)
    if mode == "bootstrap_k":
        return BootstrapKStrategy(config, chromosome, **kwargs)
    if mode == "moead_topk":
        return MOEADTopKStrategy(config, chromosome, **kwargs)
    if mode == "old":
        return None # 'old' mode is handled separately
    
    raise ValueError(f"Unknown elite_mode: {mode}")