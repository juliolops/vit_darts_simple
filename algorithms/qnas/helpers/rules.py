# In algorithms/qnas/rules.py

import numpy as np
from .configs import NetworkRulesConfig

class NetworkArchitectureRules:
    """
    Handles all NAS-specific architectural rules and constraints.
    
    This class enforces rules like no-op handling, pooling restrictions,
    and minimum lengths, both during generation (sampling) and
    during the quantum update (applying caps).
    """

    def __init__(self, rules_config: NetworkRulesConfig, fn_list: list, 
                chromosome):
        """
        Args:
            rules_config (NetworkRulesConfig): The configuration dataclass.
            fn_list (list): The list of function/operation names.
            chromosome: The QChromosomeNetwork object (for num_genes).
        """
        self.config = rules_config
        self.chromosome = chromosome
        self.fn_list = list(fn_list)

        self.terminal_op_name = self.config.terminal_op_name
        self.min_active_len = int(self.config.min_active_len)
        self.truncate_after_noop = bool(self.config.truncate_after_noop)
        self.avoid_consecutive_pool = bool(self.config.avoid_consecutive_pool)
        self.enforce_noop_in_update = bool(self.config.enforce_noop_in_update)
        self.noop_max_prob = float(self.config.noop_max_prob)
        self.noop_ramp_cap = bool(self.config.noop_ramp_cap)

        try:
            self.no_op_id = int(self.fn_list.index(self.terminal_op_name))
        except ValueError:
            self.no_op_id = None

        patterns = self.config.pool_op_name
        if not isinstance(patterns, (list, tuple, set)):
            patterns = [str(patterns)]
        self.pool_ids = [i for i, name in enumerate(self.fn_list)
                        if any(pat in str(name) for pat in patterns)]
    
    def _noop_cap_for_gene(self, j: int) -> float:
        """Calculates the maximum allowed probability for the terminal op at a given gene.

        Args:
            j (int): The index of the gene (node).

        Returns:
            float: The probability cap, ranging from 0.0 to `self.noop_max_prob`.
                    Returns 0.0 for genes before `min_active_len`.
        """
        if self.no_op_id is None:
            return 1.0
        if j < self.min_active_len:
            return 0.0
        if not self.noop_ramp_cap:
            return self.noop_max_prob
        
        L = self.chromosome.num_genes
        maxcap = self.noop_max_prob
        if L <= self.min_active_len:
            return maxcap
        
        # Linear ramp from 0 to maxcap between min_active_len and L-1
        alpha = (j - self.min_active_len + 1) / float(max(1, L - self.min_active_len))
        return float(min(maxcap, max(0.0, alpha * maxcap)))

    def apply_probability_caps(self, P_rows: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Applies terminal op caps to selected rows of a probability matrix.

        This is used before the quantum update to enforce architectural constraints
        on the current probabilities.

        Args:
            P_rows (np.ndarray): Probability vectors to be modified.
            rows (np.ndarray): Indices of the quantum individuals.
            cols (np.ndarray): Indices of the genes corresponding to the rows.

        Returns:
            np.ndarray: The modified and renormalized probability vectors.
        """
        if (self.no_op_id is None) or (not self.enforce_noop_in_update):
            return P_rows
        
        P = P_rows.copy()
        F = P.shape[1]
        eps = 1e-12

        for i in range(P.shape[0]):
            j = int(cols[i])
            cap = self._noop_cap_for_gene(j)
            
            if cap <= 0.0:
                P[i, self.no_op_id] = 0.0
            else:
                P[i, self.no_op_id] = min(P[i, self.no_op_id], cap)
            
            s = P[i].sum()
            if s <= eps:
                P[i].fill(1.0 / F)
            else:
                P[i] /= s
        return P

    def mask_noop_in_targets(self, q_rows: np.ndarray, rows: np.ndarray, cols: np.ndarray, 
                            prior_P: np.ndarray) -> np.ndarray:
        """Masks the terminal op in target distribution rows (q_rows).

        Ensures that the "winner" operation in the target distribution `q` cannot
        be the terminal op if it is forbidden at that position.

        Args:
            q_rows (np.ndarray): Target probability distributions.
            rows (np.ndarray): Indices of the quantum individuals.
            cols (np.ndarray): Indices of the genes.
            prior_P (np.ndarray): The original probability distributions, used as a fallback.

        Returns:
            np.ndarray: The modified and renormalized target distributions.
        """
        if (self.no_op_id is None) or (not self.enforce_noop_in_update):
            return q_rows
        
        Q = q_rows.copy()
        eps = 1e-12
        for i in range(Q.shape[0]):
            j = int(cols[i])
            cap = self._noop_cap_for_gene(j)
            
            if cap <= 0.0:
                Q[i, self.no_op_id] = 0.0
            
            s = Q[i].sum()
            if s <= eps:
                # Fallback to prior probability if target becomes all-zero
                row = prior_P[i]
                row_sum = row.sum()
                if row_sum > eps:
                    Q[i] = row / row_sum
                else:
                    Q[i].fill(1.0 / Q.shape[1])
            else:
                Q[i] /= s
        return Q

    def enforce_static_mask(self, P: np.ndarray) -> np.ndarray:
            """
            Globally enforces that the terminal op probability is zero before min_active_len.
            
            This method is based on the original, robust implementation.
            """
            # If no rules to apply, return the original array
            if self.no_op_id is None or (self.min_active_len <= 0):
                return P

            # Create a copy to avoid changing the original array
            P_new = P.copy()
            
            # 1. Set the no_op probability to zero in the restricted slice
            P_new[:, :self.min_active_len, self.no_op_id] = 0.0
            
            # 2. Get the sums of the modified slices
            sums = P_new[:, :self.min_active_len, :].sum(axis=-1, keepdims=True)

            # 3. Replace any 0.0 sums with a tiny value to avoid division by zero.
            sums = np.where(sums <= 1e-12, 1e-12, sums)
            
            # 4. Re-normalize the slice.
            P_new[:, :self.min_active_len, :] = P_new[:, :self.min_active_len, :] / sums
            
            return P_new

    def _renorm_or_fallback(self, p, base=None):
        """Safely renormalizes a probability vector, with fallbacks."""
        s = p.sum()
        if s > 1e-8:
            return p / s
        if base is not None:
            sb = base.sum()
            if sb > 1e-8:
                return base / sb
        # Final fallback to uniform
        F = len(p)
        return np.full_like(p, 1.0 / F if F > 0 else 0.0, dtype=float)

    def generate_classical_architectures(self, base_prob: np.ndarray, N: int) -> np.ndarray:
        """
        Generates a classical population from quantum probabilities, enforcing rules.
        (This is the core logic from generate_classical)
        """
        F = self.chromosome.num_functions
        L = self.chromosome.num_genes
        
        new_pop = np.zeros((N, L), dtype=np.int32)
        
        for ind in range(N):
            prev_was_pool = False
            truncated = False
            
            for node in range(L):
                if truncated:
                    new_pop[ind, node] = self.no_op_id if self.no_op_id is not None else 0
                    continue

                p = base_prob[ind, node, :].astype(float, copy=True)

                # Rule: No 'no_op' before min_active_len
                if (self.no_op_id is not None) and (node < self.min_active_len):
                    p[self.no_op_id] = 0.0

                # Rule: No consecutive pool operations
                if self.avoid_consecutive_pool and prev_was_pool and self.pool_ids:
                    p[self.pool_ids] = 0.0

                p = self._renorm_or_fallback(p, base=p)
                
                if p.sum() == 0: # Fallback in case of all-zero mask
                    choice = 0 # Or some default
                else:
                    choice = np.random.choice(F, p=p)
                    
                new_pop[ind, node] = choice
                
                prev_was_pool = (choice in self.pool_ids)

                # Rule: Truncate after first 'no_op'
                if (self.truncate_after_noop and
                        (self.no_op_id is not None) and
                        (choice == self.no_op_id) and
                        (node >= self.min_active_len - 1)): # -1 because it's allowed *at* min_active_len
                    
                    if node + 1 < L:
                        new_pop[ind, node + 1: L] = self.no_op_id
                    truncated = True
                    
        return new_pop