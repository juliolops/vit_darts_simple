""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Quantum population classes.

    Refactored QPopulationNetwork with new Elite Method for the Update of the Quantum Population
    - Added new crossover methods: HUX and Uniform.
    - Improved mutation strategies for better exploration of the search space.
    - Added Metrics to track the probabilities evolution. 
    Diego Páez Ardila - 2025
"""
import os
import csv
import numpy as np

from .chromosome import QChromosomeParams, QChromosomeNetwork
from .helpers.moea_helper import MOEADHelper
from .helpers.configs import NetworkRulesConfig, EliteUpdateConfig, MOEAConfig
from .helpers.metrics_logger import QPopulationMetricsLogger
from .helpers.rules import NetworkArchitectureRules
from .helpers.update_strategies import create_update_strategy

class QPopulation(object):
    """ QNAS Population to be evolved. """

    def __init__(self, num_quantum_ind, repetition, update_quantum_rate):
        """ Initialize QPopulation.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            update_quantum_rate: (float) probability that a quantum gene will be updated.
        """

        self.dtype = np.float64  # Type of quantum population arrays.

        self.chromosome = None
        self.current_pop = None
        self.current_pop_objs = None
        self.num_ind = num_quantum_ind

        self.repetition = repetition
        self.update_quantum_rate = update_quantum_rate

    def initialize_qpop(self):
        raise NotImplementedError('initialize_qpop() must be implemented in sub classes')

    def generate_classical(self):
        raise NotImplementedError('generate_classical() must be implemented in sub classes')

    def update_quantum(self, intensity):
        raise NotImplementedError('update_quantum() must be implemented in sub classes')

class QPopulationParams(QPopulation):
    """QNAS Chromosomes for the hyperparameters to be evolved."""

    def __init__(self, num_quantum_ind, params_ranges, repetition, crossover_rate, update_quantum_rate):
        """
        Args:
            num_quantum_ind: (int) number of quantum individuals.
            params_ranges: dict[str, list[float]] mapping parameter_name -> [lower, upper].
            repetition: (int) ratio between classic and quantum pop sizes.
            crossover_rate: (float or None) arithmetic crossover rate in [0, 1].
            update_quantum_rate: (float) probability a quantum gene will be updated.
        """
        super(QPopulationParams, self).__init__(num_quantum_ind, repetition, update_quantum_rate)

        self.tolerance = 1.0e-15
        self.lower = None
        self.upper = None

        if crossover_rate is None:
            self.crossover = 0.0
        else:
            self.crossover = float(crossover_rate)

        self.chromosome = QChromosomeParams(params_ranges, self.dtype)
        self.initial_lower, self.initial_upper = self.chromosome.initialize_qgenes()
        self.initialize_qpop()

    def initialize_qpop(self):
        """Initialize quantum population with self.num_ind individuals."""
        self.lower = np.tile(self.initial_lower, (self.num_ind, 1))
        self.upper = np.tile(self.initial_upper, (self.num_ind, 1))

    def classic_crossover(self, new_pop, distance):
        """
        Arithmetic crossover between the previous classic population (self.current_pop)
        and the newly sampled one (new_pop). Works even if row counts differ.
        """
        if self.current_pop is None or self.current_pop.size == 0 or self.crossover == 0.0:
            return new_pop

        Nc, G = new_pop.shape
        if G != self.chromosome.num_genes:
            raise RuntimeError(f"classic_crossover: gene count mismatch (new_pop {G} vs expected {self.chromosome.num_genes}).")

        num_curr, g2 = self.current_pop.shape
        if g2 != G:
            raise RuntimeError(f"classic_crossover: gene count mismatch (current_pop {g2} vs new_pop {G}).")

        donor_idx = np.random.randint(0, num_curr, size=Nc)
        donors = self.current_pop[donor_idx, :]

        mask = (np.random.rand(Nc, G) <= self.crossover)
        delta = (donors - new_pop) * float(distance)
        new_pop[mask] = new_pop[mask] + delta[mask]
        return new_pop

    def generate_classical(self):
        """
        Generate (num_ind * repetition) classical individuals by sampling uniformly
        within the per-gene quantum intervals [lower, upper].
        """
        G = self.chromosome.num_genes
        Nq = self.num_ind
        R = self.repetition

        rnd = np.random.rand(Nq * R, G).astype(self.dtype)
        span = (self.upper - self.lower)
        new_pop = rnd * np.tile(span, (R, 1)) + np.tile(self.lower, (R, 1))
        return new_pop

    def update_quantum(self, intensity):
        """
        Update quantum intervals self.lower and self.upper using exemplars drawn
        from self.current_pop, robust to any current_pop row count.
        """
        if self.current_pop is None or self.current_pop.size == 0:
            raise RuntimeError("update_quantum: self.current_pop must be set and non-empty before updating.")

        intensity = float(intensity)

        num_current_ind, num_genes = self.current_pop.shape
        if num_genes != self.chromosome.num_genes:
            raise RuntimeError(
                f"update_quantum: gene count mismatch (current_pop has {num_genes}, "
                f"expected {self.chromosome.num_genes})."
            )

        # Target shapes
        Nq, G = self.num_ind, self.chromosome.num_genes

        # 1) Build update mask in TARGET SHAPE: (Nq, G)
        rnd = np.random.rand(Nq, G)
        mask = np.where(rnd <= self.update_quantum_rate)
        rows, _ = mask
        if rows.size == 0:
            return

        # 2) Pick donor row for each quantum row (with replacement)
        donor_idx = np.random.randint(0, num_current_ind, size=Nq)
        donors = self.current_pop[donor_idx, :]  # shape (Nq, G)

        # 3) Compute non-negative per-gene spread across ALL current_pop
        max_genes = np.max(self.current_pop, axis=0)       # (G,)
        min_genes = np.min(self.current_pop, axis=0)       # (G,)
        spread = np.maximum(max_genes - min_genes, 0.0)    # guard tiny negatives
        diff = np.broadcast_to(spread[None, :], (Nq, G))   # (Nq, G)

        # 4) Apply updates using donors in place of current_pop
        # lower update
        upd = donors[mask] - self.lower[mask] - (diff[mask] / 2.0)
        self.lower[mask] += intensity * upd

        # upper update
        upd = donors[mask] - self.upper[mask] + (diff[mask] / 2.0)
        self.upper[mask] += intensity * upd

        # 5) Truncate to initial bounds (broadcast-safe, no 2D->1D mask indexing)
        init_lower = self.initial_lower[None, :]  # (1, G) -> broadcasts to (Nq, G)
        init_upper = self.initial_upper[None, :]

        # Clamp with tolerance using vectorized where (avoids boolean advanced indexing on 1D arrays)
        self.lower = np.where(self.lower < (init_lower - self.tolerance), init_lower, self.lower)
        self.upper = np.where(self.upper > (init_upper + self.tolerance), init_upper, self.upper)

        # 6) Safe swap if any lower > upper (rare but makes it bulletproof)
        swap_mask = self.lower > self.upper
        if np.any(swap_mask):
            tmp = self.lower.copy()
            self.lower[swap_mask] = self.upper[swap_mask]
            self.upper[swap_mask] = tmp[swap_mask]

class QPopulationNetwork(QPopulation):
    """QNAS Chromosomes for the networks to be evolved."""
    def __init__(self, num_quantum_ind: int, max_num_nodes: int, repetition: int,
                fn_list: list, initial_probs: list,
                rules_config: NetworkRulesConfig,
                elite_config: EliteUpdateConfig,
                quantum_update_config: dict,
                moea_config: MOEAConfig = None,
                experiment_path: str = ""):
        """Initializes the QPopulationNetwork.
        
        Args:
            num_quantum_ind (int): Number of quantum individuals.
            max_num_nodes (int): Maximum number of nodes (genes) in a network.
            repetition (int): Ratio of classical to quantum individuals.
            update_quantum_rate (float): Base probability for updating a quantum gene.
            fn_list (list): List of possible functions (operations) for each node.
            initial_probs (list): Defines initial probabilities for each function.
            rules_config (NetworkRulesConfig): Configuration for NAS architectural rules.
            elite_config (EliteUpdateConfig): Configuration for the elite update strategy.
            quantum_update_config (dict): The 'quantum_update' block from the YAML.
            moea_config (MOEAConfig): Configuration for MOEA/D parameters.
            experiment_path (str, optional): Path for saving metrics. Defaults to "".
        """
        super(QPopulationNetwork, self).__init__(num_quantum_ind, repetition,
                                                update_quantum_rate=quantum_update_config.get('static_rate', 0.1))
        
        # --- Store Config Objects ---
        self.rules_config = rules_config
        self.elite_config = elite_config
        self.moea_config = moea_config
        self.update_config = quantum_update_config

        self.probabilities = None
        self.chromosome = QChromosomeNetwork(max_num_nodes, fn_list, self.dtype)

        # --- Load NEW Update Schedule Params from Config ---
        self.rate_schedule_type = self.update_config.get('quantum_rate_schedule', 'cosine')
        self.max_update_schedule_type = self.update_config.get('max_update_schedule', 'cosine')
        
        self.static_update_rate = float(self.update_config.get('static_quantum_rate', 0.08))
        self.static_max_update = float(self.update_config.get('static_max_update', 0.02))
        
        self.intensity_min = self.update_config.get('intensity_min', 0.5)
        self.intensity_max = self.update_config.get('intensity_max', 1.0)
        
        rate_params = self.update_config.get('quantum_rate_schedule_params', {})
        self.rate_sched_start = rate_params.get('start', 0.08)
        self.rate_sched_end = rate_params.get('end', 0.002)
        
        max_update_params = self.update_config.get('max_update_schedule_params', {})
        self.max_up_sched_start = max_update_params.get('start', 0.3)
        self.max_up_sched_end = max_update_params.get('end', 0.16)

        # --- Load NEW Prob Cap Params from Config ---
        self.max_prob = self.update_config.get('max_prob_cap', 0.90)
        min_prob_factor = self.update_config.get('min_prob_factor', 0.01)
        self.min_prob = max(1e-8, min_prob_factor / self.chromosome.num_functions)

        # --- Elite Update Params ---
        self.elite_mode = self.elite_config.elite_mode
        self.k_elites = self.elite_config.k_elites
        self.pool_factor = self.elite_config.pool_factor
        self.ema_beta = self.elite_config.ema_beta
        self.rank_weighting = self.elite_config.rank_weighting
        self._U_total = None
        
        self.update_strategy = create_update_strategy(
            config=self.elite_config,
            chromosome=self.chromosome
        )

        # --- Update-rate schedule ---
        self.rate_start = max(0.2, self.update_quantum_rate)
        self.rate_boost = 0.40
        self.rate_end = self.update_quantum_rate

        # --- Metrics ---
        self.metrics_output = os.path.join(experiment_path, "qpop_update", "metrics_output.csv")
        os.makedirs(os.path.dirname(self.metrics_output), exist_ok=True)
        # Create the logger instance
        self.logger = QPopulationMetricsLogger(self.metrics_output)

        # Probs ---
        self.initial_probs = self.chromosome.initialize_qgenes(initial_probs=initial_probs)
    
        # --- Objective/MOEA Params ---
        self.moea_helper = None # Will be initialized by MOQNAS
        self.objective_names = None
        self.objective_sense = None
        self.num_objectives = None

        # --- Network Rules Params ---
        self.fn_list = list(fn_list)
        self.rules = NetworkArchitectureRules(
            rules_config=self.rules_config,
            fn_list=self.fn_list,
            chromosome=self.chromosome
        )

        self._parent_map = None
        
        self.initialize_qpop()

    def set_schedule_total_updates(self, total_updates: int):
        """Sets the total number of updates for scheduling learning rates.

        Args:
            total_updates (int): The total expected number of calls to `update_quantum`.
        """
        self._U_total = max(1, int(total_updates))

    def initialize_qpop(self):
        """Initializes the quantum population probabilities.

        Creates the `self.probabilities` array with shape
        (num_ind, num_genes, num_functions) and applies the initial static
        mask for the terminal operation if enabled.
        """
        self.probabilities = np.tile(self.initial_probs, (self.num_ind, self.chromosome.num_genes, 1))
        if self.rules.enforce_noop_in_update:
            self.probabilities = self.rules.enforce_static_mask(self.probabilities)

    def generate_classical(self) -> np.ndarray:
            """Generates a classical population from quantum probabilities.
            
            Delegates the work to the NetworkArchitectureRules class
            to enforce all structural constraints.
            """
            N = self.num_ind * self.repetition
            base_prob = np.tile(self.probabilities, (self.repetition, 1, 1))
            self._parent_map = np.tile(np.arange(self.num_ind, dtype=int), self.repetition)

            # Delegate the complex sampling and rule enforcement
            new_pop = self.rules.generate_classical_architectures(base_prob, N)
            
            return new_pop

    def set_objective_directions(self, names: list, sense: list, moea_config: MOEAConfig):
        """
        Initializes the MOEADHelper for multi-objective optimization.
        This method "bolts on" the MOO capabilities to the population.
        """
        if moea_config is None:
            raise ValueError("moea_config must be provided to set_objective_directions")
            
        self.objective_names = list(names)
        self.objective_sense = [("min" if s is None else str(s).lower())
                                for s in (sense or ["min"] * len(names))]
        self.num_objectives = len(self.objective_names)

        # Create and store the MOEAD helper
        self.moea_helper = MOEADHelper(
            num_objectives=self.num_objectives,
            objective_sense=self.objective_sense,
            num_quantum_ind=self.num_ind,
            moea_config=moea_config
        )

    def _sample_intensity(self, lo: float = 0.5, hi: float = 1.0) -> float:
        """Samples an update intensity from a Beta distribution.

        Args:
            lo (float, optional): The minimum intensity. Defaults to 0.5.
            hi (float, optional): The maximum intensity. Defaults to 1.0.

        Returns:
            float: A sampled intensity value.
        """
        u = np.random.beta(2.0, 5.0)
        return lo + (hi - lo) * u

    def _suggest_max_update(self) -> float:
        """Suggests a base `max_update` value based on the number of functions.

        Returns:
            float: The suggested `max_update` value.
        """
        F = self.chromosome.num_functions
        if F <= 12: return 0.07
        if F <= 32: return 0.05
        return 0.04
    
    def _cosine_schedule_cyclic(self, t: float, T: float, start: float, end: float) -> float:
        """Calculates a value based on a full cosine cycle schedule.

        The value oscillates between 'start' and 'end', completing one
        full cycle every 'T' steps.

        Args:
            t (float): Current step. Can be larger than T.
            T (float): The period, or duration, of one full cycle.
            start (float): The value at the beginning and end of the cycle
                        (at t=0, t=T, t=2T, etc.).
            end (float): The value at the midpoint of the cycle
                    (at t=T/2, t=3T/2, etc.).

        Returns:
            float: The scheduled, oscillating value.
        """
        t = t % float(T)
        w = 0.5 * (1.0 + np.cos(2.0 * np.pi * t / float(T)))
        return end + (start - end) * w

    def _cosine_annealing_schedule(self, t: float, T: float, start: float, end: float) -> float:
        """Calculates a value based on a cosine annealing schedule.

        Args:
            t (float): Current step.
            T (float): Total steps.
            start (float): Start value.
            end (float): End value.

        Returns:
            float: The scheduled value.
        """
        t = max(0, min(int(t), int(T)))
        w = 0.5 * (1.0 + np.cos(np.pi * t / float(T)))
        return end + (start - end) * w

    def _get_rate_schedule_value(self, u: int, U_total: int) -> float:
        """Gets the quantum rate based on the config strategy."""
        if self.rate_schedule_type == 'static':
            return self.static_update_rate
        
        elif self.rate_schedule_type == 'cosine':
            return self._cosine_schedule_cyclic(
                u,
                U_total,
                self.rate_sched_start,
                self.rate_sched_end
            )
        else:
            raise ValueError(f"Unknown rate_schedule: {self.rate_schedule_type}")

    def _get_max_update_schedule_value(self, u: int, U_total: int) -> float:
        """Gets the max update value based on the config strategy."""
        if self.max_update_schedule_type == 'static':
            return self.static_max_update
            
        elif self.max_update_schedule_type == 'cosine':
            base = self._suggest_max_update()
            mult = self._cosine_annealing_schedule(
                u,
                U_total,
                self.max_up_sched_start,
                self.max_up_sched_end
            )
            return base * mult
        else:
            raise ValueError(f"Unknown max_update_schedule: {self.max_update_schedule_type}")


    def _update(self, chromosomes: np.ndarray, idx: np.ndarray, update_value: np.ndarray) -> np.ndarray:
        """Applies the quantum update rule to a batch of probability vectors.

        It increases the probability of the 'winner' gene (at `idx`) and
        proportionally decreases the probabilities of the others. It ensures
        probabilities stay within the [min_prob, max_prob] bounds and renormalizes.

        Args:
            chromosomes (np.ndarray): A batch of probability vectors to update.
            idx (np.ndarray): The indices of the winner gene for each vector.
            update_value (np.ndarray): The amount to add to the winner's probability.

        Returns:
            np.ndarray: The updated and normalized probability vectors.
        """
        idx0 = np.arange(chromosomes.shape[0])
        current = chromosomes[idx0, idx]
        headroom = np.maximum(self.max_prob - current, 0.0)
        update_array = np.minimum(update_value, headroom)
        sum_values = current + update_array

        chromosomes[idx0, idx] = 0.0
        totals = np.sum(chromosomes, axis=1)
        totals = np.where(totals == 0, 1e-8, totals)
        decrease = (update_array / totals).reshape(-1, 1) * chromosomes
        chromosomes -= decrease
        chromosomes[idx0, idx] = sum_values

        chromosomes = np.maximum(chromosomes, self.min_prob)
        chromosomes /= np.sum(chromosomes, axis=1, keepdims=True)
        return chromosomes

    def update_quantum(self, intensity: float | None = None, current_gen: int | None = None):
            """Performs the main quantum population update.

            This method orchestrates the entire update process:
            1. Schedules the learning rate (`update_quantum_rate`) and max update value
            based on strategies defined in the config.
            2. Selects a random subset of (individual, gene) pairs to update.
            3. Based on `self.elite_mode`, constructs a target probability
            distribution `q_rows` for each selected gene.
            4. Calculates the update step size based on the target distribution's confidence.
            5. Calls the `_update` method to apply the changes.
            6. Enforces global constraints and logs metrics.

            Args:
                intensity (float | None, optional): A factor to scale the update step size.
                                                If None, it's sampled randomly. Defaults to None.
                current_gen (int | None, optional): The current generation/epoch index, used
                                                for scheduling and logging. Defaults to None.
            """
            u = self.logger._update_counter
            U_total = getattr(self, "_U_total", None)
            if U_total is None:
                U_total = max(1, (current_gen or 0) // 1)
            
            # 1. Set update_quantum_rate from config-driven schedule
            self.update_quantum_rate = self._get_rate_schedule_value(u, U_total)

            # 2. Sample intensity from config-driven range
            if intensity is None:
                intensity = self._sample_intensity(lo=self.intensity_min, hi=self.intensity_max)

            # 3. Set max_update from config-driven schedule
            self.max_update = self._get_max_update_schedule_value(u, U_total)

            # 4. Calculate final step size
            eta_base = float(intensity) * float(self.max_update)

            F = int(self.chromosome.num_functions)

            rand = np.random.rand(self.num_ind, self.chromosome.num_genes)
            rows, cols = np.where(rand <= self.update_quantum_rate)
            if rows.size == 0:
                return

            if self.elite_mode == "old":
                E = min(self.num_ind, self.current_pop.shape[0])
                best_classic = self.current_pop[:E]
                winners = best_classic[rows, cols]
                self.probabilities[rows, cols, :] = self._update(
                    self.probabilities[rows, cols, :], winners, eta_base)
                
                self.logger.log_update(self.probabilities, self.update_quantum_rate, 
                                    self.max_update, epoch_idx=current_gen)
                self.logger.save_metrics_csv()
                return

            P_sel = self.probabilities[rows, cols, :].astype(float, copy=True)
            P_sel = self.rules.apply_probability_caps(P_sel, rows, cols)

            q_rows = self.update_strategy.build_target_distributions(
                self.current_pop, self.current_pop_objs, rows, cols, self.moea_helper
            )

            q_rows = self.rules.mask_noop_in_targets(q_rows, rows, cols, prior_P=P_sel)
            
            random_noise = np.random.uniform(0, 1e-6, size=q_rows.shape)
            winners = np.argmax(q_rows + random_noise, axis=1)

            consensus = q_rows[np.arange(q_rows.shape[0]), winners]
            bump = eta_base * np.maximum(consensus, 1e-8)

            updated = self._update(P_sel, winners, bump)
            self.probabilities[rows, cols, :] = updated

            if self.rules.enforce_noop_in_update:
                self.probabilities = self.rules.enforce_static_mask(self.probabilities)

            self.logger.log_update(self.probabilities, self.update_quantum_rate, 
                                self.max_update, epoch_idx=current_gen)
            self.logger.save_metrics_csv()