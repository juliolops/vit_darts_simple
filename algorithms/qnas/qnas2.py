""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Refactored Q-NAS algorithm class, with modular replace & crossover.
        Diego Páez Ardila - 2025
"""

import datetime
import os
from pickle import dump, HIGHEST_PROTOCOL

import numpy as np
import time

from .population import QPopulationNetwork, QPopulationParams
from algorithms.checkpoint import save_checkpoint
from .helpers.configs import NetworkRulesConfig, EliteUpdateConfig
from .helpers.operators import apply_crossover

from utils.helpers import (
    delete_old_dirs_v2,
    init_log,
    load_pkl,
    save_pkl,
    calculate_time,
    load_history_from_json,
    save_history_to_json
)


class QNAS(object):
    """Quantum-Inspired Neural Architecture Search (refactored).

    This class orchestrates the entire evolutionary process. It manages two quantum
    populations: one for network architectures (`QPopulationNetwork`) and one for
    training hyperparameters (`QPopulationParams`). It handles the main loop,
    including individual generation, evaluation, selection, and the quantum
    update step.
    """

    def __init__(self, eval_func, experiment_path, objectives, log_file, log_level,
                data_file, use_cache=False):
        """Initializes the core QNAS algorithm object.

        Note: This sets up the framework. The specific evolutionary parameters and
        populations are configured by calling the `initialize_qnas` method.

        Args:
            eval_func (callable): A function that evaluates a list of individuals.
                It receives a list of parameter dicts and a list of network
                architectures. It must return a dictionary mapping each
                individual's 'candidate_id' to its performance metrics.
            experiment_path (str): The root directory for saving logs, caches,
                and other experiment artifacts.
            objectives (list): A list of objective names to optimize (e.g.,
                ["accuracy", "latency"]). The first objective is considered primary.
            log_file (str): The path to the log file.
            log_level (str): Logging verbosity level ("INFO", "DEBUG", or "NONE").
            data_file (str): Path to the .pkl file used for saving and resuming
                the evolution state.
            use_cache (bool, optional): If True, enables caching of evaluated
                individuals to avoid re-computation. Defaults to True.
        """
        # --- Basic settings & bookkeeping ---
        self.dtype = np.float64
        self.tolerance = 1e-15

        self.best_so_far = 0.0
        self.best_so_far_id = [0, 0]
        self.current_best_id = [0, 0]
        self.current_gen = 0
        self.eval_idx = np.array([], dtype=int)

        self.data_file = data_file
        self.eval_func = eval_func
        self.experiment_path = experiment_path
        self.objectives = objectives

        self.logger = init_log(log_level, name=__name__, file_path=log_file)
        self.use_cache = use_cache

        # Attributes set in initialize_qnas()
        self.fitnesses = None
        self.raw_fitnesses = None
        self.generations = None
        self.update_quantum_gen = None
        self.replace_method = None
        self.penalize_number = None
        self.reducing_fns_list = []
        self.penalties = None
        self.random = 0.0
        self.total_eval = 0
        self.early_stopping = None
        self.patience = None
        self.early_stopping_counter = 0
        self.last_best_so_far = 0.0
        self.since_last_mutation = 0
        self.en_pop_crossover = None
        self.pop_crossover_rate = None
        self.crossover_frequency = None
        self.save_data_freq = np.inf
        self.qpop_params = None
        self.qpop_net = None

        # Caching now lives in core/eval_cache.py (wraps eval_func).
        if self.use_cache:
            self.logger.warning(
                "The legacy per-algorithm evaluation cache was removed; pass "
                "--use_cache to enable the unified cache (core/eval_cache.py).")

        self.unique_networks_path = os.path.join(self.experiment_path, "unique_networks.pkl")
        self.unique_networks_db = (
            load_pkl(self.unique_networks_path)
            if os.path.exists(self.unique_networks_path)
            else {}
        )

    def initialize_qnas(self, num_quantum_ind, repetition, max_generations, update_quantum_gen,
                        quantum_update_config, replace_method, params_ranges, crossover_rate,
                        fn_list, initial_probs, max_num_nodes, reducing_fns_list, en_pop_crossover=False, 
                        pop_crossover_method=["hux"], pop_crossover_rate=0.25, crossover_frequency=5,
                        elite_mode="global_k", k_elites=5, pool_factor=2, ema_beta=0.7, rank_weighting=True,
                        terminal_op_name="no_op", pool_op_name="pool", min_active_len=5,
                        truncate_after_noop=True, avoid_consecutive_pool=True, enforce_noop_in_update=True, 
                        noop_max_prob=0.90, noop_ramp_cap=True, early_stopping=True, 
                        patience=10, penalize_number=0, save_data_freq=1):
        """Configures the QNAS populations and evolutionary hyperparameters.

        Args:
            num_quantum_ind (int): Number of quantum individuals.
            repetition (int): Number of classical individuals per quantum one.
            max_generations (int): Total number of generations to run.
            update_quantum_gen (int): Frequency (in generations) for quantum updates.
            quantum_update_config (dict): Configuration for quantum updates, including
                learning rate and scheduling.
            replace_method (str): Survivor selection method ("elitism" or "best").
            params_ranges (dict): Search space for hyperparameters, formatted as
                `{'param_name': [lower_bound, upper_bound]}`.
            crossover_rate (float): Crossover rate for hyperparameter chromosomes.
            fn_list (list): List of all possible operation names for network nodes.
            initial_probs (list): Initial probabilities for each operation in `fn_list`.
            max_num_nodes (int): The maximum length of a network chromosome.
            reducing_fns_list (list): List of operation names that are penalized.
            en_pop_crossover (bool, optional): If True, enables network crossover.
                Defaults to False.
            pop_crossover_method (list, optional): Crossover types for networks
                ["hux", "uniform", "one_point", "two_point"]. Defaults to ["hux"].
            pop_crossover_rate (float, optional): Fraction of the population to
                be replaced by crossover offspring. Defaults to 0.25.
            crossover_frequency (int, optional): Apply network crossover every N
                generations. Defaults to 5.
            elite_mode (str, optional): Strategy for building target distributions.
                Options: "single", "global_k", "bootstrap_k", "moead_topk".
                Defaults to "global_k".
            k_elites (int, optional): Number of elite individuals to use.
                Defaults to 5.
            pool_factor (int, optional): Multiplier for elite pool size in
                "bootstrap_k" mode. Defaults to 2.
            ema_beta (float, optional): EMA smoothing factor for global elite
                distributions. Set to 0.0 to disable. Defaults to 0.7.
            rank_weighting (bool, optional): If True, weight elite contributions by
                inverse rank. Defaults to True.
            terminal_op_name (str, optional): Name of the terminal operation.
                Defaults to "no_op".
            pool_op_name (str | list, optional): Name or pattern(s) to identify
                pooling layers. Defaults to "pool".
            min_active_len (int, optional): Minimum network length before a
                terminal op is allowed. Defaults to 5.
            truncate_after_noop (bool, optional): If True, forces all subsequent
                nodes to be terminal ops after the first one appears. Defaults to True.
            avoid_consecutive_pool (bool, optional): If True, prevents sampling
                two pooling layers in a row. Defaults to True.
            enforce_noop_in_update (bool, optional): If True, applies architecture
                rules during the quantum update. Defaults to True.
            noop_max_prob (float, optional): The maximum probability for a terminal op.
                Defaults to 0.90.
            noop_ramp_cap (bool, optional): If True, linearly increases the
                terminal op's probability cap over the chromosome length.
                Defaults to True.
            early_stopping (bool, optional): If True, enables early stopping.
                Defaults to True.
            patience (int, optional): Generations to wait for improvement before
                stopping. Defaults to 10.
            penalize_number (int, optional): Maximum allowed number of reducing
                layers before a penalty is applied. Defaults to 0.
            save_data_freq (int, optional): Save best model stats every N generations.
                Defaults to 1.
        """
        # 1) Evolution settings
        self.generations = max_generations
        self.update_quantum_gen = update_quantum_gen
        self.replace_method = replace_method
        self.penalize_number = penalize_number
        self.patience = patience
        self.early_stopping = early_stopping

        # 2) Population crossover settings
        self.en_pop_crossover = en_pop_crossover
        self.pop_crossover_rate = pop_crossover_rate
        self.crossover_frequency = crossover_frequency
        self.crossover_methods = pop_crossover_method
        self.logger.info("Network population crossover methods: %s", self.crossover_methods)

        # 3) Reducing-layer penalty setup
        if reducing_fns_list:
            self.reducing_fns_list = [
                i for i, name in enumerate(fn_list) if name in reducing_fns_list
            ]
            self.penalties = np.zeros(shape=(num_quantum_ind * repetition,))
        else:
            self.reducing_fns_list = []
            self.penalties = None

        # 4) CSV-save frequency
        if save_data_freq:
            self.save_data_freq = save_data_freq

        # 5) Build quantum-population objects
        self.qpop_params = QPopulationParams(
            num_quantum_ind=num_quantum_ind,
            params_ranges=params_ranges,
            repetition=repetition,
            crossover_rate=crossover_rate,
            update_quantum_rate=quantum_update_config.get('static_rate', 0.1),
        )

        rules_cfg = NetworkRulesConfig(
            terminal_op_name=terminal_op_name,
            pool_op_name=pool_op_name,
            min_active_len=min_active_len,
            truncate_after_noop=truncate_after_noop,
            avoid_consecutive_pool=avoid_consecutive_pool,
            enforce_noop_in_update=enforce_noop_in_update,
            noop_max_prob=noop_max_prob,
            noop_ramp_cap=noop_ramp_cap
        )

        elite_cfg = EliteUpdateConfig(
            elite_mode=elite_mode,
            k_elites=k_elites,
            pool_factor=pool_factor,
            ema_beta=ema_beta,
            rank_weighting=rank_weighting
        )

        self.qpop_net = QPopulationNetwork(
            num_quantum_ind=num_quantum_ind,
            max_num_nodes=max_num_nodes,
            repetition=repetition,
            quantum_update_config=quantum_update_config,
            fn_list=fn_list,
            initial_probs=initial_probs,
            experiment_path=self.experiment_path,
            rules_config=rules_cfg,
            elite_config=elite_cfg,
            moea_config=None
        )

        U_total = max(1, max_generations // max(1, update_quantum_gen))
        self.qpop_net.set_schedule_total_updates(U_total)

    def select_population(self, old_params: np.ndarray, old_nets: np.ndarray, old_pen: np.ndarray,
                        old_raw: np.ndarray, old_eval_idx: np.ndarray, new_params: np.ndarray,
                        new_nets: np.ndarray, new_pen: np.ndarray, new_raw: np.ndarray, new_eval_idx: np.ndarray):
        """Selects the survivor population by merging and sorting old and new individuals.

        This method combines the previous generation's individuals with the newly
        generated offspring, sorts them all by their penalized fitness in descending
        order, and keeps the top individuals to form the next generation's population.

        Args:
            old_params (np.ndarray): Hyperparameter chromosomes from the old population.
            old_nets (np.ndarray): Network chromosomes from the old population.
            old_pen (np.ndarray): Penalized fitness values of the old population.
            old_raw (np.ndarray): Raw fitness values of the old population.
            old_eval_idx (np.ndarray): Evaluation indices of the old population.
            new_params (np.ndarray): Hyperparameter chromosomes of the new offspring.
            new_nets (np.ndarray): Network chromosomes of the new offspring.
            new_pen (np.ndarray): Penalized fitness values of the new offspring.
            new_raw (np.ndarray): Raw fitness values of the new offspring.
            new_eval_idx (np.ndarray): Evaluation indices of the new offspring.

        Returns:
            tuple: A tuple containing the chromosomes, fitness values (penalized and raw),
                and evaluation indices of the selected survivor population.
        """
        num_classic = self.qpop_params.num_ind * self.qpop_params.repetition
        # 1) First generation: simply take the new population
        if self.current_gen == 0:
            k = min(num_classic, new_pen.shape[0])
            sorted_pen, sorted_raw, sorted_params, sorted_nets, sorted_eidx = self.order_pop(
                new_pen, new_raw, new_params, new_nets, new_eval_idx, selection=range(k)
            )
            return sorted_params, sorted_nets, sorted_pen, sorted_raw, sorted_eidx

        if self.replace_method == "elitism":
            selected = [0]  # Keep only the single best individual
        else:
            selected = list(range(len(old_pen)))  # Keep all previous individuals

        kept_old_pen = old_pen[selected]
        kept_old_raw = old_raw[selected]
        kept_old_params = old_params[selected]
        kept_old_nets = old_nets[selected]

        all_pen = np.concatenate([kept_old_pen, new_pen])
        all_raw = np.concatenate([kept_old_raw, new_raw])
        all_params = np.concatenate([kept_old_params, new_params])
        all_nets = np.concatenate([kept_old_nets, new_nets])
        all_eval_idx = np.concatenate([old_eval_idx[selected], new_eval_idx])

        sorted_pen, sorted_raw, sorted_params, sorted_nets, sorted_eidx = self.order_pop(
            all_pen, all_raw, all_params, all_nets, all_eval_idx, selection=range(num_classic)
        )

        return sorted_params, sorted_nets, sorted_pen, sorted_raw, sorted_eidx

    @staticmethod
    def order_pop(fitnesses: np.ndarray, raw_fitnesses: np.ndarray,
                pop_params: np.ndarray, pop_net: np.ndarray, pop_eval_idx: np.ndarray, selection=None):
        """Sorts a population by fitness and selects a subset.

        Args:
            fitnesses (np.ndarray): The fitness values (typically penalized) to sort by.
            raw_fitnesses (np.ndarray): The corresponding raw (unpenalized) fitness values.
            pop_params (np.ndarray): The hyperparameter chromosomes.
            pop_net (np.ndarray): The network chromosomes.
            pop_eval_idx (np.ndarray): The evaluation indices.
            selection (iterable, optional): A slice or list of indices to keep after
                sorting. If None, the entire sorted population is returned.

        Returns:
            tuple: A tuple containing the sorted and selected fitness values (penalized
                and raw), chromosomes, and evaluation indices.
        """
        if selection is None:
            selection = range(fitnesses.shape[0])
        idx = np.argsort(fitnesses)[::-1]  # Sort in descending order
        sorted_params = pop_params[idx][selection]
        sorted_nets = pop_net[idx][selection]
        sorted_fits = fitnesses[idx][selection]
        sorted_raw = raw_fitnesses[idx][selection]
        sorted_eval_idx = pop_eval_idx[idx][selection]
        return sorted_fits, sorted_raw, sorted_params, sorted_nets, sorted_eval_idx

    def update_best_id(self, penalized_fits: np.ndarray, eval_idx: np.ndarray):
        """Updates the record of the best individual found so far.

        This tracks both the best individual in the current generation and the
        all-time best, based on penalized fitness.

        Args:
            penalized_fits (np.ndarray): Penalized fitness values of the current population.
            eval_idx (np.ndarray): Evaluation indices of the current population.
        """
        if penalized_fits is None or penalized_fits.size == 0:
            return
        safe = np.where(np.isnan(penalized_fits), -np.inf, penalized_fits)
        i_best = int(np.argmax(safe))
        val = float(safe[i_best])
        eidx = int(eval_idx[i_best])

        self.current_best_id = [self.current_gen, eidx]
        self.current_best_value = val

        if getattr(self, "best_so_far", None) is None:
            self.best_so_far = -np.inf
            self.best_so_far_id = [-1, -1]

        if val > self.best_so_far:
            self.best_so_far = val
            self.best_so_far_id = [self.current_gen, eidx]

    def generate_classical(self):
        """Generates a new classical population by sampling from the quantum states.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the new hyperparameter
                                        chromosomes and network chromosomes.
        """
        self.random = np.random.rand()
        new_pop_params = self.qpop_params.generate_classical()
        new_pop_net = self.qpop_net.generate_classical()
        self.logger.info("Generated classical networks:\n%s", new_pop_net)
        return new_pop_params, new_pop_net

    def decode_pop(self, pop_params: np.ndarray, pop_net: np.ndarray):
        """Decodes numerical chromosomes into human-readable formats.

        Args:
            pop_params (np.ndarray): Encoded hyperparameter chromosomes.
            pop_net (np.ndarray): Encoded network chromosomes.

        Returns:
            tuple[list[dict], list]: A tuple containing a list of decoded
                                    hyperparameter dictionaries and a list of
                                    decoded network architectures.
        """
        num_ind = pop_net.shape[0]
        decoded_params = [None] * num_ind
        decoded_nets = [None] * num_ind
        for i in range(num_ind):
            decoded_params[i] = self.qpop_params.chromosome.decode(pop_params[i])
            decoded_params[i]["candidate_id"] = i
            decoded_nets[i] = self.qpop_net.chromosome.decode(pop_net[i, :])
        return decoded_params, decoded_nets

    def _eval_pop_without_cache(self, decoded_params, decoded_nets, pop_net):
        """
        Internal: Evaluates the population without using a cache, while recording
        a persistent registry of unique evaluated networks.

        Notes
        -----
        - Every individual in the population is evaluated.
        - Each unique network is stored only once in `self.unique_networks_db`.
        - Repeated networks only update the `visits` counter.
        - The registry is persisted every 5 generations using `save_pkl(...)`.
        """
        num_individuals = len(decoded_nets)
        metric_key = self.objectives[0]

        self.logger.info(
            "Evaluating population of %d individuals without cache.", num_individuals
        )

        results = self.eval_func(
            decoded_params,
            decoded_nets,
            generation=self.current_gen,
        )

        if not results:
            self.logger.warning("Evaluation function returned no results.")
            return np.full(num_individuals, np.nan, dtype=float)

        raw_fits = np.full(num_individuals, np.nan, dtype=float)

        new_unique_count = 0
        repeated_count = 0

        for i in range(num_individuals):
            net_key = tuple(pop_net[i])
            candidate_id = decoded_params[i]["candidate_id"]

            if candidate_id not in results:
                self.logger.warning(
                    "Candidate %s was not found in results.", candidate_id
                )
                continue

            fitness = float(results[candidate_id].get(metric_key, np.nan))
            raw_fits[i] = fitness

            if net_key not in self.unique_networks_db:
                self.unique_networks_db[net_key] = {
                    "fitness": fitness,
                    "first_generation": self.current_gen,
                    "first_index": i,
                    "candidate_id": candidate_id,
                    "visits": 1,
                }
                new_unique_count += 1
            else:
                self.unique_networks_db[net_key]["visits"] += 1
                repeated_count += 1

        self.total_eval += num_individuals

        if self.current_gen % 5 == 0:
            save_pkl(self.unique_networks_path, self.unique_networks_db)

        self.logger.info(
            "Total evals: %d | New unique: %d | Repeated in batch: %d | Total unique so far: %d",
            self.total_eval,
            new_unique_count,
            repeated_count,
            len(self.unique_networks_db),
        )

        return raw_fits

    def _eval_pop_with_history(self, decoded_params, decoded_nets, pop_net):
        """Internal: Evaluates the population using a persistent history database (memoization)."""
        num_individuals = len(pop_net)
        final_fitnesses = [None] * num_individuals
        to_eval_indices, to_eval_params, to_eval_nets, to_eval_keys = [], [], [], []

        self.logger.info("Checking history for %d individuals...", num_individuals)
        for i, individual_net_array in enumerate(pop_net):
            key = tuple(individual_net_array)
            if key in self.history_database:
                final_fitnesses[i] = self.history_database[key]
            else:
                to_eval_indices.append(i)
                to_eval_params.append(decoded_params[i])
                to_eval_nets.append(decoded_nets[i])
                to_eval_keys.append(key)

        self.logger.info("%d individuals found in history. %d new individuals need evaluation.",
                        num_individuals - len(to_eval_indices), len(to_eval_indices))

        if to_eval_indices:
            self.logger.info("Sending %d new individuals for batch evaluation.", len(to_eval_indices))
            results = self.eval_func(to_eval_params, to_eval_nets, generation=self.current_gen)

            if not results:
                self.logger.error("Evaluation function returned no results for the batch.")
            else:
                metric_key = self.objectives[0]
                for i, original_index in enumerate(to_eval_indices):
                    key_to_update = to_eval_keys[i]
                    candidate_id = to_eval_params[i]['candidate_id']
                    if candidate_id in results:
                        true_fitness = float(results[candidate_id].get(metric_key, 0.0))
                        final_fitnesses[original_index] = true_fitness
                        self.history_database[key_to_update] = true_fitness
                        self.total_eval += 1
                    else:
                        self.logger.warning("Candidate %s was not found in results.", candidate_id)
            save_history_to_json(self.history_database, self.history_database_path)

        return np.array([f if f is not None else 0.0 for f in final_fitnesses], dtype=float)

    def eval_pop(self, pop_params: np.ndarray, pop_net: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates a population and applies penalties.

        This method coordinates the decoding, evaluation (with or without cache),
        and penalization of a given population.

        Args:
            pop_params (np.ndarray): The hyperparameter chromosomes.
            pop_net (np.ndarray): The network chromosomes.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the penalized fitness
                                        values and the raw fitness values.
        """
        decoded_params, decoded_nets = self.decode_pop(pop_params, pop_net)
        self.logger.info("Evaluating generation %d (size %d)...", self.current_gen, len(decoded_nets))

        raw_fits = self._eval_pop_without_cache(decoded_params, decoded_nets, pop_net)

        penalized_fits = raw_fits.copy()
        if self.penalize_number and self.reducing_fns_list:
            penalties = self.get_penalties(pop_net)
            penalized_fits -= penalties

        return penalized_fits, raw_fits

    def get_penalties(self, pop_net: np.ndarray, penalty_factor: float = 0.02) -> np.ndarray:
        """Computes a penalty for networks with too many 'reducing' layers.

        Args:
            pop_net (np.ndarray): The network chromosomes to evaluate.
            penalty_factor (float, optional): The penalty amount per excess
                reducing layer. Defaults to 0.02.

        Returns:
            np.ndarray: An array of penalty values for each individual.
        """
        penalties = np.zeros(shape=(pop_net.shape[0],), dtype=float)
        for i, net in enumerate(pop_net):
            unique, counts = np.unique(net, return_counts=True)
            reducing_count = sum(
                counts[j] for j, u in enumerate(unique) if u in self.reducing_fns_list
            )
            if reducing_count > self.penalize_number:
                penalties[i] = (reducing_count - self.penalize_number)
        return penalty_factor * penalties

    def crossover_hyperparams(self, new_pop_params: np.ndarray) -> np.ndarray:
        """Applies crossover to the hyperparameter population.

        Args:
            new_pop_params (np.ndarray): The new hyperparameter chromosomes.

        Returns:
            np.ndarray: The chromosomes after applying crossover.
        """
        if self.current_gen > 0:
            try:
                new_pop_params = self.qpop_params.classic_crossover(
                    new_pop=new_pop_params,
                    distance=self.random,
                )
            except AttributeError:
                pass  # Skip if the method doesn't exist
        return new_pop_params

    def crossover_network(self, new_pop_net: np.ndarray) -> np.ndarray:
        """Applies crossover to the network population if conditions are met.

        Crossover is triggered based on generation frequency and the
        `en_pop_crossover` flag. It selects random individuals from
        new_pop_net to cross with the best from current_pop.

        Args:
            new_pop_net (np.ndarray): The new network chromosomes.

        Returns:
            np.ndarray: The network chromosomes after potentially applying crossover.
        """
        if self.current_gen > 0 and getattr(self, "en_pop_crossover", False):
            if self.current_gen % self.crossover_frequency == 0:
                
                # Determine how many to crossover based on rate
                num_off = int(len(new_pop_net) * self.pop_crossover_rate)
                
                # Ensure we don't try to crossover more than we have
                num_off = min(num_off, len(self.qpop_net.current_pop), len(new_pop_net))
                if num_off > 0:
                    # 1. Select BEST individuals from the current (old) population as first parents
                    best_current = self.qpop_net.current_pop[:num_off]
                    # 2. Select RANDOM indices from the new population to be second parents AND replaced
                    replace_indices = np.random.choice(len(new_pop_net), num_off, replace=False)
                    parents_from_new = new_pop_net[replace_indices]
                    # 3. Apply crossover using the new list of methods
                    # Make sure self.crossover_methods is loaded as a list ['hux', 'one_point', etc.]
                    try:
                        offspring = apply_crossover(
                            best_current, 
                            parents_from_new, 
                            method_keys=self.crossover_methods
                        )
                        # 4. Replace the chosen individuals in the new population with offspring
                        new_pop_net[replace_indices] = offspring
                        
                        self.logger.info(f"Crossover applied. Replaced {num_off} individuals at indices {replace_indices}")
                        
                    except Exception as e:
                        self.logger.error(f"Crossover failed: {e}")
        return new_pop_net

    @staticmethod
    def _fmt_arr(a, prec=6):
        """Formats a numpy array as a string for logging."""
        return np.array2string(
            np.asarray(a),
            formatter={'float_kind': lambda x: f"{x:.{prec}f}"},
            max_line_width=200,
            separator=' '
        )

    def log_data(self):
        """Logs a summary of the current generation's results."""
        self.logger.info(
            "Generation %d complete!\n"
            "- Best so far: %s -> %.5f\n"
            "- Penalized fitnesses: %s\n"
            "- Raw fitnesses: %s\n",
            self.current_gen,
            self.best_so_far_id,
            self.best_so_far,
            self._fmt_arr(self.fitnesses, prec=4),
            self._fmt_arr(self.raw_fitnesses, prec=4),
        )

    def save_data(self):
        """Saves the complete state of the evolution to a pickle file.

        This allows for resuming the experiment later. The data is keyed by
        the generation number.
        """
        data = load_pkl(self.data_file) if os.path.exists(self.data_file) else {}

        data[self.current_gen] = {
            "time": str(datetime.datetime.now()),
            "total_eval": self.total_eval,
            "best_so_far": self.best_so_far,
            "best_so_far_id": self.best_so_far_id,
            "fitnesses": self.fitnesses,
            "raw_fitnesses": self.raw_fitnesses,
            "lower": self.qpop_params.lower,
            "upper": self.qpop_params.upper,
            "params_pop": self.qpop_params.current_pop,
            "net_probs": self.qpop_net.probabilities,
            "num_net_nodes": self.qpop_net.chromosome.num_genes,
            "net_pop": self.qpop_net.current_pop,
        }
        self.dump_pkl_data(data)

    def dump_pkl_data(self, new_data: dict):
        """Writes data to the main pickle file.

        Args:
            new_data (dict): The dictionary to save.
        """
        with open(self.data_file, "wb") as f:
            dump(new_data, f, protocol=HIGHEST_PROTOCOL)

    def load_qnas_data(self, file_path: str):
        """Loads a previous QNAS state from a pickle file to resume an experiment.

        Args:
            file_path (str): The path to the state pickle file.
        """
        log_data = load_pkl(file_path)
        if not os.path.exists(self.data_file):
            self.dump_pkl_data(log_data)

        generation = max(log_data.keys())
        state = log_data[generation]

        self.current_gen = generation
        self.total_eval = state["total_eval"]
        self.best_so_far = state["best_so_far"]
        self.best_so_far_id = state["best_so_far_id"]
        self.qpop_net.chromosome.set_num_genes(state["num_net_nodes"])
        self.fitnesses = state["fitnesses"]
        self.raw_fitnesses = state["raw_fitnesses"]
        self.qpop_params.lower = state["lower"]
        self.qpop_params.upper = state["upper"]
        self.qpop_net.probabilities = state["net_probs"]
        self.qpop_params.current_pop = state["params_pop"]
        self.qpop_net.current_pop = state["net_pop"]

    def check_early_stopping(self) -> bool:
        """Checks if the early stopping criteria have been met.

        Stops if the best fitness has not improved by at least 0.5% over a
        number of generations defined by `self.patience`.

        Returns:
            bool: True if evolution should stop, False otherwise.
        """
        if self.current_gen > 1:
            improvement = (self.best_so_far - self.last_best_so_far) / self.last_best_so_far \
                if self.last_best_so_far != 0 else 0.0

            if improvement > 0.005:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            self.logger.info("Early stopping counter: %d", self.early_stopping_counter)
            if self.early_stopping_counter >= self.patience:
                self.logger.info("Early stopping at generation %d!", self.current_gen)
                return True

        self.last_best_so_far = self.best_so_far
        return False

    def update_quantum(self, current_gen):
        """Triggers the quantum update step for both populations.

        This is typically called every `self.update_quantum_gen` generations.

        Args:
            current_gen (int): The current generation number.
        """
        if self.current_gen > 0 and (self.current_gen % self.update_quantum_gen == 0):
            self.logger.info("Updating quantum populations...")
            self.qpop_params.update_quantum(intensity=self.random)
            self.qpop_net.update_quantum(intensity=self.random, current_gen=current_gen)

    def go_next_gen(self):
        """Performs all end-of-generation bookkeeping tasks.

        This includes:
        1. Updating the quantum populations.
        2. Backing up the evaluation cache.
        3. Saving the current state to a pickle file.
        4. Logging the generation summary.
        5. Cleaning up old model directories.
        6. Incrementing the generation counter.
        """
        self.update_quantum(self.current_gen)
        self.save_data()
        self.log_data()

        best_gen, best_idx = self.best_so_far_id
        best_id = f"{best_gen}_{best_idx}" if best_gen >= 0 and best_idx >= 0 else None
        keep_ids = [best_id] if best_id else []
        delete_old_dirs_v2(self.experiment_path, self.current_gen, keep_ids=keep_ids)

        # Generation boundary: quantum update + save_data done, g+1 not begun.
        save_checkpoint(self)
        self.current_gen += 1

    # ---- Checkpoint hooks (consumed by algorithms.checkpoint) ----

    def _checkpoint_config_block(self) -> dict:
        """Identity-defining config validated on resume (see algorithms.checkpoint)."""
        block = {
            'algorithm': type(self).__name__,
            'objectives': list(self.objectives),
            'num_quantum_ind': int(self.qpop_net.num_ind),
            'fn_list': list(self.qpop_net.chromosome.fn_list),
            'max_generations': int(getattr(self, 'max_generations', 0)
                                   or getattr(self, 'generations', 0) or 0),
            'update_quantum_gen': int(self.update_quantum_gen),
            'crossover_frequency': int(getattr(self, 'crossover_frequency', 0) or 0),
        }
        block.update(getattr(self, 'checkpoint_extra', {}) or {})
        return block

    def _checkpoint_state(self) -> dict:
        """Resumable quantum-search state (PMFs + coupled elite state)."""
        qn, qp = self.qpop_net, self.qpop_params
        return {
            'random_intensity': getattr(self, 'random', 0.0),
            'qpop_net': {
                'probabilities': qn.probabilities,
                'current_pop': qn.current_pop,
                'current_pop_objs': getattr(qn, 'current_pop_objs', None),
                'num_genes': qn.chromosome.num_genes,
                'update_counter': qn.logger._update_counter,
                'last_P': qn.logger._last_P,
                'q_ema': qn.update_strategy._q_ema,
            },
            'qpop_params': {
                'lower': qp.lower, 'upper': qp.upper, 'current_pop': qp.current_pop,
            },
        }

    def _restore_state(self, s: dict) -> None:
        qn, qp = self.qpop_net, self.qpop_params
        self.random = s['random_intensity']
        n = s['qpop_net']
        qn.chromosome.set_num_genes(n['num_genes'])
        qn.probabilities = n['probabilities']
        qn.current_pop = n['current_pop']
        if n['current_pop_objs'] is not None:
            qn.current_pop_objs = n['current_pop_objs']
        qn.logger._update_counter = n['update_counter']
        qn.logger._last_P = n['last_P']
        qn.update_strategy._q_ema = n['q_ema']
        p = s['qpop_params']
        qp.lower, qp.upper, qp.current_pop = p['lower'], p['upper'], p['current_pop']

    def evolve(self):
        """Runs the main evolutionary loop."""
        self._session_start = time.time()
        if not hasattr(self, '_elapsed_so_far'):
            self._elapsed_so_far = 0.0
        max_gen = self.generations
        if self.current_gen > 0:  # Adjust for resumed runs
            max_gen += (self.current_gen + 1)
            self.current_gen += 1

        while self.current_gen < max_gen:
            # --- Log ETA periodically ---
            if self.current_gen > 0 and (self.current_gen % 5 == 0):
                h, m, est_h, est_m = calculate_time(
                    self._session_start, time.time(), self.current_gen, max_gen, end_evol=False)
                self.logger.info("Gen %d: elapsed %dh %dm; ETA %dh %dm",
                                self.current_gen, h, m, est_h, est_m)

            # --- Generation, Crossover, and Evaluation ---
            new_p, new_n = self.generate_classical()
            new_p = self.crossover_hyperparams(new_p)
            new_n = self.crossover_network(new_n)
            new_f_pen, new_f_raw = self.eval_pop(new_p, new_n)
            new_eval_idx = np.arange(new_p.shape[0], dtype=int)

            # --- Selection ---
            next_p, next_n, next_pen, next_raw, next_eidx = self.select_population(
                self.qpop_params.current_pop, self.qpop_net.current_pop,
                self.fitnesses, self.raw_fitnesses, self.eval_idx,
                new_p, new_n, new_f_pen, new_f_raw, new_eval_idx,
            )

            # --- Update State for Next Generation ---
            self.qpop_params.current_pop = next_p
            self.qpop_net.current_pop = next_n
            self.fitnesses = next_pen
            self.raw_fitnesses = next_raw
            self.eval_idx = next_eidx
            self.update_best_id(self.fitnesses, self.eval_idx)

            # --- Bookkeeping and Cleanup ---
            self.go_next_gen()

            # --- Check for Early Stopping ---
            if self.early_stopping and self.check_early_stopping():
                break

        save_pkl(self.unique_networks_path, self.unique_networks_db)
        total_seconds = self._elapsed_so_far + (time.time() - self._session_start)
        total_h, rem = divmod(total_seconds, 3600)
        total_m, _ = divmod(rem, 60)
        self.logger.info("Total evolution time: %d hours and %d minutes (all sessions)", int(total_h), int(total_m))