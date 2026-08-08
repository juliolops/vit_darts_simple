""" 
MoQNAS: Multi-Objective Quantum-Inspired Neural Architecture Search
Based on the refactored QNAS (qnas2.py). 

This implements NSGA-II style Pareto selection over multiple objectives, 
while still using the quantum-population machinery from QNAS to evolve 
hyperparameter distributions. 
"""
import os
import time
import json
import pickle
import numpy as np
from pymoo.indicators.hv import Hypervolume
from settings import CFG_OBJ_PATH
from algorithms.pareto import dominates, fast_nondominated_sort, crowding_distance, compute_hypervolume_mixed
from algorithms.checkpoint import save_checkpoint
from .qnas2 import QNAS
from .helpers.configs import MOEAConfig
from .helpers.operators import apply_crossover
from utils.helpers import calculate_time, delete_old_dirs_v2, load_pkl, save_pkl


class MOQNAS(QNAS):
    """Multi-Objective QNAS using NSGA-II style selection.

    This class extends the single-objective QNAS algorithm to handle multiple
    competing objectives. It replaces the simple elitist selection with Pareto
    dominance ranking and crowding distance to maintain a diverse set of
    optimal trade-off solutions (the Pareto front).
    """

    def __init__(self, eval_func, experiment_path, objectives, log_file, log_level, data_file):
        """Initializes the MOQNAS algorithm.

        Args:
            eval_func (callable): A function that evaluates a list of individuals.
                It must return a dictionary mapping each individual's 'candidate_id'
                to its performance metrics dict (e.g., {'accuracy': 0.9, 'latency': 100}).
            experiment_path (str): The root directory for saving all experiment artifacts.
            objectives (list[str]): A list of objective names to optimize.
            log_file (str): The path to the log file.
            log_level (str): Logging verbosity level ("INFO", "DEBUG", or "NONE").
            data_file (str): Path to the .pkl file for saving and resuming evolution.
        """
        super().__init__(eval_func, experiment_path, objectives, log_file, log_level, data_file)
        self.num_objectives = len(objectives)
        self.data_file = data_file
        self.pop_size = None
        self.max_generations = None
        self.mo_crossover_strategy = "directional"  # Default crossover strategy
        self.unique_networks_path = os.path.join(self.experiment_path, "unique_networks.pkl")
        self.unique_networks_db = (
            load_pkl(self.unique_networks_path)
            if os.path.exists(self.unique_networks_path)
            else {}
        )

    def initialize_moqnas(self, num_quantum_ind: int, repetition: int, max_generations: int,
                        update_quantum_gen: int, quantum_update_config: dict, params_ranges: dict,
                        # --- Network Population ---
                        fn_list: list,
                        initial_probs: list,
                        max_num_nodes: int,
                        reducing_fns_list: list,
                        
                        # --- Network Crossover ---
                        crossover_rate: float,
                        en_pop_crossover: bool = False,
                        pop_crossover_method: list = ["hux"],
                        mo_crossover_strategy: str = "directional",
                        pop_crossover_rate: float = 0.25,
                        crossover_frequency: int = 5,
                        
                        # --- Elite Selection & MOEA/D ---
                        elite_mode: str = "moead_topk",
                        k_elites: int = 5,
                        pool_factor: int = 2,
                        ema_beta: float = 0.7,
                        rank_weighting: bool = True,
                        ref_dir_method: str = "das-dennis",
                        
                        # --- Network Architecture Rules ---
                        terminal_op_name: str = "no_op",
                        pool_op_name: str = "pool",
                        min_active_len: int = 5,
                        truncate_after_noop: bool = True,
                        avoid_consecutive_pool: bool = True,
                        
                        # --- No-Op Probability Management ---
                        enforce_noop_in_update: bool = True,
                        noop_max_prob: float = 0.90,
                        noop_ramp_cap: bool = True,
                        
                        # --- Stopping & Penalties ---
                        early_stopping: bool = True,
                        patience: int = 10,
                        penalize_number: float = 0,
                        
                        # --- Misc ---
                        save_data_freq: int = 1,
                        replace_method: str = "best"):
        """Configures the MOQNAS populations and evolutionary hyperparameters.

        Args:
            num_quantum_ind (int): Number of quantum individuals.
            repetition (int): Number of classical individuals per quantum one.
            max_generations (int): Total number of generations to run.
            update_quantum_gen (int): Frequency (in generations) for quantum updates.
            quantum_update_config (dict): Configuration for quantum updates, including
                learning rate and scheduling.
            params_ranges (dict): Search space for hyperparameters.
            crossover_rate (float): Crossover rate for hyperparameter chromosomes.
            fn_list (list): List of all possible operation names for network nodes.
            initial_probs (list): Initial probabilities for each operation.
            max_num_nodes (int): The maximum length of a network chromosome.
            reducing_fns_list (list): List of operation names that are penalized.
            en_pop_crossover (bool, optional): If True, enables network crossover.
                Defaults to False.
            pop_crossover_method (list, optional): Crossover types for networks
                ["hux", "uniform", "one_point", "two_point"]. Defaults to ["hux"].
            mo_crossover_strategy (str, optional): Strategy for multi-objective crossover.
                Defaults to "directional".
            pop_crossover_rate (float, optional): Fraction of the population to
                be replaced by crossover offspring. Defaults to 0.25.
            crossover_frequency (int, optional): Apply network crossover every N
                generations. Defaults to 5.
            elite_mode (str, optional): Strategy for building quantum update targets.
                Defaults to "moead_topk".
            k_elites (int, optional): Number of elite individuals to use. Defaults to 5.
            pool_factor (int, optional): Multiplier for elite pool size. Defaults to 2.
            ema_beta (float, optional): EMA smoothing factor for global elite
                distributions. Set to 0.0 to disable. Defaults to 0.7.
            rank_weighting (bool, optional): If True, weight elite contributions by
                inverse rank. Defaults to True.
            ref_dir_method (str, optional): Method for generating reference vectors
                for MOEA/D-based updates ('das-dennis' or 'dirichlet'). Defaults to "das-dennis".
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
            penalize_number (int, optional): Max allowed reducing layers before penalty.
                Defaults to 0.
            save_data_freq (int, optional): Save best model stats every N generations.
                Defaults to 1.
            replace_method (str, optional): Inherited from QNAS, not used in MOQNAS.
                Defaults to "best".
        """
        self.pop_size = num_quantum_ind * repetition
        self.max_generations = max_generations
        self.hyperparam_crossover_rate = crossover_rate
        self.mo_crossover_strategy = mo_crossover_strategy

        # Initialize the base QNAS class, which sets up the quantum populations
        super().initialize_qnas(
            num_quantum_ind=num_quantum_ind,
            params_ranges=params_ranges,
            repetition=repetition,
            max_generations=max_generations,
            crossover_rate=None,  # Not used in MOQNAS's main loop
            update_quantum_gen=update_quantum_gen,
            replace_method=replace_method,
            fn_list=fn_list,
            initial_probs=initial_probs,
            quantum_update_config=quantum_update_config,
            max_num_nodes=max_num_nodes,
            reducing_fns_list=reducing_fns_list,
            patience=patience,
            early_stopping=early_stopping,
            save_data_freq=save_data_freq,
            penalize_number=penalize_number,
            crossover_frequency=crossover_frequency,
            en_pop_crossover=en_pop_crossover,
            pop_crossover_rate=pop_crossover_rate,
            pop_crossover_method=pop_crossover_method,
            elite_mode=elite_mode,
            k_elites=k_elites,
            pool_factor=pool_factor,
            ema_beta=ema_beta,
            rank_weighting=rank_weighting,
            # Pass the new network architecture parameters to the QPopulationNetwork
            terminal_op_name=terminal_op_name,
            pool_op_name=pool_op_name,
            min_active_len=min_active_len,
            truncate_after_noop=truncate_after_noop,
            avoid_consecutive_pool=avoid_consecutive_pool,
            enforce_noop_in_update=enforce_noop_in_update,
            noop_max_prob=noop_max_prob,
            noop_ramp_cap=noop_ramp_cap,
        )

        self.classical_nets = self.qpop_net.generate_classical()
        self.classical_params = self.qpop_params.generate_classical()
        self.fits = None
        self.raw_fits = None
        self.pareto_global_population = None
        self.pareto_global_fitnesses = None
        self.pareto_global_params = None
        self.pareto_global_ids = []
        self.fronts_history = {}

        # --- MOO Setup ---
        
        # 1. Load objective config from JSON
        self.logger.info(f"Loading objective config from '{CFG_OBJ_PATH}'")
        try:
            with open(CFG_OBJ_PATH, "r") as f:
                self.objectives_info = json.load(f)["objectives"]
        except Exception as e:
            raise RuntimeError(f"Failed to load objective config: {e}")

        # 2. Build objective lists
        self.objective_names = []
        self.objective_senses = []
        for active_obj in self.objectives:
            match_found = False
            for key, info in self.objectives_info.items():
                if key in active_obj:
                    self.objective_names.append(key)
                    sense = 'max' if info['goal'] == 'maximize' else 'min'
                    self.objective_senses.append(sense)
                    match_found = True
                    break
            if not match_found:
                self.logger.warning(f"Could not find a rule for '{active_obj}'")
        
        # 3. Create the MOEAConfig object
        # (You can pull these values from self.args or config files)
        moea_cfg = MOEAConfig(
            moead_q_low=0.30,   # Or self.args.moead_q_low
            moead_q_high=0.90,  # Or self.args.moead_q_high
            topP_mult=5,        # Or self.args.topP_mult
            ref_dir_method=ref_dir_method # This was already a var in your code
        )
        
        # 4. Initialize the MOEAD helper on the quantum population
        self.qpop_net.set_objective_directions(
            names=self.objective_names, 
            sense=self.objective_senses,
            moea_config=moea_cfg
        )
        
        self.logger.info(f"Reference direction method set to: {moea_cfg.ref_dir_method}")
        self.logger.info(f"Set objective directions: {list(zip(self.objective_names, self.objective_senses))}")

        # 5. Log the directions
        log_data = self.qpop_net.moea_helper.get_directions_for_logging()
        for i, log_str in log_data.items():
            self.logger.info(f"Quantum Individual {i}: {log_str}")
        
    def multiobjective_fitness(self) -> np.ndarray:
        """
        Evaluate the current classical population on all objectives, while recording
        a persistent registry of unique evaluated networks.

        Steps:
            1. Decode classical_params and classical_nets via QNAS.decode_pop.
            2. Call self.eval_func(...) which returns a mapping from candidate_id
            to a metrics dictionary.
            3. Build (pop_size x n_obj) arrays for raw and penalized fitness.
            4. Record each unique network only once in self.unique_networks_db.
            5. Persist the registry every 5 generations using save_pkl(...).

        Returns:
            np.ndarray: Penalized fitness array of shape (pop_size, n_obj).
        """
        decoded_params, decoded_nets = self.decode_pop(
            self.classical_params,
            self.classical_nets
        )

        raw_results = self.eval_func(
            decoded_params,
            decoded_nets,
            generation=self.current_gen
        )

        N = len(decoded_nets)
        fits = np.full((N, self.num_objectives), np.nan, dtype=float)
        raw_fits = np.full((N, self.num_objectives), np.nan, dtype=float)

        if not raw_results:
            self.logger.warning("Evaluation function returned no results.")
            self.raw_fits = raw_fits
            self.fits = fits
            return fits

        new_unique_count = 0
        repeated_count = 0

        for i in range(N):
            candidate_id = decoded_params[i].get("candidate_id")

            if candidate_id not in raw_results:
                self.logger.warning(
                    "Candidate %s was not found in results.", candidate_id
                )
                continue

            metrics = raw_results[candidate_id]
            metric_vals = [
                float(metrics.get(obj_name, np.nan))
                for obj_name in self.objectives[: self.num_objectives]
            ]

            raw_fits[i] = metric_vals
            fits[i] = metric_vals

            net_key = tuple(self.classical_nets[i])

            if net_key not in self.unique_networks_db:
                self.unique_networks_db[net_key] = {
                    "fitness": metric_vals,
                    "first_generation": self.current_gen,
                    "first_index": i,
                    "candidate_id": candidate_id,
                    "visits": 1,
                }
                new_unique_count += 1
            else:
                self.unique_networks_db[net_key]["visits"] += 1
                repeated_count += 1

        if self.penalize_number and self.reducing_fns_list:
            penalties = self.get_penalties(self.classical_nets)
            fits[:, 0] -= penalties

        self.total_eval += N

        if self.current_gen % 5 == 0:
            save_pkl(self.unique_networks_path, self.unique_networks_db)

        self.logger.info(
            "Total evals: %d | New unique: %d | Repeated in batch: %d | Total unique so far: %d",
            self.total_eval,
            new_unique_count,
            repeated_count,
            len(self.unique_networks_db),
        )

        self.raw_fits = raw_fits
        self.fits = fits

        return fits

    def environmental_selection(
        self, pop: np.ndarray, fits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform NSGA-II selection on a combined population of size (2*pop_size).
        Ranks by Pareto fronts, then uses crowding distance to fill up to pop_size.
        Returns (selected_pop, selected_fits, selected_indices), where
        selected_indices is a 1D int array of length pop_size giving the row-indices
        in `pop` that were chosen.
        """
        pop_size = self.pop_size
        fronts = fast_nondominated_sort(fits, self.objective_senses)

        new_pop = np.zeros((pop_size, pop.shape[1]), dtype=pop.dtype)
        new_fits = np.zeros((pop_size, fits.shape[1]), dtype=float)
        selected_idx = []  # <-- collect indices here
        count = 0

        for front in fronts:
            # Si ya llenamos la población, salimos inmediatamente.
            if count >= pop_size:
                break

            front_size = len(front)
            if count + front_size <= pop_size:
                # Tomamos todo el frente completo
                new_pop[count : count + front_size] = pop[front]
                new_fits[count : count + front_size] = fits[front]
                selected_idx.extend(front)
                count += front_size
            else:
                # Quedan pocos huecos (rem) y el frente es más grande
                rem = pop_size - count
                if rem > 0:
                    cd = crowding_distance(fits, front)  # (len(front),)
                    # Seleccionamos los índices con mayor crowding distance
                    top_indices = np.argsort(cd)[-rem:]
                    # Convertimos esos índices relativos en índices absolutos sobre 'pop'
                    chosen = [front[i] for i in top_indices]
                    new_pop[count : count + rem] = pop[chosen]
                    new_fits[count : count + rem] = fits[chosen]
                    selected_idx.extend(chosen)
                    count = pop_size
                break  # Después de llenar con 'rem' individuos, rompemos el bucle
        selected_idx = np.array(selected_idx, dtype=int)
        return new_pop, new_fits, selected_idx

    def random_crossover_hyperparams(self, new_pop: np.ndarray) -> np.ndarray:
        """
        Perform a random‐parent “tournament” crossover on hyperparameters.

        Each child in new_pop has a chance (hyperparam_crossover_rate) to
        be crossed with a random parent from self.qpop_params.current_pop.
        """
        old_pop = self.qpop_params.current_pop
        N_old = 0 if old_pop is None else old_pop.shape[0]
        if N_old == 0:
            return new_pop

        mixed = new_pop.copy()
        for i in range(new_pop.shape[0]):
            if np.random.rand() < self.hyperparam_crossover_rate:
                p_idx = np.random.randint(N_old)
                parent = old_pop[p_idx]
                child = new_pop[i]
                mask = np.random.rand(*child.shape) < 0.5
                mixed[i] = np.where(mask, parent, child)
        return mixed


    def _get_directional_parents(self, replace_indices):
        """Helper: Selects elite parents based on direction alignment."""
        moea = self.qpop_net.moea_helper
        archive_pop = self.pareto_global_population
        archive_fits = self.pareto_global_fitnesses
        parent_map = self.qpop_net._parent_map
        
        norm_objs = moea._normalize_objectives_01(archive_fits)
        elite_parents = []
        
        for child_idx in replace_indices:
            q_idx = parent_map[child_idx]
            dir_idx = moea._ind_to_dir[q_idx]
            lam = moea._ref_dirs[dir_idx]
            scores = moea._score_weighted_sum(norm_objs, lam)
            best_idx = np.argmax(scores)
            elite_parents.append(archive_pop[best_idx])
            
        return np.array(elite_parents)

    def _get_random_elite_parents(self, num_off):
        """Helper: Selects elite parents randomly from the global archive."""
        archive_pop = self.pareto_global_population
        indices = np.random.choice(len(archive_pop), size=num_off, replace=True)
        return archive_pop[indices]

    def crossover_network(self, new_pop_net: np.ndarray) -> np.ndarray:
        """
        Applies crossover to the network population.
        Strategy depends on self.mo_crossover_strategy ('directional' or 'random_elite').
        
        Directional Crossover.
        
            Instead of crossing with the generic 'best', this method:
            1. Identifies the Quantum Parent for each child using `qpop_net._parent_map`.
            2. Retrieves the Reference Direction assigned to that Quantum Parent.
            3. Selects the Elite from the Global Pareto Archive that is best aligned with that direction.
            4. Uses that Directional Elite as the crossover partner.
            5. Delegates the actual gene swapping to `helpers.operators.apply_crossover`.
        
        """
        # 1. Standard checks
        if not (self.current_gen > 0 and getattr(self, "en_pop_crossover", False)):
            return new_pop_net
        
        if self.current_gen % self.crossover_frequency != 0:
            return new_pop_net

        # 2. Determine number of individuals to replace
        num_off = int(len(new_pop_net) * self.pop_crossover_rate)
        if num_off <= 0:
            return new_pop_net

        # 3. Check archive
        if self.pareto_global_population is None or len(self.pareto_global_population) == 0:
            self.logger.warning("Pareto global population empty. Skipping crossover.")
            return new_pop_net
            
        # 4. Select children to be replaced
        replace_indices = np.random.choice(len(new_pop_net), num_off, replace=False)
        children_selection = new_pop_net[replace_indices]

        # 5. Select Parents based on Strategy
        try:
            if self.mo_crossover_strategy == "directional":
                elite_parents = self._get_directional_parents(replace_indices)
            else:
                # Fallback to random elite (standard MOEA behavior)
                elite_parents = self._get_random_elite_parents(num_off)
                
            # 6. Apply Crossover
            offspring = apply_crossover(
                elite_parents,
                children_selection,
                method_keys=self.crossover_methods
            )
            
            # 7. Replace
            new_pop_net[replace_indices] = offspring
            self.logger.info(f"Crossover applied ({self.mo_crossover_strategy}). Replaced {num_off} individuals.")
            
        except Exception as e:
            self.logger.error(f"Crossover failed: {e}")
            
        return new_pop_net
    
    def record_and_save_history(self):
            """
            Records the current global Pareto front, calculates its hypervolume,
            and saves the entire history to a pickle file.
            """
            # 1) Build the record for the current generation from the global archive.
            gen_record = {1: []} # Storing the front in a key '1'
            
            for i in range(len(self.pareto_global_ids)):
                individual_data = {"id": self.pareto_global_ids[i]}
                for j, obj_name in enumerate(self.objectives):
                    individual_data[obj_name] = float(self.pareto_global_fitnesses[i][j])
                gen_record[1].append(individual_data)

            # 2) Calculate the hypervolume of the current global front.
            hv = compute_hypervolume_mixed(self.pareto_global_fitnesses, self.objective_senses)
            gen_record["hypervolume"] = float(hv)
            
            # 3) Add the record for this generation to the main history dictionary.
            self.fronts_history[self.current_gen] = gen_record
            
            # 4) Persist the entire history to disk.
            history_path = os.path.join(self.experiment_path, "pareto_history.pkl")
            with open(history_path, "wb") as f:
                pickle.dump(self.fronts_history, f)

    def update_global_pareto_front(self):
        """
        Update the global Pareto archive by merging it with the current population
        and finding the new set of non-dominated solutions.

        This method relies on `self.classical_ids` containing the correct,
        persistent IDs for the individuals in the current population.
        """
        # Use the persistent IDs managed by the `evolve` loop.
        curr_ids = self.classical_ids

        if self.pareto_global_population is None:
            # If the archive is empty, initialize it with the current population.
            all_pop = self.classical_nets.copy()
            all_fits = self.fits.copy()
            all_params = self.classical_params.copy()
            all_ids = curr_ids.copy()
        else:
            # Otherwise, combine the existing archive with the current population.
            all_pop = np.vstack([self.pareto_global_population, self.classical_nets])
            all_fits = np.vstack([self.pareto_global_fitnesses, self.fits])
            all_params = np.vstack([self.pareto_global_params, self.classical_params])
            all_ids = self.pareto_global_ids + curr_ids

        unique_ids, unique_indices = np.unique(all_ids, return_index=True)
        # Filter the combined populations to keep only the first occurrence of each individual.
        unique_pop = all_pop[unique_indices]
        unique_fits = all_fits[unique_indices]
        unique_params = all_params[unique_indices]
        unique_ids = list(unique_ids) # Convert back to a list
        
        # 1) Perform a full non-dominated sort on the combined set.
        fronts = fast_nondominated_sort(unique_fits, self.objective_senses)
        
        # 2) The new global Pareto front consists of all individuals in the first front.
        idx0 = fronts[0]
        
        # 3) Update the global archive class attributes.
        self.pareto_global_population  = unique_pop[idx0]
        self.pareto_global_fitnesses   = unique_fits[idx0]
        self.pareto_global_params      = unique_params[idx0]
        self.pareto_global_ids         = [unique_ids[i] for i in idx0]
        
        # 4) Compute crowding distance on the final, updated global front.
        #    This is stored for the quantum update logic in `go_next_gen`.
        self._last_cd = crowding_distance(self.pareto_global_fitnesses,
            list(range(len(self.pareto_global_fitnesses))))
    
    def go_next_gen(self):
        """
        Orchestrates end-of-generation tasks: updating the global archive,
        recording history, updating quantum populations, and cleaning up.
        """
        # 1. Update the global Pareto archive with the latest population's results.
        # This method now updates self.pareto_global_* attributes directly and
        # calculates and stores the crowding distance of the new front in self._last_cd.
        self.update_global_pareto_front()

        # 2. Record the history of the updated global front and save it to disk.
        self.record_and_save_history()

        # 3. Select a diverse subset from the global front to update the quantum populations.
        # We use the crowding distance that was calculated and stored in the previous step.
        cd = self._last_cd
        sorted_rel = np.argsort(cd)[::-1]
        # pick = sorted_rel[:self.qpop_net.num_ind]

        # 4. Set the chosen individuals as the 'parents' for the quantum update.
        self.qpop_net.current_pop = self.pareto_global_population[sorted_rel]
        self.qpop_net.current_pop_objs = self.pareto_global_fitnesses[sorted_rel]
        
        if self.pareto_global_params is not None:
            self.qpop_params.current_pop = self.pareto_global_params[sorted_rel]

        # 5. Trigger the quantum population update (the learning step).
        self.update_quantum(self.current_gen)

        # 6. Log a summary of the generation's results.
        hv = self.fronts_history[self.current_gen]['hypervolume']
        self.logger.info("Generation %d: updated global Pareto front with %d individuals and hypervolume %.2f",
            self.current_gen,
            len(self.pareto_global_population),
            hv,
        )
        display_ids = [str(item) for item in self.pareto_global_ids]
        self.logger.info("Generation %d: current global Pareto IDs:\n%s",
            self.current_gen,
            display_ids,
        )

        fitness_str = np.array2string(
            self.pareto_global_fitnesses,
            separator='  ',
            formatter={'float_kind': lambda x: f"{x:.3f}"}
        )

        self.logger.info(
            "Generation %d global Pareto fitness:\n%s (n=%d)",
            self.current_gen,
            fitness_str,
            len(self.pareto_global_population),
        )
        is_snapshot = (self.current_gen % 5 == 0) and (self.current_gen > 0)
        # 7. Clean up old model directories, keeping only those in the global archive.
        # Gen 0 artifacts must be moved first so they are in archive before the symlink
        # created by the current-gen call (gen 1 is the first go_next_gen call).
        if self.current_gen == 1:
            delete_old_dirs_v2(self.experiment_path, 0, keep_ids=self.pareto_global_ids.copy())
        delete_old_dirs_v2(self.experiment_path, self.current_gen,
                        keep_ids=self.pareto_global_ids.copy(), is_snapshot_gen=is_snapshot)

        # 8. Save other necessary data from the parent class and advance the generation counter.
        self.save_data()
        # Generation boundary: archive + quantum update + save_data complete,
        # generation g+1 has consumed no randomness yet (Area 6).
        save_checkpoint(self)
        self.current_gen += 1

    # ---- Checkpoint hooks: extend QNAS state with the external archive ----

    def _checkpoint_state(self) -> dict:
        s = super()._checkpoint_state()
        s['moqnas'] = {
            'classical_nets': self.classical_nets,
            'classical_params': self.classical_params,
            'classical_ids': list(self.classical_ids),
            'fits': self.fits,
            'raw_fits': self.raw_fits,
            'pareto_global_population': self.pareto_global_population,
            'pareto_global_fitnesses': self.pareto_global_fitnesses,
            'pareto_global_params': self.pareto_global_params,
            'pareto_global_ids': list(self.pareto_global_ids),
            'fronts_history': self.fronts_history,
        }
        return s

    def _restore_state(self, s: dict) -> None:
        super()._restore_state(s)
        m = s['moqnas']
        self.classical_nets = m['classical_nets']
        self.classical_params = m['classical_params']
        self.classical_ids = m['classical_ids']
        self.fits = m['fits']
        self.raw_fits = m['raw_fits']
        self.pareto_global_population = m['pareto_global_population']
        self.pareto_global_fitnesses = m['pareto_global_fitnesses']
        self.pareto_global_params = m['pareto_global_params']
        self.pareto_global_ids = m['pareto_global_ids']
        self.fronts_history = m['fronts_history']

    def evolve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Run MoQNAS for max_generations, maintaining a global Pareto archive.

        Workflow per generation:
        1. Generation 0:
                - (p0_params, p0_nets) = self.generate_classical()
                - f0 = self.multiobjective_fitness()
                - assign fits, raw_fits; record best_so_far
                - DO NOT call go_next_gen for gen=0
        2. For gen in [1..max_generations]:
                a) self.current_gen = gen
                b) children_params, children_nets = self.generate_classical()
                c) child_fits = self.multiobjective_fitness()
                d) Combine parents + children and run NSGA‐II
                e) Assign survivors → self.classical_nets, self.fits, self.raw_fits
                f) Resample hyperparams for next gen: children_params already from generate_classical()
                h) Call self.go_next_gen()    # handles global Pareto archiving, cleanup, logging, ++gen
                i) Early stop if check_early_stopping()
                j) Prepare (p0_params, p0_nets, f0, p0_raws) for next iteration
        3. Return final global Pareto archive
        """
        self._session_start = time.time()
        if not hasattr(self, '_elapsed_so_far'):
            self._elapsed_so_far = 0.0
        if getattr(self, '_resumed', False):
            # Resumed from a checkpoint: the survivors of the completed
            # generation g play the parent role; the loop enters at g+1.
            self.logger.info("Resuming evolution after completed generation %d.",
                             self.current_gen)
            p0_params = self.classical_params
            p0_nets = self.classical_nets
            f0 = self.fits
            p0_ids = self.classical_ids
            start_gen = self.current_gen + 1
        else:
            # 1) Generation 0: sample both hyperparams and nets via generate_classical()
            start_gen = 1
            p0_params, p0_nets = self.generate_classical()
            self.classical_params = p0_params
            self.classical_nets = p0_nets

            self.qpop_params.current_pop = p0_params
            self.qpop_net.current_pop    = p0_nets

            # Evaluate generation 0
            f0 = self.multiobjective_fitness()
            self.logger.info("Generation 0: fitnesses:\n%s", f0)

            p0_ids = [f"0_{i}" for i in range(len(p0_nets))]
            self.classical_ids = p0_ids # Initialize a new attribute to hold current IDs

            self.fits = f0
            self.raw_fits = self.raw_fits.copy()

            # Record best‐so‐far (by first objective)
            i0 = int(np.nanargmax(f0[:, 0]))
            self.best_so_far = float(f0[i0, 0])
            self.best_so_far_id = [0, i0]

        # Keep copies for parent‐combination in next loop
        p0_raws = self.raw_fits.copy()

        # 2) Main loop: generations start_gen..max_generations
        for gen in range(start_gen, self.max_generations + 1):
            self.current_gen = gen

            # 2a) Sample children classical population (both params and nets) at once
            children_params, children_nets = self.generate_classical()
            child_ids = [f"{self.current_gen}_{i}" for i in range(len(children_nets))]


            children_params = self.random_crossover_hyperparams(children_params)
            children_nets = self.crossover_network(children_nets)
            
            # 2b) Evaluate children on all objectives
            self.classical_params = children_params
            self.classical_nets = children_nets
            child_fits = self.multiobjective_fitness()
            child_raw = self.raw_fits.copy()

            # 2c) Combine parents + children
            combined_nets = np.vstack([p0_nets, children_nets])   # shape = (2*pop_size, net_dim)
            combined_fits = np.vstack([f0, child_fits])           # shape = (2*pop_size, n_obj)
            combined_raws = np.vstack([p0_raws, child_raw])       # shape = (2*pop_size, n_obj)
            combined_ids = p0_ids + child_ids                    # shape = (2*pop_size,)
            combined_params = np.vstack([p0_params, children_params])
            # 2d) NSGA‐II environmental selection
            next_nets, next_fits, survivor_idx = self.environmental_selection(combined_nets, combined_fits)

            # 2e) Assign survivors
            self.classical_nets = next_nets
            self.fits = next_fits
            self.raw_fits = combined_raws[survivor_idx]
            self.classical_ids = [combined_ids[i] for i in survivor_idx]
            self.classical_params = combined_params[survivor_idx]

            # 2h) Advance generation: update global Pareto, backup, save, log, cleanup, increment gen
            self.go_next_gen()

            # 2i) Early stopping
            if self.early_stopping and self.check_early_stopping():
                break

            # 2j) Prepare for next iteration
            p0_params = self.classical_params
            p0_nets = self.classical_nets
            f0 = self.fits
            p0_raws = self.raw_fits
            p0_ids = self.classical_ids
            if self.current_gen > 0 and (self.current_gen % 5 == 0):
                curr_time = time.time()
                h, m, est_h, est_m = calculate_time(
                    self._session_start, curr_time, self.current_gen, self.max_generations, end_evol=False
                )
                self.logger.info(
                    "Gen %d: elapsed %dh %dm; ETA %dh %dm",
                    self.current_gen, h, m, est_h, est_m,
                )
        save_pkl(self.unique_networks_path, self.unique_networks_db)
        total_seconds = self._elapsed_so_far + (time.time() - self._session_start)
        total_h, rem = divmod(total_seconds, 3600)
        total_m, _ = divmod(rem, 60)
        self.logger.info("Total evolution time: %d hours and %d minutes (all sessions)", int(total_h), int(total_m))

        # 3) Return final global Pareto archive
        return self.pareto_global_population, self.pareto_global_fitnesses