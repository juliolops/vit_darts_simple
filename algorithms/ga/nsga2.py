import os
import json
import time
import pickle
import numpy as np
from pymoo.indicators.hv import Hypervolume
from .base_ga import GA
from settings import CFG_OBJ_PATH
from algorithms.checkpoint import save_checkpoint
from algorithms.pareto import dominates, fast_nondominated_sort, crowding_distance, compute_hypervolume_mixed
from utils.helpers import delete_old_dirs_v2, calculate_time, load_pkl, save_pkl

class NSGA2(GA):
    """
    NSGA-II multi-objective genetic algorithm implementation.

    Extends a base GA to handle an arbitrary number of objectives (>=2). Implements:
        - Evaluation caching to avoid redundant fitness computations.
        - Fast non-dominated sorting with dynamic number of objectives.
        - Vectorized crowding distance calculation for diversity preservation.
        - Preallocated arrays for offspring generation and environmental selection.
        - Global Pareto front archive with crowding-based filtering.
        - History recording of all Pareto fronts per generation.
    """
    def __init__(self, eval_func, experiment_path, objectives, log_file, log_level, data_file, use_cache=False):
        """
        Initialize NSGA2 instance.

        Args:
            eval_func (callable): Function to evaluate individuals. Should accept
                (decoded_params_list, decoded_nets_list, generation) and return
                a list of fitness tuples.
            experiment_path (str): Base directory for logging and archives.
            log_file (str): File path for logging.
            log_level (int): Logging level.
            data_file (str): Path to initial data or checkpoint.
        """
        super().__init__(eval_func, experiment_path,objectives, log_file, log_level, data_file)
        # Caching now lives in core/eval_cache.py (wraps eval_func).
        self.use_cache = use_cache
        if self.use_cache:
            self.logger.warning(
                "The legacy per-algorithm evaluation cache was removed; pass "
                "--use_cache to enable the unified cache (core/eval_cache.py).")
        self.population_ids = None 
        self.pareto_global_population = None
        self.pareto_global_fitnesses = None
        self.pareto_global_ids = []
        self.fronts_history = {}
        self.objectives = objectives
        self.num_objectives = len(objectives) if objectives else None
        
        # load config objective sense
        try:
            with open(CFG_OBJ_PATH, "r") as f:
                self.objectives_info = json.load(f)["objectives"]
        except Exception as e:
            raise RuntimeError(f"Failed to load objective config: {e}")

        self.objective_names = []
        self.objective_senses = []
        # Ensure all specified objectives have corresponding info
        # Map the active objectives to their information
        for active_obj in self.objectives:
            match_found = False
            for key, info in self.objectives_info.items():
                if key in active_obj:
                    self.objective_names.append(key)
                    sense = 'max' if info['goal'] == 'maximize' else 'min'
                    self.objective_senses.append(sense)
                    match_found = True
                    break # Move to the next active_obj
            if not match_found:
                print(f"Warning: Could not find a rule for '{active_obj}'")
                
        self.unique_networks_path = os.path.join(self.experiment_path, "unique_networks.pkl")
        self.unique_networks_db = load_pkl(self.unique_networks_path) if os.path.exists(self.unique_networks_path) else {}
                

    def _evaluate_without_cache_with_registry(self, decoded_params, decoded_nets, pop):
        """
        Evaluate the full population without cache, while recording a persistent
        registry of unique evaluated networks.

        Notes
        -----
        - Every individual in the population is evaluated.
        - Each unique network is stored only once in `self.unique_networks_db`.
        - Repeated networks only update the `visits` counter.
        - The registry is persisted every 5 generations using `save_pkl(...)`.
        """
        num_individuals = len(decoded_nets)
        num_objectives = len(self.objectives)

        self.logger.info(
            "Evaluating population of %d individuals without cache.", num_individuals
        )

        results_dict = self.eval_func(
            decoded_params,
            decoded_nets,
            generation=self.current_gen,
        )

        if not results_dict:
            self.logger.warning("Evaluation function returned no results.")
            return np.full((num_individuals, num_objectives), np.nan, dtype=float)

        fits = np.full((num_individuals, num_objectives), np.nan, dtype=float)

        new_unique_count = 0
        repeated_count = 0

        for i in range(num_individuals):
            net_key = tuple(pop[i])

            if i not in results_dict:
                self.logger.warning(
                    "Individual at index %d was not found in results.", i
                )
                continue

            res_dict = results_dict[i]
            metric_vals = [float(res_dict.get(obj, np.nan)) for obj in self.objectives]
            fits[i] = metric_vals

            if net_key not in self.unique_networks_db:
                self.unique_networks_db[net_key] = {
                    "fitness": metric_vals,
                    "first_generation": self.current_gen,
                    "first_index": i,
                    "candidate_id": decoded_params[i].get("candidate_id"),
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

        return fits
    # def evaluate_population(self):
    #     """
    #     Evaluates the current population. Uses a cache if enabled, otherwise
    #     evaluates the entire population directly.
    #     """
    #     if self.use_cache:
    #         self.fitnesses = self._evaluate_with_cache()
    #     else:
    #         # Bypass the cache and evaluate the entire population
    #         decoded_nets, decoded_params = self.decode_pop()
            
    #         # eval_func returns a dictionary mapping index -> result_dict
    #         results_dict = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)
            
    #         N = len(decoded_params)
    #         fits = np.zeros((N, len(self.objectives)))
            
    #         for index, res_dict in results_dict.items():
    #             metric_vals = [res_dict[key] for key in self.objectives]
    #             fits[index] = metric_vals
                
    #         self.fitnesses = np.array(fits, dtype=float)
        
    #     return self.fitnesses
    
    def evaluate_population(self):
        """
        Evaluates the current population. Uses a cache if enabled, otherwise
        evaluates the entire population directly.
        """
        decoded_nets, decoded_params = self.decode_pop()
        self.fitnesses = self._evaluate_without_cache_with_registry(
            decoded_params=decoded_params,
            decoded_nets=decoded_nets,
            pop=self.population,
        )

        return self.fitnesses
    
    def environmental_selection(self, pop, fits):
        """
        Select the next generation population by Pareto rank and crowding.

        Fills new population front by front until population_size is reached,
        using crowding distance to select within a partial front.

        Args:
            pop (np.ndarray): Combined parent+offspring population.
            fits (np.ndarray): Corresponding fitness array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Selected population and fitness arrays.
        """
        pop_size = self.population_size
        M = fits.shape[1]
        fronts = fast_nondominated_sort(fits, self.objective_senses)
        # Preallocate result arrays
        new_pop = np.empty((pop_size, pop.shape[1]), dtype=pop.dtype)
        new_fits = np.empty((pop_size, M), dtype=float)
        survivor_indices = [] 
        count = 0
        for front in fronts:
            front_size = len(front)
            # If entire front fits
            if count + front_size <= pop_size:
                new_pop[count:count+front_size] = pop[front]
                new_fits[count:count+front_size] = fits[front]
                survivor_indices.extend(front)
                count += front_size
            else:
                rem = pop_size - count
                if rem > 0:
                    # Compute crowding distance and select top rem individuals
                    cd = crowding_distance(fits, front)
                    sel_indices = np.argsort(cd)[-rem:]
                    chosen = [front[i] for i in sel_indices]
                    new_pop[count:count+rem] = pop[chosen]
                    new_fits[count:count+rem] = fits[chosen]
                    survivor_indices.extend(chosen)
                # Either way, stop filling
                break
        return new_pop, new_fits, survivor_indices
    
    def tournament_select(self, pop, fits, rank, crowd):
        """
        Select one individual via binary tournament on rank and crowding distance.

        Args:
            pop (np.ndarray): population array.
            fits (np.ndarray): fitness array (not used here).
            rank (np.ndarray): Pareto rank per individual.
            crowd (np.ndarray): crowding distance per individual.

        Returns:
            np.ndarray: selected individual.
        """
        i, j = np.random.choice(len(pop), 2, replace=False)
        if rank[i] < rank[j]: return pop[i]
        if rank[j] < rank[i]: return pop[j]
        return pop[i] if crowd[i] > crowd[j] else pop[j]
    
    def generate_offspring(self):
        """
        Create offspring population using tournament selection, crossover, and mutation.

        Respects Pareto rank and crowding distance in tournament selection.
        Preallocates offspring array for performance.
        """
        pop_size = self.population_size
        gene_len = self.population.shape[1]
        new_pop = np.empty((pop_size, gene_len), dtype=self.population.dtype)
        fronts = fast_nondominated_sort(self.fitnesses, self.objective_senses)
        rank = np.empty(len(self.population), dtype=int)
        crowd = np.zeros(len(self.population))
        for r, fr in enumerate(fronts):
            cd = crowding_distance(self.fitnesses, fr)
            for i, idx in enumerate(fr):
                rank[idx] = r
                crowd[idx] = cd[i]
        i = 0
        while i < pop_size:
            p1 = self.tournament_select(self.population, self.fitnesses, rank, crowd)
            p2 = self.tournament_select(self.population, self.fitnesses, rank, crowd)
            if np.random.rand() < self.crossover_rate:
                c1, c2 = self.crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop[i] = self.mutate(c1)
            if i + 1 < pop_size:
                new_pop[i+1] = self.mutate(c2)
            i += 2
        self.population = new_pop

    def update_global_pareto_front(self):
        """
        Update the global Pareto buffer by combining history with current pop,
        computing non-dominated fronts, and filtering front 0 by crowding.

        Returns:
            tuple: (all_pop, all_fit, all_ids, fronts) for history recording.
        """
        curr_ids = self.population_ids
        if self.pareto_global_population is None:
            # If the archive is empty, initialize it with the current population.
            all_pop = self.population.copy() # In MOQNAS, self.classical_nets
            all_fits = self.fitnesses.copy() # In MOQNAS, self.fits
            all_params = self.classical_params.copy() if hasattr(self, 'classical_params') else None
            all_ids = curr_ids.copy()
        else:
            # Otherwise, combine the existing archive with the current population.
            all_pop = np.vstack([self.pareto_global_population, self.population]) # or self.classical_nets
            all_fits = np.vstack([self.pareto_global_fitnesses, self.fitnesses]) # or self.fits
            all_ids = self.pareto_global_ids + curr_ids
            if hasattr(self, 'classical_params') and self.pareto_global_params is not None:
                all_params = np.vstack([self.pareto_global_params, self.classical_params])
            else:
                all_params = None

        unique_ids, unique_indices = np.unique(all_ids, return_index=True)
        unique_pop = all_pop[unique_indices]
        unique_fits = all_fits[unique_indices]
        unique_params = all_params[unique_indices] if all_params is not None else None
        unique_ids = list(unique_ids) # Convert back to a list
        
        # 1) Perform a full non-dominated sort on the combined set.
        fronts = fast_nondominated_sort(unique_fits, self.objective_senses)
        
        # 2) The new global archive is the entire first front. No filtering is applied.
        idx0 = fronts[0]
        
        # 3) Update the global archive class attributes.
        self.pareto_global_population  = unique_pop[idx0]
        self.pareto_global_fitnesses   = unique_fits[idx0]
        self.pareto_global_ids         = [unique_ids[i] for i in idx0]
        if all_params is not None:
            self.pareto_global_params = unique_params[idx0]

        # 4) Compute crowding distance on the final global front for the *next* step's selection.
        self._last_cd = crowding_distance(
            self.pareto_global_fitnesses,
            list(range(len(self.pareto_global_fitnesses)))
        )

    def record_and_save_history(self):
        """
        Records the current global Pareto front, calculates its hypervolume,
        and saves the entire history to a pickle file.
        """
        # 1) Build the record for the current generation from the final global archive.
        gen_record = {1: []} # Key '1' represents the first (and only) front being saved.
        
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

    def go_next_gen(self):
        """
        Orchestrates end-of-generation tasks for NSGA-II: updating the global
        archive, recording history, logging, and cleaning up old directories.
        """
        # 1. Update the global Pareto archive.
        # This method handles updating the self.pareto_global_* attributes.
        self.update_global_pareto_front()

        # 2. Record the history of the updated global front and save it to a file.
        # This method now also calculates the hypervolume internally.
        self.record_and_save_history()

        # 3. Log a summary of the generation's results for monitoring.
        hv = self.fronts_history[self.current_gen]['hypervolume']
        self.logger.info(
            f"Gen {self.current_gen} → Hypervolume = {hv:.4f}, "
            f"Global Front Size = {len(self.pareto_global_population)}"
        )

        # 4. Clean up old model directories, keeping only those in the final global archive.
        is_snapshot = (self.current_gen % 5 == 0) and (self.current_gen > 0)
        delete_old_dirs_v2(self.experiment_path,self.current_gen,
                            keep_ids=self.pareto_global_ids.copy(),
                            is_snapshot_gen=is_snapshot,)
        if self.current_gen == 1:
            # On the first run, also clean up directories from generation 0.
            delete_old_dirs_v2(self.experiment_path, 0, keep_ids=self.pareto_global_ids.copy())

        # Generation boundary: archive + history saved, g+1 not begun.
        save_checkpoint(self)
        # 5. Advance the generation counter.
        self.current_gen += 1

    # ---- Checkpoint hooks: extend GA state with the external Pareto archive ----

    def _checkpoint_state(self) -> dict:
        s = super()._checkpoint_state()
        s.update({
            'population_ids': self.population_ids,
            'pareto_global_population': self.pareto_global_population,
            'pareto_global_fitnesses': self.pareto_global_fitnesses,
            'pareto_global_params': getattr(self, 'pareto_global_params', None),
            'pareto_global_ids': list(self.pareto_global_ids),
            'fronts_history': self.fronts_history,
        })
        return s

    def _restore_state(self, s: dict) -> None:
        super()._restore_state(s)
        self.population_ids = s['population_ids']
        self.pareto_global_population = s['pareto_global_population']
        self.pareto_global_fitnesses = s['pareto_global_fitnesses']
        self.pareto_global_params = s.get('pareto_global_params')
        self.pareto_global_ids = s['pareto_global_ids']
        self.fronts_history = s['fronts_history']

    def evolve(self):
        """
        Run the evolutionary process for num_generations.

        Workflow per generation:
            1. Generate and evaluate offspring.
            2. Combine parents and offspring for environmental selection.
            3. Archive and record Pareto fronts via go_next_gen.
            4. Optionally terminate via early stopping.

        Returns:
            tuple: (pareto_global_population, pareto_global_fitnesses)
        """
        self.logger.info("Starting NSGA-II evolution")
        start_time = time.time()

        if getattr(self, '_resumed', False):
            # Resumed: the survivors of the completed generation g (restored
            # into self.population/fitnesses/population_ids) play the parent
            # role; rebuild the loop-local parent vars and continue at g+1.
            self.logger.info("Resuming NSGA-II after completed generation %d.", self.current_gen)
            pop_old = self.population.copy()
            fits_old = self.fitnesses.copy()
            ids_old = self.population_ids.copy()
            self.current_gen += 1
        else:
            fits_old = self.evaluate_population()
            pop_old = self.population.copy()

            ids_old = [f"0_{i}" for i in range(len(pop_old))]
            self.population_ids = ids_old.copy()

            self.best_so_far = np.max(fits_old[:, 0])
            self.last_best_so_far = self.best_so_far
            self.current_gen = 1
        while self.current_gen < self.num_generations:
            self.generate_offspring()
            fits_new = self.evaluate_population()
            pop_new = self.population.copy()
            ids_new = [f"{self.current_gen}_{i}" for i in range(len(pop_new))]
            
            
            combined_pop = np.vstack([pop_old, self.population])
            combined_fits = np.vstack([fits_old, fits_new])
            combined_ids = ids_old + ids_new
            self.population, self.fitnesses, survivor_idx = self.environmental_selection(
                        combined_pop, combined_fits)
            
            self.population_ids = [combined_ids[i] for i in survivor_idx]
            
            self.go_next_gen()
            pop_old, fits_old, ids_old = self.population.copy(), self.fitnesses.copy(), self.population_ids.copy()
        
            if self.early_stopping and self.check_early_stopping():break
            
            if self.current_gen > 0 and (self.current_gen % 5 == 0):
                curr_time = time.time()
                h, m, est_h, est_m = calculate_time(
                    start_time, curr_time, self.current_gen, self.num_generations, end_evol=False
                )
                self.logger.info(
                    "Gen %d: elapsed %dh %dm; ETA %dh %dm",
                    self.current_gen, h, m, est_h, est_m,
                )

        save_pkl(self.unique_networks_path, self.unique_networks_db)
        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, _ = divmod(rem, 60)
        self.logger.info(f"Total evolution time: {hours} hours and {minutes} minutes")
        return self.pareto_global_population, self.pareto_global_fitnesses
