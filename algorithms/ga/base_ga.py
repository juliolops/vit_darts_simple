import os
import time
import datetime
import numpy as np
from pickle import dump, load, HIGHEST_PROTOCOL
from utils.helpers import delete_old_dirs_v2, init_log, calculate_time
from algorithms.checkpoint import save_checkpoint

#TODO: add docstrings to all functions
#TODO: use the utils functions for handling the folder structure
class GA(object):
    """
    Genetic Algorithm class for optimizing neural network architectures.
    It uses tournament selection, uniform crossover, and Gaussian mutation.
    The class supports early stopping based on fitness improvement.

    Attributes:
        population: Current population of individuals (chromosomes).
        current_population: Population of individuals in the current generation.
        fitnesses: Fitness values for the current population.
        best_so_far: Best fitness value found so far.
        best_so_far_id: ID of the best individual so far.
        current_gen: Current generation number.
        total_eval: Total evaluations performed.
        evaluated: Cache for previously evaluated individuals.
        eval_history: History of evaluations for each individual.
    """
    def __init__(self, eval_func, experiment_path, objectives, log_file, log_level, data_file, use_cache=False):
        """
        Initialize the GA evolution.

        Args:
            eval_func: Callable to evaluate a candidate architecture.
                Must have signature: eval_func(decoded_individual, generation)
            experiment_path: Path to folder for saving logs/models.
            log_file: Filename for log output.
            log_level: Logging level, e.g., "INFO", "DEBUG".
            data_file: Path to file for saving evolution data (pickle).
            use_cache (bool): If True, enables caching of evaluated individuals.
        """
        # GA parameters (to be set later via initialize_ga)
        self.fn_list = None
        self.params_ranges = None
        self.cont_mut_sigma = None
        self.last_best_so_far = None
        self.fixed_params = None
        self.cont_keys = None
        self.cont_max = None
        self.cont_min = None
        self.pop_params = None
        self.population_size = None
        self.num_generations = None
        self.crossover_rate = None
        self.mutation_rate = None
        self.elitism = False

        # GA state variables
        self.population = None
        self.current_population = None
        self.fitnesses = None
        self.best_so_far = -np.inf
        self.best_so_far_id = None
        self.current_gen = 0
        self.total_eval = 0
        
        self.eval_idx = None           # maps sorted rows -> original candidate_id
        self.current_best_rank = None  # position in the sorted array (for logging)

        # Early stopping parameters
        self.patience = None
        self.early_stopping_counter = 0
        self.early_stopping = None

        self.eval_func = eval_func
        self.experiment_path = experiment_path
        self.data_file = data_file
        self.objectives = objectives
        self.logger = init_log(log_level, name=__name__, file_path=log_file)
        # Caching now lives in core/eval_cache.py (wraps eval_func).
        self.use_cache = use_cache
        if self.use_cache:
            self.logger.warning(
                "The legacy per-algorithm evaluation cache was removed; pass "
                "--use_cache to enable the unified cache (core/eval_cache.py).")
        
    
    def initialize_ga(self, population_size, num_generations, max_num_nodes, fn_list, params_ranges,
                    crossover_rate, mutation_rate, early_stopping, elitism=False, patience=60, mutation_strategy="random"):
        """
        Initialize GA parameters and create the initial random population.

        Args:
            population_size: (int) number of individuals.
            num_generations: (int) maximum number of generations.
            max_num_nodes: (int) length of each individual chromosome.
            crossover_rate: (float) probability for crossover.
            mutation_rate: (float) probability for mutation (per gene).
            fn_list: List of function names (e.g., layer types) to decode chromosomes.
            elitism: (bool) whether to keep the best individual in the next generation.
            patience: (int) number of generations with little/no improvement to trigger early stopping.
            params_ranges: (dict) dictionary with min/max values for continuous parameters.
            mutation_strategy: (str) strategy for mutation ("swap", "block", "neighbor", "standard", "random").
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.patience = patience
        self.early_stopping = early_stopping
        self.fn_list = fn_list  # Store the list of function names
        self.params_ranges  = params_ranges

        # # Initialize a population of individuals.
        self.population = np.random.randint(0, len(self.fn_list), size=(population_size, max_num_nodes))
        
        # 1) split out the continuous‐valued params vs fixed ones
        self.cont_keys = [
            k for k, v in params_ranges.items()
            if isinstance(v, (list, tuple)) and len(v) == 2
        ]
        self.fixed_params = {
            k: v for k, v in params_ranges.items()
            if k not in self.cont_keys
        }
        # 2) build lower/upper arrays for the continuous dims
        if self.cont_keys:
            lowers = [params_ranges[k][0] for k in self.cont_keys]
            uppers = [params_ranges[k][1] for k in self.cont_keys]
            self.cont_min = np.array(lowers, dtype=float)
            self.cont_max = np.array(uppers, dtype=float)
        else:
            # no continuous dims
            self.cont_min = np.zeros((0,), dtype=float)
            self.cont_max = np.zeros((0,), dtype=float)
            
        # continuous pop: population_size x num_params
        num_params = len(self.cont_keys)
        if num_params > 0:
            self.pop_params = np.random.uniform(low=self.cont_min,high=self.cont_max,size=(population_size, num_params))
        else:
            self.pop_params = np.zeros((population_size, 0), dtype=float)

        # scale for Gaussian mutation on cont. genes
        self.cont_mut_sigma = 0.05
        # self.logger.info("Initial population created with size: %d", population_size)
        # self.logger.info("Population: %s", str(self.population))
        
        self.mutation_strategy = mutation_strategy  # <--- Store the preference

        # Define available strategies mapping
        self.mutation_map = {
            "swap": self.mutate_swap,
            "block": self.mutate_block,
            "neighbor": self.mutate_neighbor,
            "standard": self.mutate_gen,  # 'standard' as mutate_gen
            "random": None
        }
        
    def decode_net(self, chromosome):
        """
        Convert a chromosome (list or numpy array of integers) into the corresponding
        list of function names representing the network layers.

        Args:
            chromosome: list or numpy array, where each element is an integer index.

        Returns:
            List with function names in the order they represent the network.
        """
        # Using len(chromosome) if chromosome is a list; if it's a numpy array, chromosome.shape[0] works as well.
        decoded = [None] * len(chromosome)
        for i, gene in enumerate(chromosome):
            if gene >= 0:
                decoded[i] = self.fn_list[gene]
        return decoded
    
    def decode_pop(self):
        """ Decode a population of parameters and networks.

        Returns:
            list of decoded params and list of decoded networks.
        """

        num_individuals = self.population.shape[0]

        decoded_params = [None] * num_individuals
        decoded_nets = [None] * num_individuals
        #backbone_percentage_array = np.random.uniform(0.0, 1.0, size=(len(decoded_nets), 1))
        for i in range(num_individuals):
            decoded_nets[i] = self.decode_net(self.population[i, :])
            # build params dict:
            p = dict(self.fixed_params)  # start from fixed
            for j, key in enumerate(self.cont_keys):
                p[key] = float(self.pop_params[i,j])
            p['candidate_id'] = i
            decoded_params[i] = p

        #return decoded_params, decoded_nets
        return decoded_nets, decoded_params
    
    def order_population(self):
        """
        Order the population and corresponding fitnesses in descending order of fitness.
        This ensures that the best candidate is always first.
        """
        idx = np.argsort(self.fitnesses)[::-1]
        self.population = self.population[idx]
        self.pop_params = self.pop_params[idx]
        self.fitnesses = self.fitnesses[idx]
        if self.eval_idx is not None:
            self.eval_idx = self.eval_idx[idx]

    def _evaluate_without_cache(self, decoded_net, decoded_params):
        """Evaluates the entire population, handling dictionary-based results."""
        self.logger.info("Evaluating population of %d individuals without cache.", len(decoded_net))
        results = self.eval_func(decoded_params, decoded_net, generation=self.current_gen)

        if not results:
            return [None] * len(decoded_net)

        fitness_values = [None] * len(decoded_net)
        metric_key = self.objectives[0]

        # Correctly map results from the dictionary using candidate_id
        for i in range(len(decoded_net)):
            candidate_id = decoded_params[i]['candidate_id']
            if candidate_id in results:
                fitness_values[i] = results[candidate_id][metric_key]
        
        self.total_eval += len(self.population)
        return fitness_values

    def evaluate_population(self):
        """
        Evaluate the current population using the appropriate strategy (cached or not).
        """
        decoded_net, decoded_params = self.decode_pop()
        
        fitness_values = self._evaluate_without_cache(decoded_net, decoded_params)

        self.fitnesses = np.array(fitness_values, dtype=float)
        self.eval_idx = np.arange(self.population.shape[0], dtype=int)

        # Finalize generation results
        self.update_best_id(self.fitnesses, self.eval_idx)
        self.order_population()
        self.current_population = self.population.copy()
        self.log_data()
        return self.fitnesses

    def update_best_id(self, fitnesses_, eval_idx):
        """
        Update the best individual id based on current fitnesses_.
        Args:
            fitnesses_: numpy array of current population fitnesses_.
            eval_idx: numpy array of indices mapping sorted rows to original candidate_ids.
        """
        best_rank = int(np.nanargmax(fitnesses_))
        eval_id   = int(eval_idx[best_rank])      # original candidate_id
        best_fit  = float(fitnesses_[best_rank])

        self.current_best_rank = best_rank        # for logs
        if best_fit > self.best_so_far:
            self.best_so_far   = best_fit
            self.best_so_far_id = (self.current_gen, eval_id)

    def selection(self):
        """
        Select an individual from the population using tournament selection.
        Tournament size is set to 3 by default.
        """
        tournament_size = 3
        participants = np.random.choice(range(self.population_size), tournament_size, replace=False)
        best = participants[0]
        for idx in participants:
            if self.fitnesses[idx] > self.fitnesses[best]:
                best = idx
        return best

    @staticmethod
    def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) :
        """
        One-point crossover: a random crossover point is chosen.
        The offspring inherits genes from parent1 up to that point and parent2 for the remainder (and vice versa).
        """
        n = parent1.shape[0]
        # Ensure the crossover point is between 1 and n-1.
        point = np.random.randint(1, n)
        offspring1 = np.concatenate((parent1[:point], parent2[point:]))
        offspring2 = np.concatenate((parent2[:point], parent1[point:]))
        return offspring1, offspring2

    @staticmethod
    def two_point_crossover(parent1: np.ndarray, parent2: np.ndarray) :
        """
        Two-point crossover: Two random crossover points are selected.
        The offspring inherits the middle segment from one parent and the remaining segments from the other.
        """
        n = parent1.shape[0]
        point1, point2 = np.sort(np.random.choice(range(1, n), size=2, replace=False))
        offspring1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        offspring2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        return offspring1, offspring2

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) :
        """
        Select a crossover operator at random among available strategies.
        """
        crossover_strategies = [self.one_point_crossover, self.two_point_crossover]
        chosen_strategy = np.random.choice(crossover_strategies)
        return chosen_strategy(parent1, parent2)
    
    def mutate_swap(self, individual: np.ndarray) -> np.ndarray:
        """
        Swap mutation: with probability equal to mutation_rate,
        randomly pick two indices and swap their values.
        """
        if np.random.rand() < self.mutation_rate:
            n = individual.shape[0]
            indices = np.random.choice(n, size=2, replace=False)
            i, j = indices[0], indices[1]
            # Swap the two selected genes
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def mutate_block(self, individual: np.ndarray) -> np.ndarray:
        """
        Block mutation: with probability mutation_rate, select a contiguous block
        of genes and replace them with new random values drawn uniformly from [0, len(fn_list)-1].
        This operator uses slicing to assign the new block.
        """
        if np.random.rand() < self.mutation_rate:
            n = individual.shape[0]
            block_length = np.random.randint(1, max(2, n // 2 + 1))
            start_idx = np.random.randint(0, n - block_length + 1)
            # Replace the entire block with new random gene values
            individual[start_idx:start_idx+block_length] = np.random.randint(0, len(self.fn_list), size=block_length)
        return individual

    def mutate_neighbor(self, individual: np.ndarray) -> np.ndarray:
        """
        Neighbor mutation: adjust each gene with probability mutation_rate by adding or subtracting one.
        This is achieved using vectorized operations.
        """
        n = individual.shape[0]
        # Create a boolean mask for genes that will be mutated.
        mask = np.random.rand(n) < self.mutation_rate
        # For each gene, choose a delta of -1 or +1.
        deltas = np.random.choice([-1, 1], size=n)
        mutated = individual.copy()
        # Apply only to the genes indicated by the mask, clamping to valid range.
        mutated[mask] = np.clip(mutated[mask] + deltas[mask], 0, len(self.fn_list)-1)
        return mutated

    def mutate_gen(self, individual: np.ndarray) -> np.ndarray:
        """
        Gene-wise mutation: with probability mutation_rate per gene,
        replace that gene with a new randomly generated value.
        This implementation uses vectorized masking.
        """
        n = individual.shape[0]
        mask = np.random.rand(n) < self.mutation_rate
        new_values = np.random.randint(0, len(self.fn_list), size=n)
        mutated = individual.copy()
        mutated[mask] = new_values[mask]
        return mutated

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply mutation based on the configured strategy.
        """
        # 1. Check if we should use a specific fixed strategy
        if self.mutation_strategy != "random":
            if self.mutation_strategy in self.mutation_map:
                strategy_func = self.mutation_map[self.mutation_strategy]
                return strategy_func(individual)
            else:
                # Fallback or error logging
                self.logger.warning(f"Unknown mutation strategy '{self.mutation_strategy}'. Defaulting to standard.")
                return self.mutate_gen(individual)

        # 2. Original behavior: Randomly select one strategy
        mutation_strategies = [self.mutate_swap, self.mutate_block, self.mutate_neighbor, self.mutate_gen]
        strategy = np.random.choice(mutation_strategies)
        return strategy(individual)
    
    def _mutate_cont(self, cont_vector):
        """
        Gaussian mutation on each continuous gene with probability self.mutation_rate.
        """
        if cont_vector.size == 0:
            return cont_vector
        # which dims to mutate?
        mask = np.random.rand(*cont_vector.shape) < self.mutation_rate
        if mask.any():
            noise = np.random.normal(scale=self.cont_mut_sigma, size=cont_vector.shape)
            cont_vector = cont_vector + mask*noise
            # clip back to allowed bounds
            cont_vector = np.minimum(np.maximum(cont_vector, self.cont_min), self.cont_max)
        return cont_vector
    
    def generate_offspring(self):
        """
        Create a new population of size `self.population_size`, evolving both:
            - self.population      : categorical chromosomes (np.ndarray of ints)
            - self.pop_params      : continuous chromosome (np.ndarray of floats)
        using elitism, tournament selection, uniform crossover for the discrete part,
        and blend (arithmetic) crossover + Gaussian mutation for the continuous gene.
        """
        new_pop = []
        new_params = []

        # 1) Elitism: copy the best individual (both discrete + continuous)
        if self.elitism:
            best_idx = np.argmax(self.fitnesses)
            new_pop.append(self.population[best_idx].copy())
            new_params.append(self.pop_params[best_idx])

        # 2) Fill the rest by pairs
        while len(new_pop) < self.population_size:
            # parent selection returns indices
            i1 = self.selection()
            i2 = self.selection()

            p1, p2 = self.population[i1], self.population[i2]
            c1, c2 = self.pop_params[i1], self.pop_params[i2]

            # crossover
            if np.random.rand() < self.crossover_rate:
                # discrete: uniform mask crossover
                mask = np.random.rand(*p1.shape) < 0.5
                child1 = np.where(mask, p1, p2)
                child2 = np.where(mask, p2, p1)
                # continuous: arithmetic blend
                alpha = np.random.rand()
                child1_params = alpha * c1 + (1 - alpha) * c2
                child2_params = (1 - alpha) * c1 + alpha * c2
            else:
                child1, child2 = p1.copy(), p2.copy()
                child1_params, child2_params = c1, c2

            # mutation: discrete + continuous
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            child1_params = self._mutate_cont(np.array(child1_params))
            child2_params = self._mutate_cont(np.array(child2_params))

            new_pop.extend([child1, child2])
            new_params.extend([child1_params, child2_params])

        # 3) Trim to exact size and store back
        self.population = np.array(new_pop[:self.population_size])
        self.pop_params  = np.array(new_params[:self.population_size])

        self.logger.info(
            "New population created.\nDiscrete:\n%s\nContinuous:\n%s",
            self.population, self.pop_params
        )
    def log_data(self):
        """ Log GA evolution statistics such as generation number and fitnesses. """
        
        self.logger.info(
            f"[Gen {self.current_gen:03d}] Generation finished:\n"
            f"Best (orig id, rank): (g={int(self.best_so_far_id[0])}, "
            f"id={int(self.best_so_far_id[1])}, r={int(self.current_best_rank)}) "
            f"fitness: {float(self.best_so_far):.4f};\n"
            f"fitnesses: {self.fitnesses};"
        )

    def save_data(self):
        """
        Save current evolution data to self.data_file (pickle format) while
        preserving data from previous generations. The dictionary is updated
        so that if the generation already exists, its data is overwritten.
        """
        # If the file exists, load the current data; otherwise, start with an empty dictionary.
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = load(f)
        else:
            data = {}
        
        # Create the current generation's data dictionary.
        current_data = {
            'time': str(datetime.datetime.now()),
            'total_eval': self.total_eval,
            'best_so_far': self.best_so_far,
            'best_so_far_id': self.best_so_far_id,
            'fitnesses': self.fitnesses,
            'net_pop': self.current_population,
        }
        
        # Update (or add) the current generation in the data dictionary.
        data[self.current_gen] = current_data

        # Save the merged dictionary back to the file.
        with open(self.data_file, 'wb') as f:
            dump(data, f, protocol=HIGHEST_PROTOCOL)
    
    def check_early_stopping(self):
        """
        Compute the early stopping of the evolution. If the best fitness does not improve 
        by at least 0.005 (0.5%) for `patience` generations, the evolution stops.
        """
        if self.current_gen > 1:
            improvement = (self.best_so_far - self.last_best_so_far) / self.last_best_so_far
            if improvement > 0.005:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            self.logger.info(f"Early stopping counter: {self.early_stopping_counter}")
            if self.early_stopping_counter >= self.patience:
                self.logger.info(f"Early stopping at generation {self.current_gen}!")
                return True

        self.last_best_so_far = self.best_so_far
        return False

    def go_next_gen(self):
        """
        Perform end-of-generation routines: log data, save data, and update generation counter.
        """
        self.save_data()
        best_id_gen = self.best_so_far_id[0]
        best_id_idx = self.best_so_far_id[1]
        best_id = f"{best_id_gen}_{best_id_idx}"
        delete_old_dirs_v2(self.experiment_path,
                            self.current_gen, keep_ids=[best_id])
        # Generation boundary: state for gen g is final, g+1 not begun.
        save_checkpoint(self)
        self.current_gen += 1

    # ---- Checkpoint hooks (consumed by algorithms.checkpoint) ----

    def _checkpoint_config_block(self) -> dict:
        """Identity-defining config validated on resume.

        population_size and num_generations come from the CLI (not the
        config file) for the GA family, so they are part of the block to
        catch a resume launched with different values.
        """
        block = {
            'algorithm': type(self).__name__,
            'objectives': list(self.objectives),
            'population_size': int(self.population_size),
            'num_generations': int(self.num_generations),
        }
        block.update(getattr(self, 'checkpoint_extra', {}) or {})
        return block

    def _checkpoint_state(self) -> dict:
        """Resumable GA state: population + evolved continuous genes + fitness."""
        return {
            'population': self.population,
            'pop_params': self.pop_params,
            'fitnesses': self.fitnesses,
            'current_population': self.current_population,
        }

    def _restore_state(self, s: dict) -> None:
        self.population = s['population']
        self.pop_params = s['pop_params']
        self.fitnesses = s['fitnesses']
        self.current_population = s.get('current_population')

    def evolve(self):
        """
        Main evolution loop. Repeats until maximum generations is reached or early stopping is triggered.
        """
        self._session_start = time.time()
        if not hasattr(self, '_elapsed_so_far'):
            self._elapsed_so_far = 0.0
        self.logger.info("Starting evolution for %d generations", self.num_generations)

        if getattr(self, '_resumed', False):
            # Resumed: population/pop_params/fitnesses restored; current_gen is
            # the completed generation g, so continue the loop at g+1.
            self.logger.info("Resuming GA after completed generation %d.", self.current_gen)
            self.current_gen += 1

        while self.current_gen < self.num_generations:
            self.logger.info("Generation %d started - Evaluating...", self.current_gen)
            self.evaluate_population()
            self.generate_offspring()
            self.go_next_gen()
            if self.early_stopping and self.check_early_stopping():break

            if self.current_gen > 0 and (self.current_gen % 5 == 0):
                curr_time = time.time()
                h, m, est_h, est_m = calculate_time(
                    self._session_start, curr_time, self.current_gen, self.num_generations, end_evol=False
                )
                self.logger.info(
                    "Gen %d: elapsed %dh %dm; ETA %dh %dm",
                    self.current_gen, h, m, est_h, est_m,
                )

        total_seconds = self._elapsed_so_far + (time.time() - self._session_start)
        hours, rem = divmod(total_seconds, 3600)
        minutes, _ = divmod(rem, 60)
        self.logger.info("Total evolution time: %d hours and %d minutes (all sessions)", int(hours), int(minutes))
        self.logger.info("Best solution found at generation %d with fitness %.4f",
                        self.best_so_far_id[0] if self.best_so_far_id else -1, self.best_so_far)
        return self.population, self.fitnesses, self.best_so_far, self.best_so_far_id
