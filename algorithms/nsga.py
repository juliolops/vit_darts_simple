"""Multi-objective genetic algorithm over per-block pruning percentages.

Each individual is a vector of integer genes indexing ``percentages``. Every
generation:

1. parents are picked by binary tournament on Pareto rank (ties at random);
2. children come from one/two-point crossover and a randomly chosen mutation;
3. survivors of parents + children are chosen front by front, the last
   partial front by crowding distance (NSGA-II selection);
4. the global Pareto archive and its hypervolume are saved to
   ``<experiment_path>/pareto_history.pkl``.
"""
import os
import pickle

import numpy as np

from algorithms.pareto import compute_hypervolume, crowding_distance, fast_nondominated_sort
from core.utils import init_log

OBJECTIVE_SENSES = {'best_accuracy': 'max', 'total_flops': 'min'}


class NSGA:
    def __init__(self, eval_func, experiment_path, percentages, population_size,
                 num_generations, max_num_nodes, crossover_rate, mutation_rate,
                 log_file, log_level):
        self.eval_func = eval_func
        self.experiment_path = experiment_path
        self.percentages = list(percentages)
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.objectives = list(OBJECTIVE_SENSES)
        self.objective_senses = list(OBJECTIVE_SENSES.values())
        self.logger = init_log(log_level, name=__name__, file_path=log_file)

        self.population = np.random.randint(0, len(self.percentages),
                                            size=(population_size, max_num_nodes))
        self.fitnesses = None
        self.population_ids = None
        self.current_gen = 0
        self.pareto_global_population = None
        self.pareto_global_fitnesses = None
        self.pareto_global_ids = []
        self.fronts_history = {}

    # ---------- evaluation ----------

    def evaluate_population(self):
        decoded = [[self.percentages[g] for g in chrom] for chrom in self.population]
        results = self.eval_func(decoded, generation=self.current_gen)
        fits = np.full((len(decoded), len(self.objectives)), np.nan, dtype=float)
        for i in range(len(decoded)):
            if i in results:
                fits[i] = [float(results[i].get(obj, np.nan)) for obj in self.objectives]
            else:
                self.logger.warning("Individual %d was not found in results.", i)
        self.fitnesses = fits
        return fits

    # ---------- variation ----------

    @staticmethod
    def one_point_crossover(parent1, parent2):
        n = parent1.shape[0]
        point = np.random.randint(1, n)
        return (np.concatenate((parent1[:point], parent2[point:])),
                np.concatenate((parent2[:point], parent1[point:])))

    @staticmethod
    def two_point_crossover(parent1, parent2):
        n = parent1.shape[0]
        point1, point2 = np.sort(np.random.choice(range(1, n), size=2, replace=False))
        return (np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:])),
                np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:])))

    def crossover(self, parent1, parent2):
        strategy = np.random.choice([self.one_point_crossover, self.two_point_crossover])
        return strategy(parent1, parent2)

    def mutate_swap(self, individual):
        """With probability ``mutation_rate``, swap two random genes."""
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(individual.shape[0], size=2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def mutate_block(self, individual):
        """With probability ``mutation_rate``, redraw a random contiguous block of genes."""
        if np.random.rand() < self.mutation_rate:
            n = individual.shape[0]
            block_length = np.random.randint(1, max(2, n // 2 + 1))
            start = np.random.randint(0, n - block_length + 1)
            individual[start:start + block_length] = np.random.randint(
                0, len(self.percentages), size=block_length)
        return individual

    def mutate_neighbor(self, individual):
        """Each gene moves one step (+/-10 points) with probability ``mutation_rate``."""
        n = individual.shape[0]
        mask = np.random.rand(n) < self.mutation_rate
        deltas = np.random.choice([-1, 1], size=n)
        mutated = individual.copy()
        mutated[mask] = np.clip(mutated[mask] + deltas[mask], 0, len(self.percentages) - 1)
        return mutated

    def mutate_gene(self, individual):
        """Each gene is redrawn with probability ``mutation_rate``."""
        n = individual.shape[0]
        mask = np.random.rand(n) < self.mutation_rate
        new_values = np.random.randint(0, len(self.percentages), size=n)
        mutated = individual.copy()
        mutated[mask] = new_values[mask]
        return mutated

    def mutate(self, individual):
        strategy = np.random.choice([self.mutate_swap, self.mutate_block,
                                     self.mutate_neighbor, self.mutate_gene])
        return strategy(individual)

    def _tournament_rank_only(self, rank):
        i, j = np.random.choice(len(rank), 2, replace=False)
        if rank[i] < rank[j]:
            return self.population[i]
        if rank[j] < rank[i]:
            return self.population[j]
        return self.population[i] if np.random.rand() < 0.5 else self.population[j]

    def generate_offspring(self):
        pop_size = self.population_size
        new_pop = np.empty((pop_size, self.population.shape[1]), dtype=self.population.dtype)

        rank = np.empty(len(self.population), dtype=int)
        for r, front in enumerate(fast_nondominated_sort(self.fitnesses, self.objective_senses)):
            for idx in front:
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
                new_pop[i + 1] = self.mutate(c2)
            i += 2
        self.population = new_pop

    # ---------- selection & archive ----------

    def environmental_selection(self, pop, fits):
        """Fill ``population_size`` slots front by front; the last partial front by crowding."""
        pop_size = self.population_size
        new_pop = np.empty((pop_size, pop.shape[1]), dtype=pop.dtype)
        new_fits = np.empty((pop_size, fits.shape[1]), dtype=float)
        survivors = []
        count = 0
        for front in fast_nondominated_sort(fits, self.objective_senses):
            if count + len(front) <= pop_size:
                new_pop[count:count + len(front)] = pop[front]
                new_fits[count:count + len(front)] = fits[front]
                survivors.extend(front)
                count += len(front)
            else:
                rem = pop_size - count
                if rem > 0:
                    cd = crowding_distance(fits, front)
                    chosen = [front[i] for i in np.argsort(cd)[-rem:]]
                    new_pop[count:count + rem] = pop[chosen]
                    new_fits[count:count + rem] = fits[chosen]
                    survivors.extend(chosen)
                break
        return new_pop, new_fits, survivors

    def update_global_pareto_front(self):
        """Archive = Pareto front of (previous archive + current population), deduplicated by id."""
        if self.pareto_global_population is None:
            all_pop = self.population.copy()
            all_fits = self.fitnesses.copy()
            all_ids = list(self.population_ids)
        else:
            all_pop = np.vstack([self.pareto_global_population, self.population])
            all_fits = np.vstack([self.pareto_global_fitnesses, self.fitnesses])
            all_ids = self.pareto_global_ids + self.population_ids

        unique_ids, unique_indices = np.unique(all_ids, return_index=True)
        unique_pop = all_pop[unique_indices]
        unique_fits = all_fits[unique_indices]
        front0 = fast_nondominated_sort(unique_fits, self.objective_senses)[0]

        self.pareto_global_population = unique_pop[front0]
        self.pareto_global_fitnesses = unique_fits[front0]
        self.pareto_global_ids = [unique_ids[i] for i in front0]

    def record_and_save_history(self):
        """Append this generation's archive and hypervolume to ``pareto_history.pkl``.

        Format: ``{generation: {1: [{'id', 'best_accuracy', 'total_flops'}, ...],
        'hypervolume': float}}``.
        """
        record = {1: [{'id': self.pareto_global_ids[i],
                       **{obj: float(self.pareto_global_fitnesses[i][j])
                          for j, obj in enumerate(self.objectives)}}
                      for i in range(len(self.pareto_global_ids))]}
        record['hypervolume'] = compute_hypervolume(self.pareto_global_fitnesses,
                                                    self.objective_senses)
        self.fronts_history[self.current_gen] = record
        with open(os.path.join(self.experiment_path, 'pareto_history.pkl'), 'wb') as f:
            pickle.dump(self.fronts_history, f)

    # ---------- main loop ----------

    def evolve(self):
        self.logger.info("Starting evolution")
        fits_old = self.evaluate_population()
        pop_old = self.population.copy()
        ids_old = [f"0_{i}" for i in range(len(pop_old))]
        self.population_ids = ids_old.copy()
        self.current_gen = 1

        while self.current_gen < self.num_generations:
            self.generate_offspring()
            fits_new = self.evaluate_population()
            ids_new = [f"{self.current_gen}_{i}" for i in range(len(self.population))]

            combined_pop = np.vstack([pop_old, self.population])
            combined_fits = np.vstack([fits_old, fits_new])
            combined_ids = ids_old + ids_new
            self.population, self.fitnesses, survivors = self.environmental_selection(
                combined_pop, combined_fits)
            self.population_ids = [combined_ids[i] for i in survivors]

            self.update_global_pareto_front()
            self.record_and_save_history()
            self.logger.info(
                f"Gen {self.current_gen} -> Hypervolume = "
                f"{self.fronts_history[self.current_gen]['hypervolume']:.4f}, "
                f"Global Front Size = {len(self.pareto_global_population)}")

            self.current_gen += 1
            pop_old, fits_old, ids_old = (self.population.copy(), self.fitnesses.copy(),
                                          self.population_ids.copy())

        return self.pareto_global_population, self.pareto_global_fitnesses
