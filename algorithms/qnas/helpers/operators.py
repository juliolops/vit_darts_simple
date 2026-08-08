# In algorithms/qnas/helpers/operators.py

import numpy as np
from typing import List, Callable

def hux_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Performs Half-Uniform Crossover (HUX) on two parents.

    HUX identifies differing genes and swaps exactly half of them.

    Args:
        parent1 (np.ndarray): The first parent chromosome.
        parent2 (np.ndarray): The second parent chromosome.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the two offspring.
    """
    differing_indices = np.where(parent1 != parent2)[0]
    if len(differing_indices) == 0:
        # Parents are identical
        return parent1.copy(), parent2.copy()

    # Ensure at least one swap if they differ, otherwise // 2 can be 0
    num_swaps = max(1, len(differing_indices) // 2)
    swap_indices = np.random.choice(differing_indices, num_swaps, replace=False)
    offspring1, offspring2 = parent1.copy(), parent2.copy()
    offspring1[swap_indices], offspring2[swap_indices] = parent2[swap_indices], parent1[swap_indices]
    return offspring1, offspring2

def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Performs Uniform Crossover on two parents.

    A binary mask is created, and genes are swapped between parents
    where the mask is True.

    Args:
        parent1 (np.ndarray): The first parent chromosome.
        parent2 (np.ndarray): The second parent chromosome.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the two offspring.
    """
    chromosome_length = len(parent1)
    if chromosome_length == 0:
        return parent1.copy(), parent2.copy()
        
    crossover_mask = np.random.randint(0, 2, size=chromosome_length).astype(bool)
    offspring1, offspring2 = parent1.copy(), parent2.copy()
    offspring1[crossover_mask], offspring2[crossover_mask] = parent2[crossover_mask], parent1[crossover_mask]
    return offspring1, offspring2

# --- Start: Added from base_ga ---

def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    One-point crossover: a random crossover point is chosen.
    The offspring inherits genes from parent1 up to that point and parent2 for the remainder (and vice versa).
    """
    n = parent1.shape[0]
    if n <= 1:
        # Cannot perform one-point crossover on length 1 or 0
        return parent1.copy(), parent2.copy()
        
    # Ensure the crossover point is between 1 and n-1.
    point = np.random.randint(1, n)
    offspring1 = np.concatenate((parent1[:point], parent2[point:]))
    offspring2 = np.concatenate((parent2[:point], parent1[point:]))
    return offspring1, offspring2

def two_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-point crossover: Two random crossover points are selected.
    The offspring inherits the middle segment from one parent and the remaining segments from the other.
    """
    n = parent1.shape[0]
    if n <= 2:
        # Fallback to one-point if chromosome is too short
        return one_point_crossover(parent1, parent2)

    point1, point2 = np.sort(np.random.choice(range(1, n), size=2, replace=False))
    offspring1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    offspring2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
    return offspring1, offspring2

# --- End: Added from base_ga ---


# --- Start: New operator mapping ---

# A dictionary to map string keys to their corresponding functions
CROSSOVER_OPERATORS: dict[str, Callable] = {
    'hux': hux_crossover,
    'uniform': uniform_crossover,
    'one_point': one_point_crossover,
    'two_point': two_point_crossover
}

# --- End: New operator mapping ---


# --- Start: Modified apply_crossover ---

def apply_crossover(best_current_pop: np.ndarray, new_pop: np.ndarray, method_keys: List[str]) -> np.ndarray:
    """Applies a crossover method to generate offspring.

    It pairs individuals from `best_current_pop` and `new_pop` as parents.
    A crossover operator is randomly chosen from the list specified by
    `method_keys` for each parent pair.

    Args:
        best_current_pop (np.ndarray): Best individuals from the current population.
        new_pop (np.ndarray): Individuals from the newly generated population.
        method_keys (List[str]): A list of crossover operator keys to
                                randomly choose from (e.g., ['hux', 'one_point']).

    Returns:
        np.ndarray: A population of offspring.
    """
    
    # 1. Build the list of callable strategies from the provided keys
    try:
        available_strategies = [CROSSOVER_OPERATORS[key] for key in method_keys]
    except KeyError as e:
        raise ValueError(f"Unknown crossover method key: {e}. Available keys are: {list(CROSSOVER_OPERATORS.keys())}")
    
    if not available_strategies:
        raise ValueError("The 'method_keys' list cannot be empty.")

    # 2. Generate offspring
    offspring = []
    for parent1, parent2 in zip(best_current_pop, new_pop):
        
        # Randomly choose a strategy from the allowed list
        chosen_strategy = np.random.choice(available_strategies)
        
        child1, child2 = chosen_strategy(parent1, parent2)
        offspring.extend([child1, child2])
        
    return np.array(offspring[:len(new_pop)])

# --- End: Modified apply_crossover ---