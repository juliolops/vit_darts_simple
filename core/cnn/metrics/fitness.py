import torch
from typing import Dict
from .base import BaseMetric

class ScalarizedFitness(BaseMetric):
    """
    Computes a single, scalarized fitness value from multiple metric results.

    This metric is a post-processor that takes the final results from other
    metrics (like accuracy, loss, and hardware stats) and combines them into a
    weighted fitness score, ideal for single-objective NAS algorithms.
    It does not accumulate data per batch; its logic is executed once at the
    end of an evaluation epoch.
    """
    name = "scalarized_fitness"

    def __init__(self, metric_type: str, max_params: float = None,
                 max_inference_time: float = None, normalizers: Dict = None, **kwargs):
        """
        Initializes the scalarized fitness calculator.

        Args:
            metric_type (str): The primary metric to use ('accuracy' or 'loss').
            max_params (float, optional): Legacy threshold for the maximum
                number of parameters. Translated to
                ``normalizers['total_params']``.
            max_inference_time (float, optional): Legacy threshold for the
                maximum inference time. Translated to
                ``normalizers['cuda_inference_time']``.
            normalizers (Dict, optional): Generic mapping
                ``{objective_name: max_value}``; each entry contributes one
                penalty factor to the scalar (see ``_mofitness``). Use this
                for non-default objective sets, e.g.
                ``{'total_flops': 5.0e8}``. Mutually additive with the legacy
                arguments; an explicit ``normalizers`` entry wins on key clash.
        """
        super().__init__(metric_type=metric_type, max_params=max_params,
                         max_inference_time=max_inference_time,
                         normalizers=normalizers, **kwargs)
        self.metric_type = metric_type
        # Back-compat translation: legacy keys first, in the historical
        # factor order (params, then inference time), so old configs produce
        # the bit-identical scalar.
        merged = {}
        if max_params is not None:
            merged['total_params'] = max_params
        if max_inference_time is not None:
            merged['cuda_inference_time'] = max_inference_time
        merged.update(normalizers or {})
        self.normalizers = merged

    def reset(self):
        """This metric is stateless across batches, so reset does nothing."""
        pass

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """This metric is stateless across batches, so update does nothing."""
        pass

    def compute(self, epoch_results: Dict) -> Dict:
        """
        Calculates the fitness value using results from other metrics.

        Args:
            epoch_results (Dict): A dictionary containing the computed values
                                from all other metrics for the current epoch.

        Returns:
            Dict: A dictionary containing the final 'scalar_multi_objective' score.
        """
        # 1. Extract required values from the results dictionary
        if self.metric_type == 'accuracy':
            metric_value = epoch_results.get('accuracy', 0.0)
        elif self.metric_type == 'loss':
            metric_value = epoch_results.get('loss', float('inf'))
        else:
            raise ValueError(f"Invalid metric_type: {self.metric_type}")

        secondary = {name: epoch_results.get(name, 0) for name in self.normalizers}

        # 2. Encapsulate the original mofitness logic
        return {
            "scalar_multi_objective": self._mofitness(metric_value, secondary)
        }

    def _mofitness(self, metric_value, secondary: Dict) -> float:
        """
        Encapsulates the original weighted fitness function logic.

        Each entry of ``secondary`` (objective value keyed by name) is turned
        into a penalty factor ``ratio ** w`` against its normalizer threshold
        T: ``ratio = value / T``; ``w = -0.01`` while ``value <= T`` (soft
        reward below budget), ``w = -1`` above it (hard penalty). The scalar
        is the primary fitness times the product of all factors, in
        normalizer insertion order (preserves the historical
        params-then-inference ordering for legacy configs).
        """
        # Handle the primary metric
        if self.metric_type == 'accuracy':
            primary_fitness = metric_value / 100.0 if metric_value > 1 else metric_value
        else: # 'loss'
            primary_fitness = 1 / (1 + metric_value)

        fitness_value = primary_fitness
        for name, threshold in self.normalizers.items():
            value = secondary[name]
            w = -0.01 if value <= threshold else -1
            ratio = value / threshold if threshold > 0 else float('inf')
            factor = ratio ** w if ratio > 0 else 0
            fitness_value = fitness_value * factor

        return fitness_value * 100.0
    
class ValidationLossFitness(BaseMetric):
    """
    Computes a fitness value directly from the validation loss.

    This metric transforms the validation loss into a fitness score,
    where a lower loss results in a higher fitness value.
    The result is scaled to a range of 0-100.
    """
    name = "fitness_val_loss"

    def __init__(self, **kwargs):
        """Initializes the validation loss fitness calculator."""
        super().__init__(**kwargs)

    def reset(self):
        """This metric is stateless across batches, so reset does nothing."""
        pass

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """This metric is stateless across batches, so update does nothing."""
        pass

    def compute(self, epoch_results: Dict) -> Dict:
        """
        Calculates the fitness value from the validation loss.

        Args:
            epoch_results (Dict): A dictionary containing the computed values
                                from all other metrics for the current epoch.

        Returns:
            Dict: A dictionary containing the 'fitness_val_loss' score.
        """
        validation_loss = epoch_results.get('loss', float('inf'))

        if validation_loss == float('inf'):
            fitness_value = 0.0
        else:
            # Transform loss to a fitness value (lower loss = higher fitness)
            fitness_value = (1 / (1 + validation_loss)) * 100.0

        return {
            self.name: fitness_value
        }