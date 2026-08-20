from abc import ABC, abstractmethod
import torch

class BaseMetric(ABC):
    """
    An abstract base class for all metric implementations.

    This class defines the interface that the Trainer will use to interact with
    any metric, ensuring a consistent workflow for resetting, updating, and
    computing metric values.
    """
    def __init__(self, **kwargs):
        """
        Initializes the metric and stores all arguments in the `_init_args`
        dictionary to allow the Trainer to re-instantiate it.
        """
        self._init_args = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the primary name of the metric.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the internal state of the metric. This is called at the
        beginning of each evaluation epoch.
        """
        pass

    @abstractmethod
    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Updates the metric's state with the results from a single batch.

        Args:
            outputs (torch.Tensor): The model's raw output (logits) for the batch.
            labels (torch.Tensor): The ground truth labels for the batch.
        """
        pass

    @abstractmethod
    def compute(self) -> dict:
        """
        Computes the final metric value from its internal state.

        Returns:
            dict: A dictionary where keys are metric names and values are the
                computed results. A single metric can return multiple key-value pairs.
        """
        pass
    
    @property
    def is_post_processing(self) -> bool:
        """
        Returns True if the metric is expensive and should only be run once
        at the end of the training/evaluation process. Defaults to False.
        """
        return False