import abc
import torch
from typing import Dict, Any

class BaseArtifact(abc.ABC):
    """
    Abstract base class for all artifacts.

    An artifact is a tool used for in-depth analysis of a model's performance,
    producing complex outputs like matrices, plots, or detailed reports, rather
    than a single scalar value.

    Each artifact must implement the `compute` method. The `update` and `reset`
    methods have default implementations but can be overridden if the artifact
    needs to accumulate state across batches.
    """
    name: str = "base_artifact"

    def __init__(self, **kwargs):
        """Initializes the artifact. Can be overridden for specific setup."""
        pass

    def reset(self):
        """
        Resets the internal state of the artifact.
        
        This is called at the beginning of each evaluation epoch to ensure that
        data from previous epochs does not interfere with the current one.
        """
        pass

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Updates the artifact's state with the results from a single batch.

        Args:
            outputs (torch.Tensor): The model's raw outputs (logits).
            labels (torch.Tensor): The ground truth labels.
        """
        pass

    @abc.abstractmethod
    def compute(self) -> Dict[str, Any]:
        """
        Computes the final artifact from the accumulated data.

        This method is called once at the end of an evaluation epoch.

        Returns:
            Dict[str, Any]: A dictionary where the key is the artifact's name
                            and the value is the computed result (e.g., a matrix,
                            a path to a saved plot, etc.).
        """
        raise NotImplementedError("Each artifact must implement the compute method.")