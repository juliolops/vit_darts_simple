import torch
from .base import BaseMetric

class Accuracy(BaseMetric):
    """
    Computes the classification accuracy.
    """
    name = "accuracy"

    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0

    def reset(self):
        """Resets the counters for correct predictions and total samples."""
        self.correct = 0
        self.total = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Updates the state with predictions and labels from a batch."""
        _, predicted = torch.max(outputs, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def compute(self) -> dict:
        """Computes and returns the final accuracy as a percentage."""
        if self.total == 0:
            return {self.name: 0.0}
        return {self.name: 100 * self.correct / self.total}