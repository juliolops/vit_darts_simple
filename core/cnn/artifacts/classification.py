import torch
from typing import List, Dict
from sklearn.metrics import confusion_matrix
from .base import BaseArtifact

class ConfusionMatrix(BaseArtifact):
    """
    Generates the confusion matrix for a classification task.

    This artifact accumulates predictions and ground truth labels over an entire
    evaluation run and computes the confusion matrix at the end. The matrix
    provides a detailed breakdown of correct and incorrect predictions for
    each class.
    """
    name = "confusion_matrix"

    def __init__(self):
        """Initializes the ConfusionMatrix artifact."""
        self.reset()

    def reset(self):
        """Clears the stored predictions and labels from previous runs."""
        self._predictions: List[int] = []
        self._labels: List[int] = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Accumulates the predictions and labels from a single batch.

        Args:
            outputs (torch.Tensor): The model's output logits.
            labels (torch.Tensor): The ground truth labels.
        """
        # Get the index of the max log-probability as the prediction
        _, predicted_classes = torch.max(outputs, 1)
        
        # Move tensors to CPU, convert to numpy arrays, and extend the lists
        self._predictions.extend(predicted_classes.cpu().numpy())
        self._labels.extend(labels.cpu().numpy())

    def compute(self) -> Dict[str, List[List[int]]]:
        """
        Computes the confusion matrix from all accumulated batches.

        Returns:
            Dict[str, List[List[int]]]: A dictionary containing the confusion
                                        matrix, serialized as a list of lists
                                        for compatibility with JSON/YAML.
        """
        if not self._labels:
            return {self.name: []}
        
        # Use scikit-learn to compute the confusion matrix
        cm = confusion_matrix(y_true=self._labels, y_pred=self._predictions)
        
        # Convert numpy array to a list of lists for easy serialization
        return {self.name: cm.tolist()}