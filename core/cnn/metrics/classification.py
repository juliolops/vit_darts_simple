import torch
from sklearn.metrics import confusion_matrix
from medmnist import Evaluator
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

class MedMNIST_Metrics(BaseMetric):
    """
    Computes dataset-specific metrics (AUC and Accuracy) for MedMNIST datasets.
    """
    name = "medmnist_metrics"

    def __init__(self, dataset_name: str, data_path: str, **kwargs):
        """
        Initializes the MedMNIST evaluator.

        Args:
            dataset_name (str): The name of the MedMNIST dataset (e.g., 'bloodmnist').
            data_path (str): The root path where the dataset is stored.
        """
        super().__init__(dataset_name=dataset_name, data_path=data_path, **kwargs)

        self.evaluator = Evaluator(dataset_name, 'test', root=data_path)
        self.y_scores = []

    def reset(self):
        """Clears the stored softmax scores."""
        self.y_scores = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Stores the softmax scores from a batch."""
        self.y_scores.append(outputs.softmax(dim=-1).cpu())

    def compute(self) -> dict:
        """
        Computes and returns the AUC and evaluator-specific accuracy.
        Returns a dictionary with 'auc_score' and 'acc_medmnist'.
        """
        if not self.y_scores:
            return {"auc_score": 0.0, "acc_medmnist": 0.0}
            
        y_score_cat = torch.cat(self.y_scores, dim=0).numpy()
        auc, acc = self.evaluator.evaluate(y_score_cat)
        return {"auc_score": auc, "acc_medmnist": acc}