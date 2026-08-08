# In algorithms/qnas/metrics_logger.py

import os
import csv
import numpy as np

class QPopulationMetricsLogger:
    """Handles calculation and logging of Q-Population metrics."""

    def __init__(self, metrics_path: str):
        """
        Args:
            metrics_path (str): Full path to the output CSV file.
        """
        self.metrics_output = metrics_path
        os.makedirs(os.path.dirname(self.metrics_output), exist_ok=True)
        
        self.metrics = {
            "epoch": [], "update_idx": [], "quantum_update_rate": [],
            "max_update": [], "entropy_mean": [], "kl_mean": [],
            "frac_onehot_0p9": [],
        }
        self._last_P = None
        self._update_counter = 0
        self._last_dump_idx = 0

    def log_update(self, P: np.ndarray, update_rate: float, max_update: float, 
                   epoch_idx: int | None = None):
        """Logs key metrics about the quantum population's state after an update.

        Args:
            P (np.ndarray): The current probabilities array of the Q-Pop.
            update_rate (float): The quantum_update_rate used in this step.
            max_update (float): The max_update value used in this step.
            epoch_idx (int | None, optional): The current epoch index.
        """
        H_mean = float(self._entropy_rows(P).mean())
        if self._last_P is not None and self._last_P.shape == P.shape:
            KL_mean = float(self._kl_rows(
                P.reshape(-1, P.shape[-1]),
                self._last_P.reshape(-1, self._last_P.shape[-1])
            ).mean())
        else:
            KL_mean = float("nan")
        
        frac_oh = self._frac_onehot(P, thr=0.9)

        self.metrics["epoch"].append(int(epoch_idx) if epoch_idx is not None else len(self.metrics["epoch"]))
        self.metrics["update_idx"].append(self._update_counter)
        self.metrics["quantum_update_rate"].append(update_rate)
        self.metrics["max_update"].append(max_update)
        self.metrics["entropy_mean"].append(H_mean)
        self.metrics["kl_mean"].append(KL_mean)
        self.metrics["frac_onehot_0p9"].append(frac_oh)

        self._last_P = P.copy()
        self._update_counter += 1

    def save_metrics_csv(self, overwrite: bool = False):
        """Saves the logged metrics to a CSV file.

        Args:
            overwrite (bool, optional): If True, overwrites the file. Otherwise, appends.
        """
        cols = ["epoch", "update_idx", "quantum_update_rate", "max_update",
                "entropy_mean", "kl_mean", "frac_onehot_0p9"]
        total = len(self.metrics["update_idx"])
        start = 0 if overwrite or not os.path.exists(self.metrics_output) else self._last_dump_idx
        if start >= total:
            return

        mode = "w" if overwrite or not os.path.exists(self.metrics_output) else "a"
        write_header = (mode == "w")
        
        try:
            with open(self.metrics_output, mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                if write_header:
                    w.writeheader()
                for i in range(start, total):
                    row = {k: self.metrics[k][i] for k in cols}
                    w.writerow(row)
            self._last_dump_idx = total
        except (IOError, PermissionError) as e:
            print(f"[QPopulationMetricsLogger] Warning: Could not write metrics to {self.metrics_output}. Error: {e}")

    def _entropy_rows(self, P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Calculates the normalized entropy for each probability vector in P."""
        F = P.shape[-1]
        if F <= 1:
            return np.zeros_like(P[..., 0])
        P_clipped = np.clip(P, eps, 1.0)
        H = -(P_clipped * np.log(P_clipped)).sum(axis=-1) / np.log(F)
        return H

    def _kl_rows(self, P_new: np.ndarray, P_old: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Calculates the KL divergence between two sets of probability vectors."""
        P_new_clipped = np.clip(P_new, eps, 1.0)
        P_old_clipped = np.clip(P_old, eps, 1.0)
        return (P_new_clipped * (np.log(P_new_clipped) - np.log(P_old_clipped))).sum(axis=-1)

    def _frac_onehot(self, P: np.ndarray, thr: float = 0.9) -> float:
        """Calculates the fraction of probability vectors that are nearly one-hot."""
        if P.size == 0:
            return 0.0
        return float(np.mean(P.max(axis=-1) > thr))