# -*- coding: utf-8 -*-
import torch
from .base import BaseMetric
from .base_hardware import ModelMetrics

class HardwareMetrics(BaseMetric):
    """
    Computes hardware and model complexity metrics once per evaluation.
    Includes inference time, parameter count, FLOPs, and memory usage (MiB).
    """
    name = "hardware_metrics"
    @property
    def is_post_processing(self) -> bool:
        return True
    def __init__(self, model: torch.nn.Module, device: str, input_shape, **kwargs):
        """
        Args:
            model (torch.nn.Module): Model to evaluate.
            device (str): 'cuda' or 'cpu'.
            input_shape (tuple|list|int): Shape of a single sample (e.g., (C,H,W) or (N,C,H,W) or just int).
        """
        # keep init args for cloning in trainer
        super().__init__(model=model, device=device, input_shape=input_shape, **kwargs)

        self.model_metrics = ModelMetrics(model, device=device)
        self.input_shape = input_shape
        self._results = {}

    def reset(self):
        """Reset cached results for this epoch."""
        self._results = {}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor, groups=None):
        """No-op: hardware metrics are computed once per eval/train epoch."""
        return

    def compute(self, epoch_results=None) -> dict:
        """
        Compute hardware metrics (cached per epoch).
        - cuda_inference_time: μs
        - total_params: count
        - total_flops: count (safe, returns 0 for unsupported layers)
        - model_memory_usage: MiB (0 on CPU/no CUDA)
        """
        if self._results:
            return self._results

        # Build a dummy batch on the right device/dtype (batch size=10)
        dummy_batch = self.model_metrics.create_input(self.input_shape, batch_override=10)

        # Measure metrics with robust fallbacks
        try:
            cuda_time_us = self.model_metrics.measure_inference_time(dummy_batch, warmup_runs=5, measure_runs=10)
        except Exception as e:
            print(f"[HardwareMetrics] inference time error: {e}")
            cuda_time_us = 0.0

        try:
            total_params = self.model_metrics.measure_parameters()
        except Exception as e:
            print(f"[HardwareMetrics] params error: {e}")
            total_params = 0

        try:
            total_flops = self.model_metrics.measure_flops(self.input_shape)
        except Exception as e:
            print(f"[HardwareMetrics] flops error: {e}")
            total_flops = 0

        try:
            mem_bytes = self.model_metrics.measure_memory(self.input_shape)
            mem_mib = mem_bytes / (1024 ** 2)
        except Exception as e:
            print(f"[HardwareMetrics] memory error: {e}")
            mem_mib = 0.0

        self._results = {
            "cuda_inference_time": float(cuda_time_us),
            "total_params": int(total_params),
            "total_flops": int(total_flops),
            "model_memory_usage": float(mem_mib),
        }
        return self._results
