# -*- coding: utf-8 -*-
import time
from contextlib import contextmanager
from typing import Tuple, Union, Sequence

import torch
import torch.nn as nn


def _normalize_input_shape(shape: Union[int, Sequence[int]]) -> Tuple[int, int, int, int]:
    """
    Ensure a 4D shape (N,C,H,W) from common inputs:
        - int -> (1,1,int,int)
        - (C,H,W) -> (1,C,H,W)
        - (N,C,H,W) -> unchanged
        - (H,W) -> (1,1,H,W)
        - (C,) -> (1,C,1,1)
    """
    if isinstance(shape, int):
        return (1, 1, shape, shape)
    if isinstance(shape, (list, tuple)):
        s = tuple(shape)
        if len(s) == 4:
            return s
        if len(s) == 3:
            C, H, W = s
            return (1, C, H, W)
        if len(s) == 2:
            H, W = s
            return (1, 1, H, W)
        if len(s) == 1:
            C = s[0]
            return (1, C, 1, 1)
    # conservative fallback
    return (1, 3, 32, 32)


@contextmanager
def _model_eval_no_grad(model: nn.Module):
    """Temporarily set model to eval + no_grad, then restore training state."""
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _to_device(data, device, dtype=None):
    """Move Tensors (or lists/tuples of Tensors) to device/dtype."""
    if isinstance(data, (list, tuple)):
        return type(data)(_to_device(d, device, dtype) for d in data)
    return data.to(device=device, dtype=(dtype or data.dtype))


def _check_device(data, device: str) -> bool:
    """Check that Tensors (or lists/tuples of Tensors) are on device."""
    if isinstance(data, (list, tuple)):
        return all(_check_device(d, device) for d in data)
    return data.device == torch.device(device)


def _conv_kernel_hw(kernel_size) -> Tuple[int, int]:
    if isinstance(kernel_size, tuple):
        if len(kernel_size) == 2:
            return kernel_size
        if len(kernel_size) == 1:
            return kernel_size[0], 1
        # 3D or other exotic cases → treat as (1,1) to avoid crashes
        return 1, 1
    # int -> square
    return kernel_size, kernel_size


def _safe_hw(shape: Sequence[int]) -> Tuple[int, int]:
    """Return (H,W) if possible; (L,1) for 1D; (1,1) otherwise."""
    if len(shape) >= 2:
        return shape[-2], shape[-1]
    if len(shape) == 1:
        return shape[-1], 1
    return 1, 1


class ModelMetrics:
    """
    Hardware/complexity metrics with robust dummy-input creation and
    per-layer FLOPs counting (Conv2d, Conv1d, Linear), returning 0 for unsupported layers.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Optional NVML init (for energy metrics)
        if 'cuda' in device and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                self.nvml_initialized = True
            except Exception as e:
                print(f"Error initializing NVML: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False

    def __del__(self):
        if hasattr(self, 'nvml_initialized') and self.nvml_initialized:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------
    # Helpers
    # ------------------------------
    def create_input(self, input_shape, batch_override: int = None) -> torch.Tensor:
        """
        Create a dummy input tensor on the correct device/dtype.
        input_shape may be (C,H,W) or (N,C,H,W) or int.
        If batch_override is provided, use that as N.
        """
        norm = list(_normalize_input_shape(input_shape))
        if batch_override is not None:
            norm[0] = int(batch_override)
        param = next(self.model.parameters())
        x = torch.randn(tuple(norm), device=param.device, dtype=param.dtype)
        return x

    # ------------------------------
    # Public metrics
    # ------------------------------
    def measure_inference_time(self, input_data_or_shape, warmup_runs: int = 10, measure_runs: int = 10) -> float:
        """
        Measure average inference time in microseconds.
        Accepts either a Tensor or a shape; generates a proper dummy input if needed.
        """
        # Prepare input
        if isinstance(input_data_or_shape, torch.Tensor):
            x = input_data_or_shape
            if not _check_device(x, self.device):
                x = _to_device(x, self.device)
        else:
            x = self.create_input(input_data_or_shape)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Warmup + measure
        times = []
        with _model_eval_no_grad(self.model):
            for _ in range(warmup_runs):
                _ = self.model(x)
            for _ in range(measure_runs):
                if 'cuda' in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                _ = self.model(x)
                if 'cuda' in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.time()
                times.append(t1 - t0)

        avg_us = (sum(times) / max(1, len(times))) * 1e6
        return float(avg_us)

    def measure_parameters(self) -> int:
        """Count total parameters."""
        return int(sum(p.numel() for p in self.model.parameters()))

    def measure_memory(self, input_shape) -> int:
        """
        Peak memory (bytes) on CUDA; returns 0 on CPU/no CUDA.
        """
        if 'cuda' not in self.device or not torch.cuda.is_available():
            return 0

        x = self.create_input(input_shape)
        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        torch.cuda.reset_peak_memory_stats(device=self.device)
        with _model_eval_no_grad(self.model):
            _ = self.model(x)
        mem = torch.cuda.max_memory_allocated(device=self.device)
        return int(mem)

    def measure_flops(self, input_shape) -> int:
        """
        Count FLOPs via forward hooks for supported layers (Conv2d, Conv1d, Linear).
        Returns 0 for unsupported layers to avoid crashes.
        """
        x = self.create_input(input_shape)
        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        def count_flops(module: nn.Module, inp, out) -> int:
            try:
                tin = inp[0] if isinstance(inp, (tuple, list)) else inp
                tout = out[0] if isinstance(out, (tuple, list)) else out
            except Exception:
                tin, tout = inp, out

            # Conv2d
            if isinstance(module, nn.Conv2d):
                kH, kW = _conv_kernel_hw(module.kernel_size)
                outH, outW = _safe_hw(tout.shape)
                groups = getattr(module, 'groups', 1)
                in_ch = module.in_channels
                out_ch = module.out_channels
                # MACs per position: (in_ch/groups) * out_ch * kH * kW
                # positions: outH * outW ; multiply by 2 (mul+add) to get FLOPs
                return int((in_ch // groups) * out_ch * kH * kW * outH * outW * 2)

            # Conv1d
            if isinstance(module, nn.Conv1d):
                kL = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                outL = tout.shape[-1] if tout is not None and hasattr(tout, "shape") and len(tout.shape) >= 2 else 1
                groups = getattr(module, 'groups', 1)
                in_ch = module.in_channels
                out_ch = module.out_channels
                return int((in_ch // groups) * out_ch * kL * outL * 2)

            # Linear
            if isinstance(module, nn.Linear):
                in_f = module.in_features
                out_f = module.out_features
                batch = tout.shape[0] if tout is not None and hasattr(tout, "shape") and len(tout.shape) >= 1 else 1
                return int(in_f * out_f * batch * 2)

            # Other layers not counted
            return 0

        total_flops = 0

        def hook(m, inp, out):
            nonlocal total_flops
            try:
                total_flops += count_flops(m, inp, out)
            except Exception:
                # Be conservative: ignore failures to avoid breaking training
                pass

        hooks = []
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                hooks.append(m.register_forward_hook(hook))

        with _model_eval_no_grad(self.model):
            _ = self.model(x)

        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        return int(total_flops)

    # ------------------------------
    # Optional extra metrics
    # ------------------------------
    def measure_madd(self, input_shape) -> int:
        """Rough Multiply-Add count; prefer measure_flops()."""
        x = self.create_input(input_shape)
        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        total_madd = 0
        current = x
        with _model_eval_no_grad(self.model):
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    # output dims per Conv2d formula
                    H_in, W_in = current.shape[-2], current.shape[-1]
                    stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
                    pad = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                    dil = module.dilation if isinstance(module.dilation, tuple) else (module.dilation, module.dilation)
                    kH, kW = _conv_kernel_hw(module.kernel_size)
                    out_h = int((H_in + 2 * pad[0] - dil[0] * (kH - 1) - 1) / stride[0] + 1)
                    out_w = int((W_in + 2 * pad[1] - dil[1] * (kW - 1) - 1) / stride[1] + 1)
                    madd = module.in_channels * module.out_channels * kH * kW * out_h * out_w
                    total_madd += int(madd)
                    current = torch.zeros((current.shape[0], module.out_channels, out_h, out_w), device=current.device, dtype=current.dtype)
                elif isinstance(module, nn.Linear):
                    madd = module.in_features * module.out_features
                    total_madd += int(madd)
                    current = torch.zeros((current.shape[0], module.out_features), device=current.device, dtype=current.dtype)
                else:
                    # skip
                    pass
        return int(total_madd)

    def measure_energy_consumption(self, input_data, warmup_runs=10, measure_runs=10):
        """
        Average power (W) and total energy (J) over measure_runs.
        Requires NVML on CUDA; returns (None, None) if not available.
        """
        if not self.nvml_initialized:
            print("NVML is not initialized. Cannot measure energy consumption.")
            return None, None

        # Prepare input
        if isinstance(input_data, torch.Tensor):
            x = input_data if _check_device(input_data, self.device) else _to_device(input_data, self.device)
        else:
            x = self.create_input(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        power_measurements = []
        total_time = 0.0
        with _model_eval_no_grad(self.model):
            # warmup
            for _ in range(warmup_runs):
                _ = self.model(x)

            # measure
            for _ in range(measure_runs):
                if 'cuda' in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()

                t0 = time.time()
                power_start = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                _ = self.model(x)
                if 'cuda' in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.time()
                power_end = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

                elapsed = t1 - t0
                total_time += elapsed
                avg_power = (power_start + power_end) / 2.0
                energy = avg_power * elapsed
                power_measurements.append(energy)

        total_energy = float(sum(power_measurements))
        average_power = float(total_energy / total_time) if total_time > 0 else 0.0
        return average_power, total_energy
