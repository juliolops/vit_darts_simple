""" Copyright (c) 2025, Diego Páez
* Licensed under the MIT license

- Trains and evaluates one candidate model for the evolutionary search.
- Uses a pluggable metrics system: the trainer is agnostic to which metrics
    are computed, so accuracy and the hardware metrics are just plugins.
"""


import os
import copy
import time
import torch
from torch.amp import GradScaler, autocast
from .metrics.base import BaseMetric
from utils.helpers import create_info_file, init_log
from core.precision import resolve_precision
from settings import TRAIN_TIMEOUT


class BaseTrainer:
    """
    Trains one candidate and reports the metrics the search optimizes.

    Handles the epoch loop, validation, best-checkpoint selection, mixed
    precision and the pluggable metrics.
    Args:
        model_instance (torch.nn.Module): The neural network model to be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        params (dict): Dictionary of training parameters and configuration.
        metrics (list of BaseMetric): List of metric instances to compute during training and evaluation.
    Attributes:
        model (torch.nn.Module): The model being trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        params (dict): Training parameters.
        device (torch.device): Device for computation (CPU or CUDA).
        scaler (torch.cuda.amp.GradScaler): Mixed precision scaler.
        best_accuracy (float): Best validation accuracy achieved.
        best_validation_loss (float): Best validation loss achieved.
        best_epoch (int): Epoch with the best validation accuracy.
        best_model_path (str): Path to save the best model checkpoint.
        logger (logging.Logger): Logger for training events.
    Methods:
        _forward_pass(inputs, labels):
            Performs a forward pass with optional mixed precision and computes the loss.
        train_epoch():
            Trains the model for one epoch and returns average loss and accuracy.
        evaluate(loader):
            Evaluates the model on a given data loader and returns average loss and accuracy.
        release_gpu_memory():
            Releases GPU memory by clearing the CUDA cache.
        should_evaluate(epoch, start_eval_epoch):
            Determines if evaluation should be performed at the current epoch.
        train(debug=False):
            Main training loop that manages training, validation, checkpointing, and metric computation.
    Notes:
        - Mixed precision via torch.amp (fp16/bf16); see core/precision.py.
        - Metrics are plugins, so the objective set is config-driven.
    """
    def __init__(self, model_instance, criterion, optimizer, train_loader, val_loader, test_loader,
                params: dict, metrics: list[BaseMetric]):
        self.model = model_instance.to(params['device'])
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.params = params
        self.device = torch.device(params['device'])

        # Precision policy, resolved once ('fp32' | 'fp16' | 'bf16').
        self.precision = resolve_precision(self.params)
        if (self.precision == 'bf16' and self.device.type == 'cuda'
                and not torch.cuda.is_bf16_supported()):
            raise RuntimeError(
                f"precision='bf16' was requested but CUDA device "
                f"'{torch.cuda.get_device_name(self.device)}' has no native "
                f"bfloat16 support. Use precision='fp16' or 'fp32'.")
        self._autocast_dtype = {'fp16': torch.float16,
                                'bf16': torch.bfloat16}.get(self.precision, torch.float16)
        self._autocast_enabled = self.precision in ('fp16', 'bf16')
        # GradScaler exists ONLY for fp16; bf16 has fp32's dynamic range and
        # needs no loss scaling, so bf16/fp32 take the plain backward path.
        self.scaler = GradScaler(self.device.type, enabled=True) if self.precision == 'fp16' else None
        # --- Pluggable Metrics System ---
        self.post_processing_metrics = [
            m for m in metrics if m.is_post_processing or 'epoch_results' in m.compute.__code__.co_varnames
        ]
        self.primary_metrics = [m for m in metrics if m not in self.post_processing_metrics]

        # Validation needs its own metric instances so the running state does
        # not mix with the training pass.
        self.val_primary_metrics = [m.__class__(**m._init_args) for m in self.primary_metrics]

        # --- State Tracking & Checkpointing ---
        self.best_accuracy = 0.0
        self.best_validation_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state_dict = None
        self.best_model_path = os.path.join(self.params['model_path'], 'best_model.pth')
        os.makedirs(self.params['model_path'], exist_ok=True)

        # Initialize the logger. logs/ is created here (not at import time) so
        # importing this module stays side-effect free.
        phase = self.params.get('phase', 'evolution')
        log_directory = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_directory, exist_ok=True)
        log_file = os.path.join(log_directory, f"{phase}.log")
        self.logger = init_log(log_level="INFO", name=__name__, file_path=log_file)

    def _forward_pass(self, inputs, labels):
        """
        Performs a forward pass through the model with the given inputs and labels.
        Moves the input data and labels to the appropriate device, adjusts label shape and type for multi-class tasks,
        and computes the model outputs and loss. Supports mixed precision inference if enabled in parameters.
        Args:
            inputs (torch.Tensor): Input data batch.
            labels (torch.Tensor): Corresponding labels for the input data.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the model outputs, computed loss, and processed labels.
        """
        # Move data to device
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        # Adjust labels for multi-class tasks
        task = self.params.get('task', 'classification')
        if task == 'multi-class':
            labels = labels.squeeze().long()
        
        # Run forward pass under the configured precision policy
        with autocast(self.device.type, dtype=self._autocast_dtype, enabled=self._autocast_enabled):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
        
        return outputs, loss, labels
    
    def _run_epoch(self, loader, is_training: bool, metric_set: list[BaseMetric]):
        """
        A generic method to run one epoch of training or evaluation.
        """
        self.model.train(is_training)
        
        for metric in metric_set:
            metric.reset()
        
        total_loss = 0.0
        total_examples = 0
        
        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for batch in loader:
                if len(batch) == 3:
                    inputs, labels, groups = batch
                elif len(batch) == 2:
                    inputs, labels = batch
                    groups = None
                else:
                    raise ValueError(f"DataLoader returned a batch with {len(batch)} elements, but expected 2 or 3.")
                outputs, loss, labels = self._forward_pass(inputs, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    if self.scaler is not None:  # fp16: scale, unscale before clipping
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:  # bf16 / fp32: no loss scaling, same clipping
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                bs = inputs.size(0) if hasattr(inputs, "size") else 1
                total_loss += loss.item() * bs
                total_examples += bs

                for metric in metric_set:
                    if 'groups' in metric.update.__code__.co_varnames:
                        metric.update(outputs.detach(), labels.detach(), groups.detach() if groups is not None else None)
                    else:
                        metric.update(outputs.detach(), labels.detach())

        epoch_results = {}
        for metric in metric_set:
            epoch_results.update(metric.compute())
        
        epoch_results['loss'] = total_loss / max(1, total_examples)
        
        # for metric in self.post_processing_metrics:
        #     epoch_results.update(metric.compute(epoch_results))
            
        return epoch_results

    def train(self, debug=False):
        """
        The main training loop, orchestrating epochs, evaluation, and final result aggregation.
        """
        max_epochs = self.params['max_epochs']
        epochs_to_eval = self.params['epochs_to_eval']
        start_eval_epoch = max_epochs - epochs_to_eval
        val_results = {}
        t0 = time.time()
        
        training_losses, training_accuracies = [], []
        validation_losses, validation_accuracies = [], []

        for epoch in range(1, max_epochs + 1):
            train_results = self._run_epoch(self.train_loader, is_training=True, metric_set=self.primary_metrics)
            training_losses.append(train_results.get('loss', 0))
            training_accuracies.append(train_results.get('accuracy', 0))

            timeout = int(self.params.get('train_timeout', TRAIN_TIMEOUT))
            if epoch < start_eval_epoch and (time.time() - t0) > timeout:
                self.logger.info("Timeout reached (%ds)", timeout)
                raise TimeoutError()
            
            if self.should_evaluate(epoch, start_eval_epoch):
                val_results = self._run_epoch(self.val_loader, is_training=False, metric_set=self.val_primary_metrics)
                validation_losses.append(val_results.get('loss', 0))
                validation_accuracies.append(val_results.get('accuracy', 0))
                
                current_accuracy = val_results.get('accuracy', 0.0)
                current_loss = val_results.get('loss', float('inf'))

                if current_accuracy > self.best_accuracy:
                    self.best_accuracy = current_accuracy
                    create_info_file(self.params['model_path'], {'best_accuracy': self.best_accuracy}, 'best_accuracy.txt')
                    self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model.state_dict(), self.best_model_path)

                if current_loss < self.best_validation_loss:
                    self.best_validation_loss = current_loss
                    self.best_epoch = epoch

                if debug:
                    if epoch >= start_eval_epoch:
                        self.logger.info("Epoch [%d/%d] - Training Loss: %.4f - Validation Loss: %.4f - Validation Accuracy: %.2f%%",
                                        epoch, max_epochs, train_results.get('loss',0), current_loss, current_accuracy)
                    elif epoch % 5 == 0:
                        self.logger.info("Epoch [%d/%d] - Training Loss: %.4f - Training Accuracy: %.2f%%",
                                        epoch, max_epochs, train_results.get('loss',0), train_results.get('accuracy',0))

        # --- Aggregate the proxy accuracy over the eval window ---
        # Model selection above keeps the best-val-accuracy checkpoint; the
        # reported scalar is aggregated per eval_window_agg (max|mean|last).
        agg = self.params.get('eval_window_agg', 'max')
        if agg not in ('max', 'mean', 'last'):
            raise ValueError(f"eval_window_agg must be 'max', 'mean' or 'last', got {agg!r}")
        if validation_accuracies:
            if agg == 'max':
                self.best_accuracy = max(validation_accuracies)
            elif agg == 'mean':
                self.best_accuracy = sum(validation_accuracies) / len(validation_accuracies)
            else:  # 'last'
                self.best_accuracy = validation_accuracies[-1]

        # --- Final Result Aggregation ---
        total_training_time = time.time() - t0
        self.params['training_time'] = total_training_time
        
        # 1. Start from the last validation results
        combined_final_results = dict(val_results)
        # 2. Run all post-processing metrics (e.g. HardwareMetrics) ONCE
        if self.post_processing_metrics:
            # Load the best model before running expensive metrics
            if self.best_model_state_dict is not None:
                self.model.load_state_dict(self.best_model_state_dict)
                self.model.eval()
            
            with torch.no_grad():
                for metric in self.post_processing_metrics:
                    try:
                        combined_final_results.update(metric.compute(epoch_results=combined_final_results))
                    except TypeError:
                        combined_final_results.update(metric.compute())
        # Compile the final, comprehensive results dictionary
        final_output = {
            'training_losses': training_losses,
            'training_accuracies': training_accuracies,
            'validation_losses': validation_losses,
            'validation_accuracies': validation_accuracies,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'training_time': total_training_time,
        }

        # Add the validation + post-processing (hardware) metrics
        final_output.update(combined_final_results)
        
        # Defaults so training_params.txt always carries the hardware metrics,
        # even for a candidate that failed before HardwareMetrics ran.
        pack = {
            'total_params':          0,
            'cuda_inference_time':   0.0,
            'model_memory_usage':    0.0,
            'total_flops':           0,
        }

        self.params.update({
            **{k: final_output.get(k, v) for k, v in pack.items()},
            'best_accuracy':         self.best_accuracy,
            'best_validation_loss':  self.best_validation_loss,
        })
        
    
        # Escritura única del archivo
        clean_params = self._get_clean_params_for_saving()
        create_info_file(self.params['model_path'], clean_params, 'training_params.txt')
        self.release_gpu_memory()

        return final_output

    def should_evaluate(self, epoch, start_eval_epoch):
        """
        Determines whether evaluation should be performed at the given epoch.

        Args:
            epoch (int): The current epoch number.
            start_eval_epoch (int): Evaluation starts after this epoch.

        Returns:
            bool: True if this epoch should be validated.

        Only the last 'epochs_to_eval' epochs are validated, since that
        window is what the proxy accuracy is aggregated over.
        """
        return epoch > start_eval_epoch
    
    def release_gpu_memory(self):
        """
        Releases unused GPU memory by clearing the device cache.

        Clears the CUDA or MPS cache depending on which device this trainer is
        using; a no-op on CPU.
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()

    def _get_clean_params_for_saving(self):
        """
        Creates a deep copy of the params dictionary and removes non-serializable
        or unnecessary items like the model object before saving to a file.
        """
        # Use deepcopy to avoid modifying the original params object
        params_to_save = copy.deepcopy(self.params)

        # Metrics needing the live model instance (e.g. HardwareMetrics) get it
        # injected into their 'params' dict (see master.create_metrics_from_config);
        # strip it back out so the non-serializable model object never ends up
        # in training_params.txt.
        for metric_cfg in params_to_save.get('metrics', []):
            if 'params' in metric_cfg:
                metric_cfg['params'].pop('model', None)

        return params_to_save

