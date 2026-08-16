""" Copyright (c) 2025, Diego Páez
* Licensed under the MIT license

- trainer module handles the training and evaluation of CNN models for evolutionary
    algorithms.
- It includes a base trainer class and a specialized ResNet trainer class.
- Refactored trainer module to support a pluggable metrics system while
    retaining direct loss calculation for training consistency.
- The trainer is now agnostic to the specific metrics being calculated,
    making it highly extensible for new evaluation criteria like fairness.
"""


import os
import copy
import time
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                    MultiStepLR, ReduceLROnPlateau)

from . import model
from .artifacts import BaseArtifact
from .metrics.base import BaseMetric
from utils.helpers import create_info_file, init_log
from core.precision import resolve_precision
from settings import TRAIN_TIMEOUT


class BaseTrainer:
    """
    BaseTrainer is a base class for training, evaluating, and managing deep 
    learning models using PyTorch.
    It provides a unified interface for model training, validation, testing, 
    metric computation, and logging, with support for mixed precision training, 
    learning rate scheduling, and multi-objective fitness evaluation.
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
        update_scheduler(scheduler, metric=None):
            Updates the learning rate scheduler based on the specified policy.
        release_gpu_memory():
            Releases GPU memory by clearing the CUDA cache.
        reset_and_load_best_model(best_model_path):
            Reinitializes and loads the best model checkpoint.
        _initialize_scheduler(max_epochs):
            Initializes the learning rate scheduler based on configuration.
        should_evaluate(epoch, start_eval_epoch):
            Determines if evaluation should be performed at the current epoch.
        compute_mo_fitness(total_params, cuda_inference_time):
            Computes scalarized multi-objective fitness for model selection.
        get_final_metrics():
            Computes final model metrics such as inference time, parameter count, FLOPs, and memory usage.
        train(debug=False):
            Main training loop that manages training, validation, checkpointing, and metric computation.
    Notes:
        - Supports mixed precision training via torch.cuda.amp.
        - Handles both single-objective and multi-objective (e.g., accuracy, loss, parameters, inference time) optimization.
        - Designed for extensibility and integration with neural architecture search workflows.
    """
    def __init__(self, model_instance, criterion, optimizer, train_loader, val_loader, test_loader,
                params: dict, metrics: list[BaseMetric], artifacts: list[BaseArtifact]):
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

        # Clones are only needed for primary metrics.
        self.val_primary_metrics = [m.__class__(**m._init_args) for m in self.primary_metrics]
        self.test_primary_metrics = [m.__class__(**m._init_args) for m in self.primary_metrics]

        self.artifacts = artifacts # List of artifact instances to compute after training


        # --- State Tracking & Checkpointing ---
        self.best_accuracy = 0.0
        self.best_validation_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
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

    def _run_test_phase(self):
        """
        Runs the final test phase, computing both test metrics and all artifacts.
        This method is called only once at the end of the 'retrain' or 'resnet' phase.
        
        Returns:
            dict: A dictionary containing results from both test metrics and artifacts.
        """
        if self.test_loader is None:
            return {}

        self.model = self.reset_and_load_best_model(self.best_model_path)

        batch_processors = self.test_primary_metrics + self.artifacts
        for processor in batch_processors:
            processor.reset()
        for processor in getattr(self, "post_processing_metrics", []):
            processor.reset()

        self.model.eval()

        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in self.test_loader:
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    inputs, labels, _ = batch

                outputs, batch_loss, labels = self._forward_pass(inputs, labels)

                bs = inputs.size(0) if hasattr(inputs, "size") else len(labels)
                total_loss += float(batch_loss.item()) * bs
                total_examples += bs

                for processor in batch_processors:
                    processor.update(outputs.detach(), labels.detach())

        all_final_processors = self.test_primary_metrics + self.post_processing_metrics + self.artifacts
        final_results = {}
        for processor in all_final_processors:
            try:
                final_results.update(processor.compute(epoch_results=final_results))
            except TypeError:
                final_results.update(processor.compute())

        if 'loss' not in final_results and total_examples > 0:
            final_results['loss'] = total_loss / total_examples

        self.logger.info(
            "Experiment: %s - Test loss: %.4f - Test accuracy: %.2f%%",
            self.params['experiment_path'],
            float(final_results.get('loss', 0.0)),
            float(final_results.get('accuracy', 0.0)),
        )
        return final_results
    
    def train(self, debug=False):
        """
        The main training loop, orchestrating epochs, evaluation, and final result aggregation.
        """
        max_epochs = self.params['max_epochs']
        epochs_to_eval = self.params['epochs_to_eval']
        patience_max = self.params.get('patience_retrain', max_epochs)
        base_fraction = self.params.get('delta_fraction', 0.005)
        start_eval_epoch = max_epochs - epochs_to_eval
        val_results = {}   
        test_results = {}  
        t0 = time.time()
        
        training_losses, training_accuracies = [], []
        validation_losses, validation_accuracies = [], []

        phase = self.params.get('phase')
        if phase == 'retrain':
            self.logger.info("Retraining evolved model %s ...", self.params['experiment_path'])
            self.scheduler = self._initialize_scheduler()

        for epoch in range(1, max_epochs + 1):
            train_results = self._run_epoch(self.train_loader, is_training=True, metric_set=self.primary_metrics)
            training_losses.append(train_results.get('loss', 0))
            training_accuracies.append(train_results.get('accuracy', 0))

            timeout = int(self.params.get('train_timeout', TRAIN_TIMEOUT))
            if epoch < start_eval_epoch and (time.time() - t0) > timeout and phase != 'retrain':
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

                if self.best_validation_loss == float('inf'):
                    self.best_validation_loss = current_loss
                    self.no_improve_count = 0
                    continue
                min_delta = self.best_validation_loss * base_fraction
                if (self.best_validation_loss - current_loss) > min_delta:
                    self.best_validation_loss = current_loss
                    self.best_epoch = epoch
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                    if self.no_improve_count >= patience_max and phase == 'retrain':
                        self.logger.info("Early stopping at epoch %d", epoch)
                        break
                
                if phase == 'retrain':
                    self.update_scheduler(metric=current_loss)
                    if epoch % 25 == 0:
                        self.logger.info("Experiment: %s: Epoch [%d/%d] - Train Loss: %.2f - Val Loss: %.2f - Val Acc: %.2f%%",
                                        self.params['experiment_path'], epoch, max_epochs, train_results.get('loss',0), current_loss, current_accuracy)
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
        
        # 1. Run the test phase if needed (retrain/resnet)
        test_results = {}
        if (phase == 'retrain' or phase == 'resnet') and self.test_loader is not None:
            test_results = self._run_test_phase()

        # 2. Combine all intermediate results into one dictionary
        combined_final_results = {**val_results, **test_results}
        # 3. Run all post-processing metrics ONCE, using the combined results
        if self.post_processing_metrics and phase == 'evolution':
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
        # Compile the final, comprehensive results dictionary in the original format
        final_output = {
            'training_losses': training_losses,
            'training_accuracies': training_accuracies,
            'validation_losses': validation_losses,
            'validation_accuracies': validation_accuracies,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'training_time': total_training_time,
            'test_accuracy': test_results.get('accuracy'),
            'test_loss': test_results.get('loss'),
        }
        
        # Add all other computed metrics from test and post-processing
        #final_output.update(test_results)
        final_output.update(combined_final_results)
        
        pack = {
            'total_params':          0,
            'cuda_inference_time':   0.0,
            'model_memory_usage':    0.0,
            'total_flops':           0,
            'fitness_val_loss':      0.0,
            'scalar_multi_objective': 0.0,
        }

        # Un solo update a self.params:
        self.params.update({
            **{k: final_output.get(k, v) for k, v in pack.items()},
            'best_accuracy':         self.best_accuracy,
            'best_validation_loss':  self.best_validation_loss,
            'test_accuracy':         test_results.get('accuracy', None),
            'test_loss':             test_results.get('loss', None),
        })
        
    
        # Escritura única del archivo
        clean_params = self._get_clean_params_for_saving()
        create_info_file(self.params['model_path'], clean_params, 'training_params.txt')
        self.release_gpu_memory()

        return final_output

    def _initialize_scheduler(self):
        """
        Initializes and returns a learning rate scheduler for the optimizer based on the specified parameters.

        Args:
            max_epochs (int): The maximum number of training epochs, used for certain schedulers.

        Returns:
            torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.ReduceLROnPlateau or None:
                The initialized learning rate scheduler object, or None if no scheduler is specified.

        Supported schedulers:
            - 'exponential': ExponentialLR with gamma=0.9.
            - 'reduce_on_plateau': ReduceLROnPlateau with patience=5 and factor=0.1.
            - 'cosine': CosineAnnealingLR with T_max=max_epochs and eta_min=0.
            - 'multistep': MultiStepLR with milestones at 50% and 75% of max_epochs, gamma=0.1.
        """
        scheduler_name = self.params.get('lr_scheduler')
        max_epochs = self.params.get('max_epochs', 150)
        if scheduler_name == 'exponential':
            return ExponentialLR(self.optimizer, gamma=0.9)
        elif scheduler_name == 'reduce_on_plateau':
            return ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1)
        elif scheduler_name == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=0)
        elif scheduler_name == 'multistep':
            milestones = [int(0.5 * max_epochs), int(0.75 * max_epochs)] 
            return MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        return None

    def update_scheduler(self, metric=None):
        """
        Updates the learning rate scheduler based on the specified strategy.

        If the scheduler is set to 'reduce_on_plateau' and a metric is provided, 
        the scheduler's step method is called with the metric to adjust the learning rate 
        based on the metric's value (typically validation loss). Otherwise, the scheduler's 
        step method is called without arguments to perform a standard update.

        Args:
            metric (float, optional): The value of the monitored metric (e.g., validation loss) 
                used when the scheduler is 'reduce_on_plateau'. Defaults to None.

        """
        if self.scheduler is None: return
        if isinstance(self.scheduler, ReduceLROnPlateau): 
            self.scheduler.step(metric)
        else: 
            self.scheduler.step()

    def should_evaluate(self, epoch, start_eval_epoch):
        """
        Determines whether evaluation should be performed at the given epoch.

        Args:
            epoch (int): The current epoch number.
            start_eval_epoch (int): The epoch number after which evaluation should start during the 'evolution' phase.

        Returns:
            bool: True if evaluation should be performed based on the current phase and epoch, False otherwise.

        Phases:
            - 'evolution': Evaluation starts after 'start_eval_epoch'.
            - 'retrain' or 'resnet': Evaluation is always performed.
        """
        phase = self.params.get('phase')
        if phase == 'evolution' and epoch > start_eval_epoch:
            return True
        elif phase == 'retrain' or phase == 'resnet':
            return True
        return False
    
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

    def reset_and_load_best_model(self, best_model_path):
        """
        Reinitializes the model and loads the weights from the specified best model checkpoint.
        This method creates a new instance of the model using the current parameters,
        filters and assigns the appropriate functions, performs a dummy forward pass to
        initialize the model's layers, loads the saved state dictionary from the given
        checkpoint path, moves the model to the specified device, and returns the loaded model.
        Args:
            best_model_path (str): Path to the checkpoint file containing the best model's 
            state_dict.
        Returns:
            torch.nn.Module: The reinitialized model with weights loaded from the checkpoint.
        """
        # Reinitialize the model and load weights from the best model checkpoint.
        backbone_trainable = True if self.params.get('phase') == 'retrain' else False
        best_model = model.NetworkGraph(num_classes=self.params["num_classes"],
                                        input_shape=self.params['input_shape'],
                                        network_config=self.params['network_config'],
                                        backbone_name=self.params['backbone_name'],
                                        backbone_percentage=self.params['backbone_percentage'],
                                        backbone_trainable=backbone_trainable)

        if self.params.get('network_config') == 'backbone':
                best_model.auto_resize_backbone = False  # disable auto upscaling for ablations/strict 32x32
        
        filtered_dict = {key: item for key, item in self.params['fn_dict'].items() if key in self.params['net_list']}
        best_model.create_functions(fn_dict=filtered_dict, net_list=self.params['net_list'])
        input_random = torch.randn(self.params['input_shape'])
        with torch.no_grad():
            _ = best_model(input_random)
        best_model.load_state_dict(torch.load(best_model_path, weights_only=True))
        best_model.to(self.params['device'])
        best_model.eval()
        return best_model