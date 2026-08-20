""" Copyright (c) 2025, Diego Páez
* Licensed under the MIT license

- Master module for training and evaluating CNN models.
- Updated to support the pluggable metrics system by dynamically
        creating metric objects from the configuration.

"""
import os
import logging
import traceback
import torch
import torch.nn as nn

from typing import Dict, List, Union, Any
from . import model, trainer

from .artifacts import ConfusionMatrix
from .metrics import Accuracy, HardwareMetrics
from .metrics.fitness import ScalarizedFitness, ValidationLossFitness

# Handler configuration is deferred to the run instance (init_log downstream,
# once experiment_path is known); importing this module must be side-effect free.
LOGGER = logging.getLogger(__name__)


def create_metrics_from_config(config: dict, model_instance, device, input_shape) -> list:
    """
    Build metric instances from config['metrics'].
    Injects runtime params (model, device, input_shape/img_size) when needed.

    Args:
        config (dict): Experiment/train spec dict (must include 'metrics': [...]).
        model_instance (torch.nn.Module): The model to evaluate.
        device (str): 'cuda' or 'cpu'.
        input_shape (tuple): (B, C, H, W)

    Returns:
        list: List of metric objects.
    """
    metric_instances = []
    if 'metrics' not in config:
        return []

    # Maps metric names from the YAML file to their corresponding Python classes
    metric_map = {
        "Accuracy": Accuracy,
        "ConfusionMatrix": ConfusionMatrix,
        "HardwareMetrics": HardwareMetrics,
        "ScalarizedFitness": ScalarizedFitness,
        "ValidationLossFitness": ValidationLossFitness,
    }

    for metric_config in config['metrics']:
        metric_name = metric_config['name']
        if metric_name in metric_map:
            MetricClass = metric_map[metric_name]
            params = metric_config.get('params', {})

            # Automatically inject runtime parameters needed by certain metrics
            if metric_name == "HardwareMetrics":
                params['model'] = model_instance
                params['device'] = device
                params['input_shape'] = input_shape[1:]  # Pass (C, H, W)

            metric_instances.append(MetricClass(**params))

    return metric_instances

def create_artifacts_from_config(config: dict) -> list:
    """
    Reads the 'artifacts' section from the configuration and creates the
    corresponding artifact class instances.
    """
    artifact_instances = []
    artifacts_config = config.get('artifacts', [])
    if not artifacts_config:
        return []

    # Maps artifact names from the YAML file to their corresponding Python classes
    artifact_map = {
        "ConfusionMatrix": ConfusionMatrix,
        # Aquí añadirías futuros artefactos, ej: "ROCAUCPlot": ROCAUCPlot
    }

    for artifact_cfg in artifacts_config:
        artifact_name = artifact_cfg['name']
        if artifact_name in artifact_map:
            ArtifactClass = artifact_map[artifact_name]
            params = artifact_cfg.get('params', {})
            artifact_instances.append(ArtifactClass(**params))
            
    return artifact_instances

def create_optimizer(net, params):
    """
    Backbone-aware optimizer creator with sensible defaults.
    - Works even if params lacks 'learning_rate' and 'weight_decay'
    - Uses discriminative LRs (smaller for backbone, larger for head) when network_config == 'backbone'
    """
    # Normalize optimizer name and provide defaults per optimizer
    opt_name = str(params.get('optimizer', 'AdamW')).lower()
    default_lrs = {
        'adamw':  1e-3,
        'adam':   1e-3,
        'sgd':    1e-2,
        'rmsprop':1e-2,
    }
    base_lr      = float(params.get('learning_rate', default_lrs.get(opt_name, 1e-3)))
    weight_decay = float(params.get('weight_decay', 0.0))

    # Detect backbone config
    is_backbone_cfg = (
        str(params.get('network_config', '')).lower() == 'backbone'
        and hasattr(net, 'backbone') and (net.backbone is not None)
    )

    if is_backbone_cfg:
        # Split parameter groups; respect requires_grad (frozen backbone -> empty group)
        backbone_params = [p for p in net.backbone.parameters() if p.requires_grad]
        head_params     = [p for n, p in net.named_parameters()
                        if not n.startswith('backbone') and p.requires_grad]

        # Discriminative learning rates with safe defaults
        bb_lr   = float(params.get('backbone_lr', max(1e-5, base_lr / 3.0)))
        head_lr = float(params.get('head_lr', base_lr))

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": bb_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr})

        # If somehow everything is frozen, fall back to any trainable params
        if not param_groups:
            param_groups = [{"params": [p for p in net.parameters() if p.requires_grad], "lr": base_lr}]

        # Build optimizer
        if opt_name == 'adamw':
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        elif opt_name == 'adam':
            return torch.optim.Adam(param_groups, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            momentum = float(params.get('momentum', 0.9))
            nesterov = bool(params.get('nesterov', True))
            return torch.optim.SGD(param_groups, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        elif opt_name == 'rmsprop':
            return torch.optim.RMSprop(param_groups, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {params.get('optimizer')}")

    # ---- Non-backbone fallback: single param group, safe defaults ----
    # Only trainable parameters: in the ViT space everything but the
    # classifier head is frozen, so handing the optimizer the frozen tensors
    # would just waste optimizer state on parameters that never get a grad.
    trainable = [p for p in net.parameters() if p.requires_grad]
    if opt_name == 'adamw':
        return torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return torch.optim.Adam(trainable, lr=base_lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = float(params.get('momentum', 0.9))
        nesterov = bool(params.get('nesterov', True))
        return torch.optim.SGD(trainable, lr=base_lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    elif opt_name == 'rmsprop':
        return torch.optim.RMSprop(trainable, lr=base_lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {params.get('optimizer')}")


def setup_additional_params(params, id_num=None):
    """
    Set additional parameters such as model path, generation, and individual if an identifier is provided.
    This is useful for the evolution phase where each model has a unique ID.

    Args:
        params (Dict[str, Any]): Configuration dictionary.
        id_num (str, optional): A string identifier in the format "generation_individual".

    Returns:
        Dict[str, Any]: Updated configuration dictionary with additional keys (e.g. 'model_path',
                        'generation', 'individual').
    """
    generation = id_num.split('_')[0]
    individual = id_num.split('_')[1]
    
    temp_root = os.path.join(params['experiment_path'], 'results', f'gen_{generation}')
    os.makedirs(temp_root, exist_ok=True)
    
    model_path = os.path.join(temp_root, id_num)
    os.makedirs(model_path, exist_ok=True)
    params['model_path'] = model_path
    params['generation'] = generation
    params['individual'] = individual
    return params

def create_model(params: Dict[str, Any]) -> nn.Module:
    """
    Creates and initializes a model instance based on the provided parameters.
    This function is responsible for building the network graph and initializing it.
    """
    if params.get('search_space', 'cnn') == 'vit':
        # ViT space: the chromosome carries a head-keep percentage per block;
        # which heads survive comes from the DARTS alphas, not the GA.
        from core.vit import build_pruned_vit, load_alphas  # local: keeps timm optional
        alphas = load_alphas(params['vit_alphas_path'])
        return build_pruned_vit(
            net_list=params['net_list'],
            fn_dict=params['fn_dict'],
            alphas=alphas,
            num_classes=params['num_classes'],
            model_name=params.get('vit_model_name', 'vit_base_patch16_224'),
            pretrained=params.get('vit_pretrained', True),
        )

    net = model.NetworkGraph(num_classes=params['num_classes'],
                            input_shape=params['input_shape'],
                            network_config=params['network_config'],
                            backbone_name=params['backbone_name'],
                            backbone_percentage=params.get('backbone_percentage', 1.0),
                            backbone_trainable=False)

    if params.get('network_config') == 'backbone':
        net.auto_resize_backbone = False

    # Create functions and run a dummy forward pass to initialize layers
    filtered_dict = {key: val for key, val in params['fn_dict'].items() if key in params['net_list']}
    has_cbam_key = any(key.startswith('cbam') for key in filtered_dict)
    net.create_functions(fn_dict=filtered_dict, net_list=params['net_list'], cbam=has_cbam_key)

    dummy_input = torch.randn(params['input_shape'])
    with torch.no_grad():
        _ = net(dummy_input)

    return net

def create_trainer(params: Dict[str, Any], model: nn.Module, train_loader, val_loader, test_loader):
    """
    Creates a trainer instance for a given model.
    """
    optimizer = create_optimizer(model, params)
    criterion = nn.CrossEntropyLoss()

    metrics = create_metrics_from_config(
        config=params,
        model_instance=model,
        device=params['device'],
        input_shape=params['input_shape']
    )
    artifacts = create_artifacts_from_config(config=params)

    return trainer.BaseTrainer(model, criterion, optimizer,
                            train_loader, val_loader, test_loader, params,
                            metrics, artifacts)

# This function is now a simple wrapper around the two new functions
def create_model_and_trainer(params, train_loader, val_loader, test_loader):
    """
    Creates the model and the corresponding trainer instance.
    """
    net = create_model(params)
    trainer_instance = create_trainer(params, net, train_loader, val_loader, test_loader)
    return trainer_instance

def run_training_phase(params: Dict[str, Any],
                        fn_dict: Dict[str, Any] = None,
                        net_list: List[str] = None, decoded_params: Union[Dict[str, Any], List] = None,
                        id_num: str = None, debug: bool = False,
                        train_loader=None, val_loader=None, test_loader=None) -> Dict[str, Any]:
    """
    Generic function to update parameters, create the trainer, and run training.
    It updates fn_dict, net_list, and additional parameters if provided, and then
    creates a trainer instance and runs it.

    Args:
        params (Dict[str, Any]): Configuration dictionary.
        fn_dict (Dict[str, Any], optional): Dictionary of layer definitions.
        decoded_params (List, optional): List of decoded parameters for the model.
        debug (bool, optional): If True, returns the raw results dictionary for debugging.
        net_list (List[str], optional): List of layer names to be used in the network.
        id_num (str, optional): Identifier for the model (used in the evolution phase).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader (can be None for certain phases).

    Returns:
        Dict[str, Any]: Dictionary containing the training results.
    """
    if fn_dict is not None:
        params['fn_dict'] = fn_dict
    if net_list is not None:
        params['net_list'] = net_list
    if id_num is not None:
        params = setup_additional_params(params, id_num=id_num)
        
    if isinstance(decoded_params, dict) and 'backbone_percentage' in decoded_params:
        params['backbone_percentage'] = decoded_params['backbone_percentage']
    else:
        # If decoded params is not being evolved, get the value from the config file
        # or set it to 1.0 by default
        params['backbone_percentage'] = params.get('backbone_percentage', 1.0)

    trainer = create_model_and_trainer(params, train_loader, val_loader, test_loader)

    results_dict = trainer.train(debug=debug)

    return results_dict, trainer.best_model_path

# --- Entry-point functions ---
def fitness(id_num: str, params: Dict[str, Any], 
            fn_dict: Dict[str, Any], net_list: List[str],
            decoded_params: Dict[str, Any],
            train_loader: torch.utils.data.DataLoader, 
            val_loader: torch.utils.data.DataLoader,
            debug: bool = False) -> Dict[str, Any]:
    """
    Train and evaluate a model using evolved networks in the evolution phase.
    Updates the mutable container return_val with key performance metrics based on the fitness_metric in params.

    Args:
        id_num (str): Identifier for the model in the format "generation_individual".
        params (Dict[str, Any]): Configuration dictionary with evolved networks.
        fn_dict (Dict[str, Any]): Dictionary of layer definitions.
        decoded_params (Dict[str, Any]): Decoded parameters for the model.
        net_list (List[str]): List of layer names to be used in the network.
        train_loader (torch.utils.data.DataLoader): Training DataLoader.
        debug (bool, optional): If True, returns the raw results dictionary for debugging.

    Returns:
        Dict[str, Any]: Dictionary containing the training results.

    Raises:
        Exception: Propagates any exception encountered during training after setting return_val to zeros.
    """
    try:
        results_dict, model_path = run_training_phase(params, fn_dict, net_list, decoded_params, id_num, debug, train_loader, val_loader, None)
        LOGGER.info(f"Training of _path {id_num} finished, best {params['fitness_metric']}: {round(results_dict[params['fitness_metric']], 2)}")
        return results_dict, model_path
    except Exception as e:
        LOGGER.error(f"An error occurred during training of model {id_num}: {e}")
        LOGGER.error(traceback.format_exc()) # This will log the full traceback
        # Set return_val to zeros if an error occurs
        raise e

