""" Copyright (c) 2025, Diego Páez
* Licensed under the MIT license

- Builds and trains one candidate: turns a decoded chromosome into a pruned
  ViT, wires up the configured metrics, and runs the training loop.
"""
import os
import logging
import traceback
import torch
import torch.nn as nn

from typing import Dict, List, Union, Any
from . import trainer

from .metrics import Accuracy, HardwareMetrics

# Handler configuration is deferred to the run instance (init_log downstream,
# once experiment_path is known); importing this module must be side-effect free.
LOGGER = logging.getLogger(__name__)

# Metric names accepted in the config's `metrics:` list.
METRIC_MAP = {
    "Accuracy": Accuracy,
    "HardwareMetrics": HardwareMetrics,
}


def create_metrics_from_config(config: dict, model_instance, device, input_shape) -> list:
    """
    Build metric instances from config['metrics'].

    Args:
        config (dict): Experiment/train spec dict (must include 'metrics': [...]).
        model_instance (torch.nn.Module): The model to evaluate.
        device (str): 'cuda', 'mps' or 'cpu'.
        input_shape (tuple): (B, C, H, W)

    Returns:
        list: List of metric objects.
    """
    metric_instances = []
    for metric_config in config.get('metrics', []):
        metric_name = metric_config['name']
        if metric_name not in METRIC_MAP:
            continue
        params = dict(metric_config.get('params', {}))
        # HardwareMetrics measures the live model, so it needs it injected.
        if metric_name == "HardwareMetrics":
            params['model'] = model_instance
            params['device'] = device
            params['input_shape'] = input_shape[1:]  # Pass (C, H, W)
        metric_instances.append(METRIC_MAP[metric_name](**params))
    return metric_instances


def create_optimizer(net, params):
    """Optimizer over the trainable parameters only.

    Everything but the classifier head is frozen (see ``core/vit.py``), so
    handing the optimizer the frozen tensors would just waste optimizer state
    on parameters that never receive a gradient.
    """
    opt_name = str(params.get('optimizer', 'AdamW')).lower()
    default_lrs = {'adamw': 1e-3, 'adam': 1e-3, 'sgd': 1e-2, 'rmsprop': 1e-2}
    base_lr = float(params.get('learning_rate', default_lrs.get(opt_name, 1e-3)))
    weight_decay = float(params.get('weight_decay', 0.0))

    trainable = [p for p in net.parameters() if p.requires_grad]
    if opt_name == 'adamw':
        return torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return torch.optim.Adam(trainable, lr=base_lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return torch.optim.SGD(trainable, lr=base_lr,
                               momentum=float(params.get('momentum', 0.9)),
                               nesterov=bool(params.get('nesterov', True)),
                               weight_decay=weight_decay)
    elif opt_name == 'rmsprop':
        return torch.optim.RMSprop(trainable, lr=base_lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {params.get('optimizer')}")


def setup_additional_params(params, id_num=None):
    """
    Set the per-candidate output directory (results/gen_<g>/<g>_<i>) and record
    which generation/individual this candidate is.
    """
    generation, individual = id_num.split('_')[0], id_num.split('_')[1]

    temp_root = os.path.join(params['experiment_path'], 'results', f'gen_{generation}')
    os.makedirs(temp_root, exist_ok=True)

    model_path = os.path.join(temp_root, id_num)
    os.makedirs(model_path, exist_ok=True)
    params['model_path'] = model_path
    params['generation'] = generation
    params['individual'] = individual
    return params


def create_model(params: Dict[str, Any]) -> nn.Module:
    """Build the pruned ViT this chromosome describes.

    Each gene is the percentage of attention heads its block keeps; which
    heads survive comes from the DARTS alphas, not from the GA.
    """
    from core.vit import build_pruned_vit, load_alphas  # local: keeps timm optional

    return build_pruned_vit(
        net_list=params['net_list'],
        fn_dict=params['fn_dict'],
        alphas=load_alphas(params['vit_alphas_path']),
        num_classes=params['num_classes'],
        model_name=params.get('vit_model_name', 'vit_base_patch16_224'),
        pretrained=params.get('vit_pretrained', True),
    )


def create_model_and_trainer(params, train_loader, val_loader, test_loader):
    """Build the model and the trainer that will run it."""
    net = create_model(params)
    metrics = create_metrics_from_config(
        config=params, model_instance=net,
        device=params['device'], input_shape=params['input_shape'])

    return trainer.BaseTrainer(net, nn.CrossEntropyLoss(), create_optimizer(net, params),
                               train_loader, val_loader, test_loader, params, metrics)


def run_training_phase(params: Dict[str, Any],
                        fn_dict: Dict[str, Any] = None,
                        net_list: List[str] = None, decoded_params: Union[Dict[str, Any], List] = None,
                        id_num: str = None, debug: bool = False,
                        train_loader=None, val_loader=None, test_loader=None) -> Dict[str, Any]:
    """
    Update params with this candidate's architecture, create the trainer and
    run it.

    Args:
        params (Dict[str, Any]): Configuration dictionary.
        fn_dict (Dict[str, Any], optional): Search-space definition.
        decoded_params (List, optional): Decoded hyperparameters for the model.
        debug (bool, optional): If True, logs per-epoch progress.
        net_list (List[str], optional): Decoded chromosome (one gene per block).
        id_num (str, optional): Identifier "generation_individual".
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader (unused during evolution).

    Returns:
        Dict[str, Any]: Dictionary containing the training results.
    """
    if fn_dict is not None:
        params['fn_dict'] = fn_dict
    if net_list is not None:
        params['net_list'] = net_list
    if id_num is not None:
        params = setup_additional_params(params, id_num=id_num)

    trainer_instance = create_model_and_trainer(params, train_loader, val_loader, test_loader)
    results_dict = trainer_instance.train(debug=debug)

    return results_dict, trainer_instance.best_model_path


# --- Entry-point function ---
def fitness(id_num: str, params: Dict[str, Any],
            fn_dict: Dict[str, Any], net_list: List[str],
            decoded_params: Dict[str, Any],
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            debug: bool = False) -> Dict[str, Any]:
    """
    Train and evaluate one evolved candidate.

    Args:
        id_num (str): Identifier for the model in the format "generation_individual".
        params (Dict[str, Any]): Configuration dictionary.
        fn_dict (Dict[str, Any]): Search-space definition.
        decoded_params (Dict[str, Any]): Decoded hyperparameters for the model.
        net_list (List[str]): Decoded chromosome (one gene per block).
        train_loader (torch.utils.data.DataLoader): Training DataLoader.
        debug (bool, optional): If True, logs per-epoch progress.

    Returns:
        Dict[str, Any]: Dictionary containing the training results.

    Raises:
        Exception: Propagated after logging, so the caller can score the
            candidate 0.0 instead of losing the whole generation.
    """
    try:
        results_dict, model_path = run_training_phase(
            params, fn_dict, net_list, decoded_params, id_num, debug,
            train_loader, val_loader, None)
        LOGGER.info(f"Training of _path {id_num} finished, best "
                    f"{params['fitness_metric']}: "
                    f"{round(results_dict[params['fitness_metric']], 2)}")
        return results_dict, model_path
    except Exception as e:
        LOGGER.error(f"An error occurred during training of model {id_num}: {e}")
        LOGGER.error(traceback.format_exc())
        raise e
