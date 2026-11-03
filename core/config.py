"""Load and validate the experiment YAML, applying command-line overrides."""
import yaml

REQUIRED = {
    'num_blocks': int, 'prune_mlp': bool, 'percentages': list,
    'vit_model_name': str, 'vit_pretrained': bool, 'vit_alphas_path': str,
    'batch_size': int, 'eval_batch_size': int, 'max_epochs': int, 'epochs_to_eval': int,
    'learning_rate': float, 'mlp_learning_rate': float, 'weight_decay': float,
    'data_path': str, 'train_split': float, 'split_seed': int, 'loader_seed': int,
    'limit_data_value': int, 'num_workers': int, 'threads': int,
}

# Optional keys and their defaults (older configs without them run on CIFAR-10).
OPTIONAL = {'dataset': 'cifar10', 'imagenet_classes': 0, 'imagenet_pretrained_head': True,
            'flops_counter': 'hooks'}

# CLI flags that, when given, replace the config value.
OVERRIDES = ('data_path', 'limit_data_value', 'threads')


def load_config(args: dict) -> dict:
    with open(args['config_file'], encoding='utf-8') as f:
        params = yaml.safe_load(f)

    for key, expected in REQUIRED.items():
        if key not in params:
            raise KeyError(f"'{key}' not found in {args['config_file']}.")
        if not isinstance(params[key], expected):
            raise TypeError(f"'{key}' should be {expected.__name__} but is "
                            f"{type(params[key]).__name__}.")

    for key, default in OPTIONAL.items():
        params.setdefault(key, default)
    if params['dataset'] not in ('cifar10', 'cifar100', 'imagenet'):
        raise ValueError("dataset must be 'cifar10', 'cifar100' or 'imagenet', "
                         f"got {params['dataset']!r}.")
    if params['flops_counter'] not in ('hooks', 'pytorch'):
        raise ValueError("flops_counter must be 'hooks' (original, Linear/Conv only) or "
                         f"'pytorch' (FlopCounterMode), got {params['flops_counter']!r}.")
    if not isinstance(params['imagenet_classes'], int) or params['imagenet_classes'] < 0:
        raise ValueError("imagenet_classes must be an int >= 0 (0 = all classes).")

    for percent in params['percentages']:
        if not isinstance(percent, int) or not 0 < percent <= 100:
            raise ValueError(f"percentages must be ints in (0, 100], got {percent!r}.")
    # Ascending, so a +/-1 gene mutation is a one-step change in the percentage.
    params['percentages'] = sorted(params['percentages'])

    if params['epochs_to_eval'] >= params['max_epochs']:
        raise ValueError('epochs_to_eval must be < max_epochs.')

    # One head gene per block, plus one MLP gene per block when pruning the MLP.
    params['num_genes'] = params['num_blocks'] * (2 if params['prune_mlp'] else 1)

    for key in OVERRIDES:
        if args.get(key) is not None:
            params[key] = args[key]
    params['seed'] = args['seed']
    return params
