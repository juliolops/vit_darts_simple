"""Load and validate the experiment YAML, applying command-line overrides."""
import yaml

REQUIRED = {
    'max_num_nodes': int, 'percentages': list,
    'vit_model_name': str, 'vit_pretrained': bool, 'vit_alphas_path': str,
    'batch_size': int, 'eval_batch_size': int, 'max_epochs': int, 'epochs_to_eval': int,
    'learning_rate': float, 'weight_decay': float,
    'data_path': str, 'train_split': float, 'split_seed': int, 'loader_seed': int,
    'limit_data_value': int, 'num_workers': int, 'threads': int,
}

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

    for percent in params['percentages']:
        if not isinstance(percent, int) or not 0 < percent <= 100:
            raise ValueError(f"percentages must be ints in (0, 100], got {percent!r}.")
    # Ascending, so a +/-1 gene mutation is a one-step change in the percentage.
    params['percentages'] = sorted(params['percentages'])

    if params['epochs_to_eval'] >= params['max_epochs']:
        raise ValueError('epochs_to_eval must be < max_epochs.')

    for key in OVERRIDES:
        if args.get(key) is not None:
            params[key] = args[key]
    params['seed'] = args['seed']
    return params
