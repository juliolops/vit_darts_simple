"""Dataset download and metadata helpers for MoQ-NAS.

``download_dataset``/``setup_dataset_info`` (plus the module-level
``dataset_cache``) extracted verbatim from ``utils/helpers.py`` (Block C
of the refactor roadmap). ``helpers.py`` re-exports these names until it
becomes a facade (stage C.6), so both import paths are equivalent.
Importing this module downloads nothing; downloads only happen when
``download_dataset`` is called.
"""
import os

import torchvision.datasets
from torchvision.transforms import ToTensor

from utils.io import load_yaml


def download_dataset(params: dict):
    """
    Downloads the specified dataset if it is not already available locally.

    Parameters:
    - params (dict): A dictionary containing the parameters for the dataset.
        - 'data_path' (str): The path where the dataset should be stored.
        - 'dataset' (str): The name of the dataset to be downloaded.

    If the dataset directory specified by 'data_path' does not exist, it will be created,
    and the dataset will be downloaded from torchvision. If the dataset already exists,
    it will print a message and skip the download.

    Raises:
    - ValueError: If the dataset is not found in torchvision.datasets.
    """
    data_path = params['data_path']
    dataset_name = params['dataset'].lower()

    download_status = not os.path.exists(data_path)

    if download_status:
        os.makedirs(data_path)

        if hasattr(torchvision.datasets, dataset_name.upper()):
            dataset_class = getattr(torchvision.datasets, dataset_name.upper())
            dataset_class(data_path, download=True, transform=ToTensor())
        else:
            raise ValueError(f"Dataset class {dataset_name} not found in torchvision.datasets.")
        return False
    else:
        return True

# Process-wide cache of validated dataset metadata, keyed by dataset name.
# Lets setup_dataset_info skip re-reading/validating the YAML on repeated
# calls (e.g. once per trainer instance); bypass with
# params['force_reload_dataset_info'] = True.
dataset_cache = {}

def _validate_dataset_info(dataset_info: dict, dataset_name: str):
    """Validate the metadata mapping loaded for a dataset.

    Parameters
    ----------
    dataset_info : dict
        Metadata loaded from the dataset YAML (or ``data_info.txt``); must
        contain ``num_classes``, ``task`` and a 3-element ``shape``
        (``[C, H, W]``).
    dataset_name : str
        Dataset name, used only in error messages.

    Raises
    ------
    ValueError
        If ``dataset_info`` is not a dict or ``shape`` is not ``[C, H, W]``.
    KeyError
        If any required key is missing.
    """
    if not isinstance(dataset_info, dict):
        raise ValueError(f"Dataset info for '{dataset_name}' must be a dict, got {type(dataset_info)}")
    missing = [k for k in ("num_classes", "task", "shape") if k not in dataset_info]
    if missing:
        raise KeyError(f"Dataset info for '{dataset_name}' missing keys: {missing}")
    shape = dataset_info["shape"]
    if not (isinstance(shape, (list, tuple)) and len(shape) == 3):
        raise ValueError(f"'shape' must be [C, H, W] for '{dataset_name}', got: {shape}")

def setup_dataset_info(params):
    """
    Update 'params' with dataset-specific information, prioritizing YAML configs.

    Expects in params:
        - 'dataset' (str)
        - 'batch_size' (int)
        - 'config_path_dataset' (str) -> preferred YAML path, e.g. 'dataset_configs/cifar10_vit.yaml'
        OR
        - 'data_path' (str) with an existing 'data_info.txt' (written by the loader)

    Sets:
        - 'num_classes' (int)
        - 'task' (str)
        - 'input_shape' ([B, C, H, W])
    """
    if "dataset" not in params:
        raise KeyError("params['dataset'] is required")
    if "batch_size" not in params:
        raise KeyError("params['batch_size'] is required")

    dataset_name = str(params["dataset"]).lower()

    # Use cache unless the caller explicitly asks to refresh
    if dataset_name in dataset_cache and not params.get("force_reload_dataset_info", False):
        dataset_info = dataset_cache[dataset_name]
    else:
        # Prefer YAML config
        cfg_path = params.get("config_path_dataset")
        if cfg_path and os.path.isfile(cfg_path):
            dataset_info = load_yaml(cfg_path)
        else:
            # Fallback to info file written by the loader
            data_info_path = os.path.join(params.get("data_path", ""), "data_info.txt")
            if not os.path.isfile(data_info_path):
                raise FileNotFoundError(
                    "Could not find dataset YAML at params['config_path_dataset'] "
                    "and fallback 'data_info.txt' does not exist at: "
                    f"{data_info_path}"
                )
            dataset_info = load_yaml(data_info_path)

        _validate_dataset_info(dataset_info, dataset_name)
        dataset_cache[dataset_name] = dataset_info

    # Populate params
    params["num_classes"] = int(dataset_info["num_classes"])
    params["task"] = str(dataset_info["task"])
    c, h, w = [int(x) for x in dataset_info["shape"]]
    params["input_shape"] = [int(params["batch_size"]), c, h, w]
    return params
