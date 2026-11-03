"""Datasets for the pretrained ViT, at 224x224: CIFAR-10 or ImageNet.

Pick the dataset with ``dataset`` in the config (``cifar10``, the default, or
``imagenet``). Data protocol, the same for every phase and both datasets:

- **train**: a stratified ``train_split`` share of the dataset's *train* split
  (``split_seed``). DARTS trains the MLP + classifier weights on it, the search
  trains every candidate on it, and ``retrain.py`` trains the final model on it.
- **val**: the rest of that train split. DARTS updates the alphas on it, the
  search scores candidate accuracy on it, and ``retrain.py`` picks the best
  epoch on it.
- **test**: a held-out split, reachable only through ``build_test_dataset``,
  which only ``retrain.py`` calls, once, after training. For CIFAR-10 it is the
  official test set; for ImageNet, whose test labels are not public, it is the
  official validation set (50,000 images), as is standard practice.

ImageNet is read from folders (it cannot be downloaded automatically)::

    <data_path>/train/<class>/*.JPEG     # 1.28M images, 1000 class folders
    <data_path>/val/<class>/*.JPEG       # 50,000 images, the same class folders

The class folders must sort in the ImageNet-1k class order (WordNet ids such as
``n01440764``, or zero-padded indices ``0000``..``0999``), so that class ``i``
matches output ``i`` of the pretrained ImageNet classifier, which is then kept.
``imagenet_classes: N`` (N > 0) restricts every split to N classes spread evenly
over the 1000, for quicker experiments.
"""
import os

import numpy as np
import torch
import torchvision.datasets as tvd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

DATASETS = ('cifar10', 'imagenet')
INPUT_SHAPE = (3, 224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_1K = 1000
_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# --------------------------------------------------------------------------- common

class _TransformWrapper(Dataset):
    """Apply ``tfm`` lazily to the images of ``subset``."""

    def __init__(self, subset, tfm):
        self.subset, self.tfm = subset, tfm

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]
        return self.tfm(x), y


def _balanced_subset(dataset, labels, k, rng, num_classes):
    """``k // num_classes`` random samples of each class (at least one)."""
    per_cls = max(1, k // num_classes)
    chosen = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        chosen.extend(idx[:per_cls].tolist())
    return Subset(dataset, np.asarray(chosen, dtype=int))


def dataset_name(params: dict) -> str:
    return params.get('dataset', 'cifar10')


# --------------------------------------------------------------------------- CIFAR-10

def _cifar_transform():
    return Compose([Resize(INPUT_SHAPE[1:]), ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# --------------------------------------------------------------------------- ImageNet

class _ImageList(Dataset):
    """``(path, label)`` pairs read as RGB PIL images."""

    def __init__(self, samples):
        self.samples = samples
        self.targets = [label for _, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            return img.convert('RGB'), label


def _class_folders(root: str):
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"ImageNet folder '{root}' not found. Expected <data_path>/train/<class>/ and "
            f"<data_path>/val/<class>/ (see core/data.py).")
    return sorted(d.name for d in os.scandir(root) if d.is_dir())


def _imagenet_selection(params: dict):
    """All class folders and the indices of the ones in use (``imagenet_classes``)."""
    classes = _class_folders(os.path.join(params['data_path'], 'train'))
    n_wanted = int(params.get('imagenet_classes', 0))
    if 0 < n_wanted < len(classes):
        selected = np.unique(np.linspace(0, len(classes) - 1, n_wanted).round().astype(int)).tolist()
    else:
        selected = list(range(len(classes)))
    return classes, selected


def _imagenet_samples(params: dict, split: str):
    """``(path, new_label)`` for the selected classes of ``split`` ('train' or 'val')."""
    classes, selected = _imagenet_selection(params)
    root = os.path.join(params['data_path'], split)
    available = set(_class_folders(root))
    missing = [classes[i] for i in selected if classes[i] not in available]
    if missing:
        raise FileNotFoundError(f"{len(missing)} class folders missing in '{root}', "
                                f"e.g. {missing[:3]}. train/ and val/ need the same classes.")
    samples = []
    for new_label, idx in enumerate(selected):
        folder = os.path.join(root, classes[idx])
        files = sorted(e.path for e in os.scandir(folder)
                       if e.is_file() and e.name.lower().endswith(_IMG_EXTENSIONS))
        samples.extend((path, new_label) for path in files)
    if not samples:
        raise FileNotFoundError(f"No images found under '{root}'.")
    return samples


def _imagenet_transform(params: dict):
    """The pretrained model's own eval preprocessing (resize, center crop, normalization),
    read from timm; falls back to resize 256 / crop 224 / ImageNet statistics."""
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
        model = timm.create_model(params['vit_model_name'], pretrained=False)
        cfg = resolve_model_data_config(model)
        return create_transform(**cfg, is_training=False)
    except Exception as exc:  # pragma: no cover - only without a usable timm
        print(f"[data] timm preprocessing unavailable ({exc}); using resize 256 + crop 224")
        from torchvision.transforms import CenterCrop
        return Compose([Resize(256), CenterCrop(INPUT_SHAPE[1:]), ToTensor(),
                        Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# --------------------------------------------------------------------------- public API

def num_classes(params: dict) -> int:
    """Number of outputs of the classifier for the configured dataset."""
    if dataset_name(params) == 'imagenet':
        return len(_imagenet_selection(params)[1])
    return 10


def head_classes(params: dict):
    """ImageNet-1k classifier outputs to keep from the pretrained head, or ``None``
    for a new, randomly initialized head.

    Only ImageNet with exactly 1000 class folders (in ImageNet-1k order) reuses
    the pretrained head; ``imagenet_pretrained_head: false`` turns that off.
    """
    if dataset_name(params) != 'imagenet' or not params.get('imagenet_pretrained_head', True):
        return None
    classes, selected = _imagenet_selection(params)
    return selected if len(classes) == IMAGENET_1K else None


def build_datasets(params: dict):
    """Train/val datasets: a stratified split of the dataset's *train* split, optionally
    limited to ``limit_data_value`` images in total (class-balanced on both sides).

    Never touches the test set.
    """
    name = dataset_name(params)
    if name == 'cifar10':
        raw = tvd.CIFAR10(params['data_path'], train=True, download=True, transform=None)
        tfm = _cifar_transform()
    elif name == 'imagenet':
        raw = _ImageList(_imagenet_samples(params, 'train'))
        tfm = _imagenet_transform(params)
    else:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}.")
    labels = np.asarray(raw.targets).astype(int)
    n_classes = int(labels.max()) + 1

    train_split, split_seed = params['train_split'], params['split_seed']
    limit_data_value = params['limit_data_value']
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_split, random_state=split_seed)
    train_idx, val_idx = next(sss.split(np.arange(len(labels)), labels))
    train_ds, val_ds = Subset(raw, train_idx), Subset(raw, val_idx)

    if 0 < limit_data_value < len(train_ds) + len(val_ds):
        k_train = min(max(1, int(round(limit_data_value * train_split))), len(train_ds))
        k_val = min(max(1, limit_data_value - k_train), len(val_ds))
        rng = np.random.default_rng(split_seed)
        train_ds = _balanced_subset(train_ds, labels[train_idx], k_train, rng, n_classes)
        val_ds = _balanced_subset(val_ds, labels[val_idx], k_val, rng, n_classes)

    return _TransformWrapper(train_ds, tfm), _TransformWrapper(val_ds, tfm)


def build_test_dataset(params: dict):
    """The held-out test set: CIFAR-10's official test set (10,000 images) or
    ImageNet's official validation set (50,000 images, fewer with ``imagenet_classes``).

    Reserved for the final evaluation in ``retrain.py``: no other phase may call this.
    """
    if dataset_name(params) == 'imagenet':
        return _TransformWrapper(_ImageList(_imagenet_samples(params, 'val')),
                                 _imagenet_transform(params))
    return tvd.CIFAR10(params['data_path'], train=False, download=True, transform=_cifar_transform())


def build_loaders(params: dict, device: str):
    """Train/val loaders sharing one seeded generator (reseeded per candidate)."""
    train_ds, val_ds = build_datasets(params)
    generator = torch.Generator()
    generator.manual_seed(int(params['loader_seed']))
    common = dict(num_workers=int(params['num_workers']),
                  pin_memory=device.startswith('cuda'), generator=generator)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=params['eval_batch_size'], shuffle=False, **common)
    return train_loader, val_loader
