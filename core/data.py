"""CIFAR-10 served at 224x224 with ImageNet normalization for the pretrained ViT.

Data protocol (the same for every phase):

- **train**: a stratified ``train_split`` share of the CIFAR-10 train set
  (``split_seed``). DARTS trains the MLP + classifier weights on it, the search
  trains every candidate on it, and ``retrain.py`` trains the final model on it.
- **val**: the rest of the CIFAR-10 train set. DARTS updates the alphas on it,
  the search scores candidate accuracy on it, and ``retrain.py`` picks the best
  epoch on it.
- **test**: the official CIFAR-10 test set, a separate file. It is reachable
  only through ``build_test_dataset``, which only ``retrain.py`` calls, once,
  after training, for the final evaluation.
"""
import numpy as np
import torch
import torchvision.datasets as tvd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

NUM_CLASSES = 10
INPUT_SHAPE = (3, 224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _transform():
    return Compose([Resize(INPUT_SHAPE[1:]), ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class _TransformWrapper(Dataset):
    """Apply ``tfm`` lazily to the images of ``subset``."""

    def __init__(self, subset, tfm):
        self.subset, self.tfm = subset, tfm

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]
        return self.tfm(x), y


def _balanced_subset(dataset, labels, k, rng):
    """``k // NUM_CLASSES`` random samples of each class."""
    per_cls = max(1, k // NUM_CLASSES)
    chosen = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        chosen.extend(idx[:per_cls].tolist())
    return Subset(dataset, np.asarray(chosen, dtype=int))


def build_datasets(data_path: str, train_split: float, split_seed: int, limit_data_value: int):
    """Train/val datasets: a stratified split of the CIFAR-10 *train* set, optionally
    limited to ``limit_data_value`` images in total (class-balanced on both sides).

    Never touches the test set.
    """
    raw = tvd.CIFAR10(data_path, train=True, download=True, transform=None)
    labels = np.asarray(raw.targets).astype(int)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_split, random_state=split_seed)
    train_idx, val_idx = next(sss.split(np.arange(len(labels)), labels))
    train_ds, val_ds = Subset(raw, train_idx), Subset(raw, val_idx)

    if 0 < limit_data_value < len(train_ds) + len(val_ds):
        k_train = min(max(1, int(round(limit_data_value * train_split))), len(train_ds))
        k_val = min(max(1, limit_data_value - k_train), len(val_ds))
        rng = np.random.default_rng(split_seed)
        train_ds = _balanced_subset(train_ds, labels[train_idx], k_train, rng)
        val_ds = _balanced_subset(val_ds, labels[val_idx], k_val, rng)

    tfm = _transform()
    return _TransformWrapper(train_ds, tfm), _TransformWrapper(val_ds, tfm)


def build_test_dataset(data_path: str):
    """The official CIFAR-10 test set (10,000 images).

    Reserved for the final evaluation in ``retrain.py``: no other phase may call this.
    """
    return tvd.CIFAR10(data_path, train=False, download=True, transform=_transform())


def build_loaders(params: dict, device: str):
    """Train/val loaders sharing one seeded generator (reseeded per candidate)."""
    train_ds, val_ds = build_datasets(params['data_path'], params['train_split'],
                                      params['split_seed'], params['limit_data_value'])
    generator = torch.Generator()
    generator.manual_seed(int(params['loader_seed']))
    common = dict(num_workers=int(params['num_workers']),
                  pin_memory=device.startswith('cuda'), generator=generator)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=params['eval_batch_size'], shuffle=False, **common)
    return train_loader, val_loader
