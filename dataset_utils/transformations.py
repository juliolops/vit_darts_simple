# dataset_utils/transformations.py
from torchvision.transforms import Compose, ToTensor, Normalize, TrivialAugmentWide

def build_transforms(spec, data_augmentation: bool):
    """
    spec: DatasetSpec or object with .name .shape .mean .std
    Return: (train_transform, eval_transform)
    """
    # --- Heads (PIL-space ops) ---
    # Generic datasets (cifar10): optionally add light augmentation to train.
    train_head, eval_head = [], []
    if data_augmentation:
        train_head.append(TrivialAugmentWide(num_magnitude_bins=31))
    # eval_head stays empty (identity) unless you add dataset-specific geometry

    # --- Shared tail (tensor-space ops) ---
    tail = [ToTensor()]
    if getattr(spec, "mean", None) is not None and getattr(spec, "std", None) is not None:
        tail.append(Normalize(mean=spec.mean, std=spec.std))

    train_tf = Compose([*train_head, *tail])
    eval_tf  = Compose([*eval_head,  *tail])
    return train_tf, eval_tf