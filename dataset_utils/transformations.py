# dataset_utils/transformations.py
from torchvision.transforms import (
    Compose, ToTensor, Normalize, Resize,
    TrivialAugmentWide, RandomResizedCrop, RandomHorizontalFlip, CenterCrop
)

def build_transforms(spec, data_augmentation: bool):
    """
    spec: DatasetSpec or object with .name .shape .mean .std
    Return: (train_transform, eval_transform)
    """
    ds_name = (spec.name or "").lower()

    # --- Heads (PIL-space ops) ---
    train_head, eval_head = [], []

    if "atleta" in ds_name:
        # spec.shape expected as (C, H, W)
        _, h, w = spec.shape
        resize = Resize((h, w))
        train_head.append(resize)
        eval_head.append(resize)

        if data_augmentation:
            train_head.append(TrivialAugmentWide(num_magnitude_bins=31))

    elif "person" in ds_name or "face" in ds_name:
        t = getattr(spec, "transform", None)
        img_size = int(t.get("img_size", spec.shape[1])) if isinstance(t, dict) else int(spec.shape[1])

        # Eval (and non-aug train) use standard resize + center crop
        eval_head.extend([Resize(int(img_size * 256 / 224)), CenterCrop(img_size)])

        if data_augmentation:
            # Stronger train-time aug for face/person
            train_head.extend([
                RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                RandomHorizontalFlip(),
                TrivialAugmentWide(num_magnitude_bins=31),
            ])
        else:
            # Mirror eval geometry if no aug
            train_head.extend(eval_head)

    else:
        # Generic datasets: optionally add light augmentation to train
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