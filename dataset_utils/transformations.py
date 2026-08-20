# dataset_utils/transformations.py
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, TrivialAugmentWide

# Native resolution of the raw dataset images; a spec asking for anything
# larger (e.g. 224 for a ViT) gets an explicit Resize.
_CIFAR_NATIVE_HW = (32, 32)


def build_transforms(spec, data_augmentation: bool):
    """
    spec: DatasetSpec or object with .name .shape .mean .std
    Return: (train_transform, eval_transform)
    """
    # --- Heads (PIL-space ops) ---
    train_head, eval_head = [], []

    # Resize only when the spec's (H, W) differs from CIFAR-10's native 32x32,
    # i.e. upscale to the 224x224 a pretrained ViT expects.
    _, height, width = spec.shape
    if (height, width) != _CIFAR_NATIVE_HW:
        resize = Resize((height, width))
        train_head.append(resize)
        eval_head.append(resize)

    if data_augmentation:
        train_head.append(TrivialAugmentWide(num_magnitude_bins=31))

    # --- Shared tail (tensor-space ops) ---
    tail = [ToTensor()]
    if getattr(spec, "mean", None) is not None and getattr(spec, "std", None) is not None:
        tail.append(Normalize(mean=spec.mean, std=spec.std))

    train_tf = Compose([*train_head, *tail])
    eval_tf  = Compose([*eval_head,  *tail])
    return train_tf, eval_tf
