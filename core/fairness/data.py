# moq-nas/core/fairness/data.py

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- Square helpers (new) ----------

class PadToSquare(object):
    """Letterbox pad to square with a given fill color (no content crop)."""
    def __init__(self, fill=(0, 0, 0)):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        # ImageOps.pad keeps aspect by padding/cropping; we just pad manually
        new_img = Image.new("RGB", (side, side), self.fill)
        new_img.paste(img, ((side - w) // 2, (side - h) // 2))
        return new_img

def make_transforms(img_size: int = 224, mode: str = "letterbox", train: bool = True) -> transforms.Compose:
    """
    mode: 'letterbox' -> center-crop square then resize
          'letterbox'   -> pad to square then resize (no crop)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    square_ops: List[transforms.Transform] = []
    if mode == "letterbox":
        # If the input is non-square (e.g., 224x244), crop to square around center
        # Then resize to desired side.
        square_ops = [
            transforms.CenterCrop(min(224, 244)),  # safe default if originals are 224x244; works for larger too
            transforms.Resize((img_size, img_size)),
        ]
    elif mode == "letterbox":
        square_ops = [
            PadToSquare(fill=(0, 0, 0)),
            transforms.Resize((img_size, img_size)),
        ]
    else:
        raise ValueError("mode must be 'letterbox' or 'letterbox'")

    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
        ]
    else:
        aug = []  # deterministic for eval/fairness

    return transforms.Compose(
        [transforms.ToTensor()] +  # convert first so PadToSquare can come before or after; here we keep it PIL -> Tensor at end
        # NOTE: keeping PIL pipeline until ToTensor() is actually better, so reorder:
        # We'll rebuild below properly as PIL->PIL->...->ToTensor().
        []
    )

# Rebuild make_transforms to ensure proper PIL ordering
def make_transforms(img_size: int = 224, mode: str = "letterbox", train: bool = True) -> transforms.Compose:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if mode == "letterbox":
        squaring = [transforms.CenterCrop( min(img_size*3, 1024) )]  # wide crop guard; replaced below per dataset
        # We’ll do a pragmatic approach: center-crop to square using the shorter side of the current image at runtime
        # torchvision doesn't have dynamic min-side center-crop; but CenterCrop(s) works if s <= min(W,H).
        # To be robust across sizes, we do a two-step: short Resize so min side >= img_size, then CenterCrop(img_size), then final resize (no-op).
        squaring = [
            transforms.Resize(img_size if img_size >= 224 else 224),  # ensure min side reasonably large
            transforms.CenterCrop(img_size),
        ]
    elif mode == "letterbox":
        squaring = [
            PadToSquare(fill=(0, 0, 0)),
            transforms.Resize((img_size, img_size)),
        ]
    else:
        raise ValueError("mode must be 'letterbox' or 'letterbox'")

    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
        ]
    else:
        aug = []  # deterministic for eval/fairness

    return transforms.Compose(
        squaring
        + aug
        + [
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

# ---------- 1. Dataset Classes (unchanged except small tidy) ----------

class BinaryFolderDataset(Dataset):
    """Dataset for binary classification from a folder structure."""
    def __init__(self, root: str, split: str = "train", tfm: Optional[transforms.Compose] = None,
                 pos_name: Optional[str] = None, neg_name: Optional[str] = None):

        self.root = Path(root) / split
        if not self.root.is_dir():
            raise FileNotFoundError(f"Directory for split '{split}' not found at: {self.root}")

        self.tfm = tfm
        self.samples: List[Tuple[Path, int]] = []

        if pos_name is None or neg_name is None:
            subdirs = {d.name for d in self.root.iterdir() if d.is_dir()}
            if {"person", "non_person"}.issubset(subdirs):
                pos_name, neg_name = "person", "non_person"
            elif {"face", "non_face"}.issubset(subdirs):
                pos_name, neg_name = "face", "non_face"
            else:
                raise ValueError(f"Could not auto-detect class folders in {self.root}.")

        exts = {".jpg", ".jpeg", ".png"}
        for label, class_name in [(1, pos_name), (0, neg_name)]:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            for p in class_dir.rglob("*"):
                if p.suffix.lower() in exts:
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
        return img, label


class FacetEvalDataset(Dataset):
    """
    Dataset for fairness evaluation on FACET (skin tone).
    It loads images, crops faces, and can use an on-disk cache to speed up re-runs.
    """
    def __init__(self, csv_path: str, tfm: Optional[transforms.Compose] = None, cache_dir: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        self.tfm = tfm
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        required_cols = {"image_path", "x", "y", "width", "height", "skin_tone_probs"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"FACET CSV must contain columns: {required_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        if self.cache_dir:
            p = Path(row["image_path"])
            cached_filename = f"{p.stem}_{int(row['x'])}_{int(row['y'])}_{int(row['width'])}_{int(row['height'])}.jpg"
            cached_path = self.cache_dir / cached_filename

            if cached_path.exists():
                img = Image.open(cached_path).convert("RGB")
            else:
                img = Image.open(row["image_path"]).convert("RGB")
                box = (row["x"], row["y"], row["x"] + row["width"], row["y"] + row["height"])
                img = img.crop(box)
                try:
                    img.save(cached_path, quality=95)
                except Exception:
                    pass
        else:
            img_path = row["image_path"]
            box = (row["x"], row["y"], row["x"] + row["width"], row["y"] + row["height"])
            img = Image.open(img_path).convert("RGB").crop(box)

        if self.tfm:
            img = self.tfm(img)

        soft_labels = torch.tensor(json.loads(row["skin_tone_probs"]), dtype=torch.float32)
        return img, soft_labels


class FairFaceEvalDataset(Dataset):
    """Dataset for fairness evaluation on FairFace (race)."""
    def __init__(self, csv_path: str, tfm: Optional[transforms.Compose] = None):
        self.df = pd.read_csv(csv_path)
        self.tfm = tfm
        required_cols = {"image_path", "race"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"FairFace CSV must contain columns: {required_cols}")

        self.race_labels = sorted(self.df['race'].unique())
        self.race_to_idx = {race: i for i, race in enumerate(self.race_labels)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        race = row["race"]

        img = Image.open(img_path).convert("RGB")
        if self.tfm:
            img = self.tfm(img)

        label_idx = self.race_to_idx[race]
        return img, label_idx

# ---------- 2. Factories ----------

def get_default_transforms(img_size: int = 224) -> dict:
    """
    Backward-compatible wrapper; uses center-crop squaring.
    For small sizes (e.g., 96/128), returns sensible train/eval pipelines.
    """
    tf_train = make_transforms(img_size=img_size, mode="letterbox", train=True)
    tf_val   = make_transforms(img_size=img_size, mode="letterbox", train=False)
    return {'train': tf_train, 'val': tf_val}

def create_binary_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    tf_train: transforms.Compose,
    tf_val: transforms.Compose,
    pos_name: str = None,
    neg_name: str = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = BinaryFolderDataset(root=data_root, split='train', tfm=tf_train, pos_name=pos_name, neg_name=neg_name)
    val_dataset   = BinaryFolderDataset(root=data_root, split='val',   tfm=tf_val,   pos_name=pos_name, neg_name=neg_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def create_eval_loader(
    dataset_name: str,
    csv_path: str,
    batch_size: int,
    img_size: int = 224,
    cache_dir: Optional[str] = ".cache/facet_crops",
    square_mode: str = "letterbox",   # new: choose squaring for fairness eval
) -> DataLoader:
    """
    Creates a DataLoader for fairness evaluation datasets with deterministic transforms.
    """
    tf_eval = make_transforms(img_size=img_size, mode=square_mode, train=False)

    if dataset_name.lower() == 'facet':
        dataset = FacetEvalDataset(csv_path, tfm=tf_eval, cache_dir=cache_dir)
    elif dataset_name.lower() == 'fairface':
        dataset = FairFaceEvalDataset(csv_path, tfm=tf_eval)
    else:
        raise ValueError(f"Evaluation dataset '{dataset_name}' is not supported. Use 'facet' or 'fairface'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
