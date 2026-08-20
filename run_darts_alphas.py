"""Phase 1: learn the attention-head importances (alphas) with DARTS.

Reuses ``vit_transformer_search.py`` unchanged (``build_darts_vit`` and
``train_darts_epoch``) and writes the resulting per-head alphas to JSON.
The genetic search (``run_all_evolution.py --algo nsga3`` with a ViT
config) then reads that file to decide which heads a given pruning
percentage keeps.

    python run_darts_alphas.py --epochs 1 --limit_train 2000 \
        --output darts_alphas/vit_base_cifar10.json
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from core.vit import extract_alphas, save_alphas
from vit_transformer_search import build_darts_vit, train_darts_epoch


def _resolve_device() -> torch.device:
    """CUDA if present, else Apple MPS, else CPU (same order as the search)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main(args):
    device = _resolve_device()
    print(f"[darts] device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)

    # DARTS needs a train/val split: weights are updated on train, the
    # architecture alphas on val (that separation is the point of DARTS).
    if args.limit_train > 0 and args.limit_train < len(full):
        generator = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(full), generator=generator)[:args.limit_train].tolist()
        full = Subset(full, idx)

    train_size = len(full) // 2
    val_size = len(full) - train_size
    train_ds, val_ds = random_split(
        full, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=pin)
    print(f"[darts] train={len(train_ds)} val={len(val_ds)} batch={args.batch_size}")

    model, weight_params, alpha_params = build_darts_vit(
        model_name=args.model_name, num_classes=10)
    model = model.to(device)

    optimizer_w = torch.optim.AdamW(weight_params, lr=1e-3, weight_decay=1e-4)
    optimizer_alpha = torch.optim.Adam(alpha_params, lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        print(f"\n--- Época {epoch + 1}/{args.epochs} ---")
        train_darts_epoch(model=model, train_loader=train_loader, val_loader=val_loader,
                          optimizer_w=optimizer_w, optimizer_alpha=optimizer_alpha,
                          criterion=criterion, device=device)

    alphas = extract_alphas(model)
    save_alphas(args.output, alphas)
    print(f"\n[darts] alphas salvos em {args.output}")
    for i, block_alphas in enumerate(alphas):
        ranked = sorted(range(len(block_alphas)), key=lambda h: block_alphas[h], reverse=True)
        print(f"  bloco {i:02d}: cabeças por importância {ranked}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=str, default='darts_alphas/vit_base_cifar10.json',
                        help='Where to write the alphas JSON.')
    parser.add_argument('--data_path', type=str, default='data',
                        help='CIFAR-10 root directory.')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--limit_train', type=int, default=0,
                        help='Use only this many training images (0 = full CIFAR-10 train set). '
                             'A small value makes the DARTS phase feasible on a laptop.')
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
