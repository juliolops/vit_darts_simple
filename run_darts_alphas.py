"""Phase 1: learn the importances (alphas) of attention heads and MLP neurons with DARTS.

Uses ``build_darts_vit`` and ``train_darts_epoch`` from
``vit_transformer_search.py``: on each step the alphas are updated on a
validation batch and the MLP + classifier weights on a training batch, while
the attention layers stay pretrained. The resulting per-head and per-neuron
weights are written to JSON; the genetic search (``run_all_evolution.py``)
then reads that file to decide which heads and neurons a pruning percentage keeps.

The train/val split comes from ``config.yaml`` (``train_split``, ``split_seed``),
so DARTS and the search use the same images. The CIFAR-10 test set is never read.

    python run_darts_alphas.py --config_file config.yaml --epochs 3 --limit_data_value 10000
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.config import load_config
from core.data import build_datasets
from core.utils import resolve_device, set_global_seeds
from core.vit import extract_alphas, save_alphas
from vit_transformer_search import build_darts_vit, train_darts_epoch


def main(args):
    set_global_seeds(args.seed)
    params = load_config({'config_file': args.config_file, 'seed': args.seed,
                          'data_path': args.data_path,
                          'limit_data_value': args.limit_data_value})
    output = args.output or params['vit_alphas_path']
    device = resolve_device()
    print(f"[darts] device: {device}")

    train_ds, val_ds = build_datasets(params['data_path'], params['train_split'],
                                      params['split_seed'], params['limit_data_value'])
    common = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                  pin_memory=device.startswith('cuda'),
                  generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, **common)
    val_loader = DataLoader(val_ds, **common)
    print(f"[darts] train={len(train_ds)} val={len(val_ds)} batch={args.batch_size}")

    model, weight_params, alpha_params = build_darts_vit(
        model_name=params['vit_model_name'], num_classes=10)
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
    save_alphas(output, alphas)
    print(f"\n[darts] alphas salvos em {output}")
    for i, (head_w, mlp_w) in enumerate(zip(alphas['heads'], alphas['mlp'])):
        ranked = sorted(range(len(head_w)), key=lambda h: head_w[h], reverse=True)
        print(f"  bloco {i:02d}: cabeças por importância {ranked} | "
              f"pesos MLP min={min(mlp_w):.4f} max={max(mlp_w):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config_file', type=str, default='config.yaml',
                        help='Experiment config: ViT model, data split and alphas path.')
    parser.add_argument('--output', type=str, default=None,
                        help='Where to write the alphas JSON (default: vit_alphas_path in the config).')
    parser.add_argument('--data_path', type=str, default=None,
                        help='CIFAR-10 root directory (default: data_path in the config).')
    parser.add_argument('--limit_data_value', type=int, default=0,
                        help='Images used in total, train + val, class-balanced '
                             '(0 = the whole train/val split, 50,000 images).')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
