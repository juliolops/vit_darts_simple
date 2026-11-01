"""Phase 1: learn the importances (alphas) of attention heads and MLP neurons with DARTS.

Uses ``build_darts_vit`` and ``train_darts_epoch`` from
``vit_transformer_search.py``: on each step the alphas are updated on a
validation batch and the MLP + classifier weights on a training batch, while
the attention layers stay pretrained. The resulting per-head and per-neuron
weights are written to JSON; the genetic search (``run_all_evolution.py``)
then reads that file to decide which heads and neurons a pruning percentage keeps.

The train/val split comes from ``config.yaml`` (``train_split``, ``split_seed``),
so DARTS and the search use the same images. The CIFAR-10 test set is never read.

Stability safeguards (defaults below): a lower learning rate for the pretrained
MLP weights (``--lr_w 1e-4``), gradient clipping (``--grad_clip 5``), a warm-up in
which only the weights train and the alphas stay fixed (``--warmup_epochs 1``), and
the alphas of the epoch with the best validation accuracy are the ones saved, so
a late divergence cannot spoil them. Per-epoch metrics go to ``<output>.history.json``.

    python run_darts_alphas.py --config_file config.yaml --epochs 3 --limit_data_value 10000
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.config import load_config
from core.data import build_datasets
from core.utils import resolve_device, set_global_seeds
from core.vit import extract_alphas, save_alphas
from vit_transformer_search import build_darts_vit, evaluate, train_darts_epoch


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

    optimizer_w = torch.optim.AdamW(weight_params, lr=args.lr_w, weight_decay=1e-4)
    optimizer_alpha = torch.optim.Adam(alpha_params, lr=args.lr_alpha, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    # At least one epoch must update the alphas, so the warm-up is capped at epochs - 1.
    warmup = min(args.warmup_epochs, max(args.epochs - 1, 0))
    if warmup < args.warmup_epochs:
        print(f"[darts] warm-up reduzido para {warmup} época(s): --epochs={args.epochs} "
              f"precisa de ao menos 1 época atualizando os alphas.")
    print(f"[darts] lr_w={args.lr_w} lr_alpha={args.lr_alpha} grad_clip={args.grad_clip} "
          f"warmup_epochs={warmup}")

    eval_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.startswith('cuda'))
    best_acc, best_epoch, alphas, history = -1.0, None, None, []

    for epoch in range(args.epochs):
        update_alphas = epoch >= warmup
        phase = 'alphas + pesos' if update_alphas else 'warm-up: só pesos'
        print(f"\n--- Época {epoch + 1}/{args.epochs} ({phase}) ---")
        train_darts_epoch(model=model, train_loader=train_loader, val_loader=val_loader,
                          optimizer_w=optimizer_w, optimizer_alpha=optimizer_alpha,
                          criterion=criterion, device=device,
                          update_alphas=update_alphas, grad_clip=args.grad_clip)

        val_loss, val_acc = evaluate(model, eval_loader, criterion, device)
        history.append({'epoch': epoch + 1, 'update_alphas': update_alphas,
                        'val_loss': val_loss, 'val_acc': val_acc})
        print(f"[darts] época {epoch + 1}: validação completa -> loss={val_loss:.4f} acc={val_acc:.2f}%")

        # Only epochs that trained the alphas compete: warm-up alphas are still uniform.
        if update_alphas and val_acc > best_acc:
            best_acc, best_epoch, alphas = val_acc, epoch + 1, extract_alphas(model)
            print(f"[darts] nova melhor época ({best_epoch}): alphas guardados")

    save_alphas(output, alphas)
    history_path = os.path.splitext(output)[0] + '.history.json'
    with open(history_path, 'w') as f:
        json.dump({'best_epoch': best_epoch, 'best_val_acc': best_acc, 'args': vars(args),
                   'epochs': history}, f, indent=2)
    print(f"\n[darts] alphas da época {best_epoch} (val acc {best_acc:.2f}%) salvos em {output}")
    print(f"[darts] histórico por época em {history_path}")
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
    parser.add_argument('--lr_w', type=float, default=1e-4,
                        help='Learning rate of the pretrained MLP + classifier weights '
                             '(1e-3 destroys the pretrained features).')
    parser.add_argument('--lr_alpha', type=float, default=3e-4, help='Learning rate of the alphas.')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Max gradient norm for weights and alphas (0 = no clipping).')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Initial epochs that train only the weights, with the alphas fixed.')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
