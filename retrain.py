"""Final step: retrain one searched architecture and evaluate it ONCE on the test set.

The architecture comes from ``pareto_history.pkl`` (``--id``, e.g. ``1_3``) or is
given directly (``--chromosome``). It is pruned exactly as in the search, trains
its MLPs + classifier on the train split for ``--epochs``, and the epoch with the
best validation accuracy is kept. Only then is the official CIFAR-10 test set
loaded, for a single evaluation of that model. No phase before this one reads
the test set, so the test accuracy is an unbiased estimate.

    python retrain.py --config_file config.yaml \
        --pareto_history experiments/exp1/pareto_history.pkl --id 1_3 --epochs 10
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.config import load_config
from core.data import build_loaders, build_test_dataset
from core.training import _format_seconds, build_candidate, count_flops, make_optimizer, run_epoch_stats
from core.utils import resolve_device, set_global_seeds


def find_chromosome(history_path: str, candidate_id: str):
    """Chromosome of ``candidate_id``: looked up in the Pareto fronts (latest generation
    first), then among all evaluated candidates (histories that store them)."""
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    for key in ('front', 'evaluated'):
        for generation in sorted(history, reverse=True):
            for point in history[generation].get(key, []):
                if point['id'] == candidate_id:
                    return point['chromosome'], point
    raise KeyError(f"Candidate '{candidate_id}' is not in {history_path}.")


def main(args):
    # Line-buffered output, so progress shows up immediately when piped to `tee`.
    sys.stdout.reconfigure(line_buffering=True)
    set_global_seeds(args.seed)
    params = load_config({'config_file': args.config_file, 'seed': args.seed,
                          'data_path': args.data_path,
                          'limit_data_value': args.limit_data_value})

    if args.chromosome is not None:
        chromosome, search_point, name = args.chromosome, None, 'custom'
    else:
        chromosome, search_point = find_chromosome(args.pareto_history, args.id)
        name = args.id
    if len(chromosome) != params['num_genes']:
        raise ValueError(f"Chromosome has {len(chromosome)} genes but {args.config_file} "
                         f"expects {params['num_genes']} (num_blocks x "
                         f"{'2' if params['prune_mlp'] else '1'}).")
    percentages = [params['percentages'][g] for g in chromosome]

    device = resolve_device()
    train_loader, val_loader = build_loaders(params, device)
    print(f"[retrain] {name}: percentages={percentages}")
    print(f"[retrain] device={device} train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} epochs={args.epochs}")

    model = build_candidate(percentages, params, device)
    optimizer = make_optimizer(model, params)
    criterion = nn.CrossEntropyLoss()

    history, best_val, best_epoch, best_state = [], -1.0, 0, None
    run_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        print(f"\n[retrain] --- epoch {epoch}/{args.epochs} ---", flush=True)
        train = run_epoch_stats(model, train_loader, device, criterion, optimizer,
                                log_every=args.log_every, desc=f"[retrain] epoch {epoch} treino ")
        val = run_epoch_stats(model, val_loader, device, criterion)
        history.append({'epoch': epoch, 'train': train, 'val': val})
        improved = val['acc1'] > best_val
        if improved:
            best_val, best_epoch = val['acc1'], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        done = time.perf_counter() - run_start
        eta = done / epoch * (args.epochs - epoch)
        print(f"[retrain] epoch {epoch}/{args.epochs}: "
              f"treino loss={train['loss']:.4f} acc@1={train['acc1']:.2f}% acc@5={train['acc5']:.2f}% | "
              f"val loss={val['loss']:.4f} acc@1={val['acc1']:.2f}% acc@5={val['acc5']:.2f}%"
              f"{'  <- melhor até agora' if improved else ''}", flush=True)
        print(f"[retrain] progresso: {epoch}/{args.epochs} épocas ({100 * epoch / args.epochs:.0f}%) | "
              f"época em {_format_seconds(time.perf_counter() - epoch_start)} | "
              f"total {_format_seconds(done)} | faltam ~{_format_seconds(eta)}", flush=True)

    # Final evaluation: the first and only time the test set is loaded.
    model.load_state_dict(best_state)
    test_loader = DataLoader(build_test_dataset(params['data_path']),
                             batch_size=params['eval_batch_size'], shuffle=False,
                             num_workers=params['num_workers'],
                             pin_memory=device.startswith('cuda'))
    test = run_epoch_stats(model, test_loader, device, criterion)
    test_acc = test['acc1']

    results = {
        'id': name,
        'chromosome': list(chromosome),
        'percentages': percentages,
        'search_objectives': search_point and {k: search_point[k]
                                               for k in ('best_accuracy', 'total_flops')},
        'epochs': args.epochs,
        'train_images': len(train_loader.dataset),
        'val_accuracy_per_epoch': [h['val']['acc1'] for h in history],
        'history': history,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val,
        'best_val_top5': history[best_epoch - 1]['val']['acc5'],
        'test_accuracy': test_acc,
        'test_top5': test['acc5'],
        'test_loss': test['loss'],
        'train_time_seconds': time.perf_counter() - run_start,
        'test_images': len(test_loader.dataset),
        'total_flops': count_flops(model, device),
        'total_params': sum(p.numel() for p in model.parameters()),
        'config_file': args.config_file,
        'seed': args.seed,
    }
    out_dir = os.path.join(args.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(best_state, os.path.join(out_dir, 'model_state.pt'))

    print(f"\n[retrain] {name}: best epoch {best_epoch} (val {best_val:.2f}%) -> "
          f"TEST acc@1 {test_acc:.2f}% acc@5 {test['acc5']:.2f}% on {len(test_loader.dataset)} images | "
          f"FLOPs {results['total_flops'] / 1e9:.2f} G | params {results['total_params'] / 1e6:.1f} M")
    print(f"[retrain] saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config_file', type=str, required=True,
                        help='The same config used by the search (alphas, prune_mlp, percentages).')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--id', type=str, help="Candidate id in the Pareto history, e.g. '1_3'.")
    source.add_argument('--chromosome', type=int, nargs='+', help='Genes, e.g. 3 5 7 ...')
    parser.add_argument('--pareto_history', type=str, default=None,
                        help='pareto_history.pkl of the search (required with --id).')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--limit_data_value', type=int, default=0,
                        help='Train + val images, class-balanced (0 = the whole split).')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='retrain_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=100,
                        help='Print training progress every N batches (0 = only per epoch).')
    parsed = parser.parse_args()
    if parsed.id is not None and parsed.pareto_history is None:
        parser.error('--pareto_history is required with --id')
    main(parsed)
