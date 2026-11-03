"""Train and score one candidate: accuracy (maximize) and FLOPs (minimize)."""
import time

import torch
import torch.nn as nn

from core.data import INPUT_SHAPE, head_classes, num_classes
from core.vit import build_pruned_vit, load_alphas


FLOPS_COUNTERS = ('hooks', 'pytorch')


def _count_flops_hooks(model: nn.Module, device: str) -> int:
    """The project's original counter: Conv2d and Linear layers only (2 x multiply-adds),
    via forward hooks. The attention matrix products (q @ k^T, attn @ v), softmax,
    GELU and LayerNorm are not counted, so it undercounts by ~4% on ViT-Base."""
    total = 0

    def hook(module, inp, out):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            k_h, k_w = module.kernel_size
            out_h, out_w = out.shape[-2:]
            total += (module.in_channels // module.groups) * module.out_channels \
                * k_h * k_w * out_h * out_w * 2
        else:
            # A Linear runs once per token, i.e. per leading dim of its output.
            rows = 1
            for dim in out.shape[:-1]:
                rows *= int(dim)
            total += module.in_features * module.out_features * rows * 2

    hooks = [m.register_forward_hook(hook) for m in model.modules()
             if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros((1, *INPUT_SHAPE), device=device))
    for h in hooks:
        h.remove()
    return int(total)


def _count_flops_pytorch(model: nn.Module, device: str) -> int:
    """PyTorch's own counter (``torch.utils.flop_counter.FlopCounterMode``): every matrix
    multiplication and convolution the forward pass runs, including the attention products
    (also inside ``scaled_dot_product_attention``), at 2 FLOPs per multiply-add.
    Element-wise ops (softmax, GELU, LayerNorm, residual adds) are not counted.
    Halve the result to get MACs, the unit many ViT papers call "FLOPs"."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(torch.zeros((1, *INPUT_SHAPE), device=device))
    return int(counter.get_total_flops())


def count_flops(model: nn.Module, device: str, method: str = 'hooks') -> int:
    """FLOPs of one forward pass on a single image (2 FLOPs per multiply-add).

    ``method``: ``'hooks'`` (the original counter, Linear/Conv only) or ``'pytorch'``
    (``FlopCounterMode``, also counts the attention products). Choose it with
    ``flops_counter`` in the config; compare results only within one method.
    """
    if method == 'hooks':
        return _count_flops_hooks(model, device)
    if method == 'pytorch':
        return _count_flops_pytorch(model, device)
    raise ValueError(f"Unknown FLOPs counter {method!r}; expected one of {FLOPS_COUNTERS}.")


def _format_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def run_epoch_stats(model, loader, device, criterion, optimizer=None,
                    log_every: int = 0, desc: str = '') -> dict:
    """One pass over ``loader``; trains when ``optimizer`` is given.

    Returns ``{'loss', 'acc1', 'acc5'}`` (accuracies in %, top-5 = the label is
    among the 5 highest scores). With ``log_every > 0``, prints the running
    loss/top-1/top-5, the elapsed time and an ETA every ``log_every`` batches.
    """
    training = optimizer is not None
    model.train(training)
    correct1 = correct5 = total = 0
    loss_sum = 0.0
    n_batches = len(loader)
    start = time.perf_counter()
    with torch.enable_grad() if training else torch.no_grad():
        for step, (inputs, labels) in enumerate(loader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            top5 = outputs.topk(min(5, outputs.size(1)), dim=1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += (top5 == labels.view(-1, 1)).any(dim=1).sum().item()
            loss_sum += loss.item() * labels.size(0)
            total += labels.size(0)
            if log_every and (step % log_every == 0 or step == n_batches):
                elapsed = time.perf_counter() - start
                eta = elapsed / step * (n_batches - step)
                print(f"{desc}step {step}/{n_batches} ({100 * step / n_batches:.0f}%) | "
                      f"loss {loss_sum / total:.4f} acc@1 {100 * correct1 / total:.2f}% "
                      f"acc@5 {100 * correct5 / total:.2f}% | {_format_seconds(elapsed)} "
                      f"(faltam ~{_format_seconds(eta)})", flush=True)
    if not total:
        return {'loss': 0.0, 'acc1': 0.0, 'acc5': 0.0}
    return {'loss': loss_sum / total, 'acc1': 100 * correct1 / total, 'acc5': 100 * correct5 / total}


def _run_epoch(model, loader, device, criterion, optimizer=None) -> float:
    """One pass over ``loader``; trains when ``optimizer`` is given. Returns top-1 accuracy (%)."""
    return run_epoch_stats(model, loader, device, criterion, optimizer)['acc1']


def build_candidate(percentages, params: dict, device: str) -> nn.Module:
    """Pretrained ViT pruned as ``percentages`` says: one head percentage per block,
    followed by one MLP percentage per block when ``prune_mlp`` is on."""
    n_blocks = params['num_blocks']
    return build_pruned_vit(percentages[:n_blocks], load_alphas(params['vit_alphas_path']),
                            num_classes(params),
                            mlp_percentages=percentages[n_blocks:] if params['prune_mlp'] else None,
                            model_name=params['vit_model_name'],
                            pretrained=params['vit_pretrained'],
                            head_classes=head_classes(params)).to(device)


def make_optimizer(model: nn.Module, params: dict) -> torch.optim.Optimizer:
    """AdamW over the classifier and the MLPs, the only trainable parts.

    The new classifier needs a large learning rate; the pretrained MLPs a small one,
    or a few steps wipe out the pretrained features.
    """
    return torch.optim.AdamW(
        [{'params': model.get_classifier().parameters(), 'lr': float(params['learning_rate'])},
         {'params': [p for b in model.blocks for p in b.mlp.parameters()],
          'lr': float(params['mlp_learning_rate'])}],
        weight_decay=float(params['weight_decay']))


def evaluate_candidate(percentages, params: dict, device: str, train_loader, val_loader) -> dict:
    """Prune, fine-tune the MLPs and the classifier, and return the two objectives.

    ``best_accuracy`` is the mean validation accuracy over the last
    ``epochs_to_eval`` epochs.
    """
    model = build_candidate(percentages, params, device)
    optimizer = make_optimizer(model, params)
    criterion = nn.CrossEntropyLoss()

    max_epochs, epochs_to_eval = params['max_epochs'], params['epochs_to_eval']
    val_accuracies = []
    for epoch in range(1, max_epochs + 1):
        _run_epoch(model, train_loader, device, criterion, optimizer)
        if epoch > max_epochs - epochs_to_eval:
            val_accuracies.append(_run_epoch(model, val_loader, device, criterion))

    return {'best_accuracy': sum(val_accuracies) / len(val_accuracies),
            'total_flops': count_flops(model, device, params['flops_counter'])}
