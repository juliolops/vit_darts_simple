"""Train and score one candidate: accuracy (maximize) and FLOPs (minimize)."""
import torch
import torch.nn as nn

from core.data import INPUT_SHAPE, NUM_CLASSES
from core.vit import build_pruned_vit, load_alphas


def count_flops(model: nn.Module, device: str) -> int:
    """FLOPs of one forward pass on a single image, counting Conv2d and Linear
    layers (2 x multiply-adds). The attention matrix products are not counted."""
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


def _run_epoch(model, loader, device, criterion, optimizer=None) -> float:
    """One pass over ``loader``; trains when ``optimizer`` is given. Returns accuracy (%)."""
    training = optimizer is not None
    model.train(training)
    correct = total = 0
    with torch.enable_grad() if training else torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if training:
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total if total else 0.0


def evaluate_candidate(percentages, params: dict, device: str, train_loader, val_loader) -> dict:
    """Prune, fine-tune the MLPs and the classifier, and return the two objectives.

    ``percentages`` holds one head percentage per block, followed by one MLP
    percentage per block when ``prune_mlp`` is on. ``best_accuracy`` is the
    mean validation accuracy over the last ``epochs_to_eval`` epochs.
    """
    n_blocks = params['num_blocks']
    model = build_pruned_vit(percentages[:n_blocks], load_alphas(params['vit_alphas_path']),
                             NUM_CLASSES,
                             mlp_percentages=percentages[n_blocks:] if params['prune_mlp'] else None,
                             model_name=params['vit_model_name'],
                             pretrained=params['vit_pretrained']).to(device)
    # The new classifier needs a large learning rate; the pretrained MLPs a small one,
    # or a few steps wipe out the pretrained features.
    optimizer = torch.optim.AdamW(
        [{'params': model.get_classifier().parameters(), 'lr': float(params['learning_rate'])},
         {'params': [p for b in model.blocks for p in b.mlp.parameters()],
          'lr': float(params['mlp_learning_rate'])}],
        weight_decay=float(params['weight_decay']))
    criterion = nn.CrossEntropyLoss()

    max_epochs, epochs_to_eval = params['max_epochs'], params['epochs_to_eval']
    val_accuracies = []
    for epoch in range(1, max_epochs + 1):
        _run_epoch(model, train_loader, device, criterion, optimizer)
        if epoch > max_epochs - epochs_to_eval:
            val_accuracies.append(_run_epoch(model, val_loader, device, criterion))

    return {'best_accuracy': sum(val_accuracies) / len(val_accuracies),
            'total_flops': count_flops(model, device)}
