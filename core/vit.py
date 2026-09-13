"""ViT pruning: attention heads driven by DARTS importances, optionally MLP neurons.

``run_darts_alphas.py`` learns one alpha per attention head and saves them;
the genetic search then evolves one gene per transformer block holding the
*percentage* of heads that block keeps, and a gene of 40% keeps the 40% of
heads with the largest alpha. So DARTS decides *which* heads matter and the
GA decides *how much* to prune. With MLP pruning enabled, a second gene per
block sets the percentage of MLP hidden neurons kept, ranked by weight magnitude.

Pruning is surgical rather than masking: ``qkv``/``proj`` (and ``fc1``/``fc2``)
are rebuilt with only the survivors, so a pruned model really is cheaper.
"""
import json
import os
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_alphas(model) -> List[List[float]]:
    """Per-block, per-head importances from a DARTS-wrapped ViT (softmaxed)."""
    return [F.softmax(b.attn.alphas.detach().float(), dim=0).cpu().tolist()
            for b in model.blocks]


def save_alphas(path: str, alphas: List[List[float]]) -> None:
    """Write ``alphas`` to ``path`` as JSON, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(alphas, f, indent=2)


def load_alphas(path: str) -> List[List[float]]:
    """Load alphas, pointing at the DARTS step if they were never generated."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"DARTS alphas not found at '{path}'. Run the DARTS phase first:\n"
            f"    python run_darts_alphas.py --output {path}")
    with open(path) as f:
        return json.load(f)


def select_heads_by_alpha(block_alphas: Sequence[float], percentage: float) -> List[int]:
    """Indices of the heads to keep, largest alpha first. Never returns empty."""
    num_heads = len(block_alphas)
    n_keep = max(1, min(num_heads, round(num_heads * float(percentage) / 100.0)))
    alphas = torch.as_tensor(list(block_alphas), dtype=torch.float32)
    return sorted(int(i) for i in torch.topk(alphas, k=n_keep).indices.tolist())


def prune_attention_heads(attn: nn.Module, kept_heads: Sequence[int]) -> None:
    """Shrink a timm ``Attention`` in place to ``kept_heads``.

    ``qkv`` packs the projections as rows ``[q | k | v]``, each holding
    ``num_heads * head_dim`` rows; ``proj`` consumes those same head slots as
    columns. Both are rebuilt from the surviving slices. ``head_dim`` (and so
    ``scale``) is untouched, so each surviving head computes exactly what it
    did before.
    """
    head_dim, old_heads = attn.head_dim, attn.num_heads
    n_keep = len(kept_heads)
    if n_keep == old_heads:
        return

    # Row/column indices of the surviving heads within one q/k/v segment.
    head_slice = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim)
                            for h in kept_heads])
    qkv_rows = torch.cat([head_slice + seg * old_heads * head_dim for seg in range(3)])
    new_dim = n_keep * head_dim
    kw = {'device': attn.qkv.weight.device, 'dtype': attn.qkv.weight.dtype}

    new_qkv = nn.Linear(attn.qkv.weight.shape[1], 3 * new_dim,
                        bias=attn.qkv.bias is not None, **kw)
    new_proj = nn.Linear(new_dim, attn.proj.weight.shape[0],
                         bias=attn.proj.bias is not None, **kw)
    with torch.no_grad():
        new_qkv.weight.copy_(attn.qkv.weight[qkv_rows])
        new_proj.weight.copy_(attn.proj.weight[:, head_slice])
        if attn.qkv.bias is not None:
            new_qkv.bias.copy_(attn.qkv.bias[qkv_rows])
        if attn.proj.bias is not None:
            new_proj.bias.copy_(attn.proj.bias)

    attn.qkv, attn.proj = new_qkv, new_proj
    attn.num_heads, attn.attn_dim = n_keep, new_dim


def prune_mlp_neurons(mlp: nn.Module, percentage: float) -> None:
    """Shrink a timm ``Mlp`` in place to ``percentage`` of its hidden neurons.

    There are no DARTS alphas for MLP neurons, so importance is the L2 norm of
    each neuron's outgoing weights in ``fc2`` -- the standard magnitude
    criterion for structured pruning. ``fc1`` keeps the surviving rows, ``fc2``
    the matching columns; both ends stay 768 wide, so the block output is
    unchanged in shape.
    """
    hidden = mlp.fc1.out_features
    n_keep = max(1, min(hidden, round(hidden * float(percentage) / 100.0)))
    if n_keep == hidden:
        return

    keep = torch.topk(mlp.fc2.weight.norm(dim=0), k=n_keep).indices.sort().values
    kw = {'device': mlp.fc1.weight.device, 'dtype': mlp.fc1.weight.dtype}
    new_fc1 = nn.Linear(mlp.fc1.in_features, n_keep, bias=mlp.fc1.bias is not None, **kw)
    new_fc2 = nn.Linear(n_keep, mlp.fc2.out_features, bias=mlp.fc2.bias is not None, **kw)
    with torch.no_grad():
        new_fc1.weight.copy_(mlp.fc1.weight[keep])
        new_fc2.weight.copy_(mlp.fc2.weight[:, keep])
        if mlp.fc1.bias is not None:
            new_fc1.bias.copy_(mlp.fc1.bias[keep])
        if mlp.fc2.bias is not None:
            new_fc2.bias.copy_(mlp.fc2.bias)

    mlp.fc1, mlp.fc2 = new_fc1, new_fc2


def build_pruned_vit(head_percentages: List[int], alphas: List[List[float]], num_classes: int,
                     mlp_percentages: List[int] = None,
                     model_name: str = 'vit_base_patch16_224',
                     pretrained: bool = True) -> nn.Module:
    """Build a ViT with block ``i`` keeping ``head_percentages[i]`` % of its heads and,
    if ``mlp_percentages`` is given, ``mlp_percentages[i]`` % of its MLP neurons.

    Only the MLPs and the classifier are trainable. The attention layers, patch
    embedding, positional embedding and LayerNorms keep their pretrained weights
    frozen, as in the DARTS phase.
    """
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    n_blocks = len(model.blocks)
    for name, genes in (('head', head_percentages), ('MLP', mlp_percentages)):
        if genes is not None and len(genes) != n_blocks:
            raise ValueError(
                f"Got {len(genes)} {name} genes but {model_name} has {n_blocks} "
                f"blocks. Set num_blocks to {n_blocks}.")
    if len(alphas) != n_blocks:
        raise ValueError(
            f"Got alphas for {len(alphas)} blocks but {model_name} has {n_blocks}. "
            f"Regenerate them with run_darts_alphas.py --model_name {model_name}.")

    for block, block_alphas, percent in zip(model.blocks, alphas, head_percentages):
        prune_attention_heads(block.attn, select_heads_by_alpha(block_alphas, percent))
    if mlp_percentages is not None:
        for block, percent in zip(model.blocks, mlp_percentages):
            prune_mlp_neurons(block.mlp, percent)

    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks:
        for param in block.mlp.parameters():
            param.requires_grad = True
    for param in model.get_classifier().parameters():
        param.requires_grad = True
    return model
