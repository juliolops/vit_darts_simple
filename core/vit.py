"""ViT pruning driven by DARTS importances for attention heads and MLP neurons.

``run_darts_alphas.py`` learns one alpha per attention head and one per MLP
hidden neuron and saves them; the genetic search then evolves genes holding
the *percentage* of heads (and, with MLP pruning, of MLP neurons) each block
keeps, and a gene of 40% keeps the 40% with the largest alpha. So DARTS
decides *which* heads and neurons matter and the GA decides *how much* to prune.

Pruning is surgical rather than masking: ``qkv``/``proj`` (and ``fc1``/``fc2``)
are rebuilt with only the survivors, so a pruned model really is cheaper.
"""
import json
import os
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

Alphas = Dict[str, List[List[float]]]


def extract_alphas(model) -> Alphas:
    """Per-block importances from a DARTS-wrapped ViT, as the ``N * softmax`` weights
    DARTS applied: ``{'heads': [[12 floats] x blocks], 'mlp': [[3072 floats] x blocks]}``."""
    from vit_transformer_search import normalized_weights

    def weights(alphas):
        return normalized_weights(alphas.detach().cpu().double()).tolist()

    return {'heads': [weights(b.attn.alphas) for b in model.blocks],
            'mlp': [weights(b.mlp.alphas) for b in model.blocks]}


def save_alphas(path: str, alphas: Alphas) -> None:
    """Write ``alphas`` to ``path`` as JSON, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(alphas, f)


def load_alphas(path: str) -> Alphas:
    """Load alphas, pointing at the DARTS step if they were never generated."""
    regenerate = f"Run the DARTS phase first:\n    python run_darts_alphas.py --output {path}"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DARTS alphas not found at '{path}'. {regenerate}")
    with open(path) as f:
        alphas = json.load(f)
    if not isinstance(alphas, dict) or set(alphas) != {'heads', 'mlp'}:
        raise ValueError(f"'{path}' has no MLP alphas (old format). {regenerate}")
    return alphas


def select_by_alpha(alphas: Sequence[float], percentage: float) -> List[int]:
    """Sorted indices of the ``percentage`` % units with the largest alpha. Never empty."""
    n_keep = max(1, min(len(alphas), round(len(alphas) * float(percentage) / 100.0)))
    values = torch.as_tensor(list(alphas), dtype=torch.float64)
    return sorted(int(i) for i in torch.topk(values, k=n_keep).indices.tolist())


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


def prune_mlp_neurons(mlp: nn.Module, kept_neurons: Sequence[int]) -> None:
    """Shrink a timm ``Mlp`` in place to ``kept_neurons`` hidden neurons.

    ``fc1`` keeps the surviving rows, ``fc2`` the matching columns; both ends
    stay 768 wide, so the block output is unchanged in shape.
    """
    if len(kept_neurons) == mlp.fc1.out_features:
        return

    keep = torch.as_tensor(list(kept_neurons), dtype=torch.long)
    kw = {'device': mlp.fc1.weight.device, 'dtype': mlp.fc1.weight.dtype}
    new_fc1 = nn.Linear(mlp.fc1.in_features, len(keep), bias=mlp.fc1.bias is not None, **kw)
    new_fc2 = nn.Linear(len(keep), mlp.fc2.out_features, bias=mlp.fc2.bias is not None, **kw)
    with torch.no_grad():
        new_fc1.weight.copy_(mlp.fc1.weight[keep])
        new_fc2.weight.copy_(mlp.fc2.weight[:, keep])
        if mlp.fc1.bias is not None:
            new_fc1.bias.copy_(mlp.fc1.bias[keep])
        if mlp.fc2.bias is not None:
            new_fc2.bias.copy_(mlp.fc2.bias)

    mlp.fc1, mlp.fc2 = new_fc1, new_fc2


def build_pruned_vit(head_percentages: List[int], alphas: Alphas, num_classes: int,
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
    if len(alphas['heads']) != n_blocks or len(alphas['mlp']) != n_blocks:
        raise ValueError(
            f"Got alphas for {len(alphas['heads'])} blocks but {model_name} has {n_blocks}. "
            f"Regenerate them with run_darts_alphas.py --model_name {model_name}.")

    for block, block_alphas, percent in zip(model.blocks, alphas['heads'], head_percentages):
        prune_attention_heads(block.attn, select_by_alpha(block_alphas, percent))
    if mlp_percentages is not None:
        for block, block_alphas, percent in zip(model.blocks, alphas['mlp'], mlp_percentages):
            prune_mlp_neurons(block.mlp, select_by_alpha(block_alphas, percent))

    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks:
        for param in block.mlp.parameters():
            param.requires_grad = True
    for param in model.get_classifier().parameters():
        param.requires_grad = True
    return model
