"""ViT attention-head pruning driven by DARTS importances.

Two phases share this module. ``run_darts_alphas.py`` learns one alpha per
attention head and saves them; the genetic search then evolves one gene per
transformer block holding the *percentage* of heads that block keeps, and a
gene of 40% keeps the 40% of heads with the largest alpha. So DARTS decides
*which* heads matter and the GA decides *how much* to prune.

Pruning is surgical rather than masking: ``qkv``/``proj`` are rebuilt with
only the surviving heads, so a pruned model really is cheaper.
"""
import json
import os
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Alphas: written by the DARTS phase, read by the search
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------

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


def build_pruned_vit(net_list: List[str], fn_dict: Dict[str, dict],
                     alphas: List[List[float]], num_classes: int,
                     model_name: str = 'vit_base_patch16_224',
                     pretrained: bool = True) -> nn.Module:
    """Build a ViT head-pruned per the decoded chromosome.

    ``net_list`` is one gene per transformer block, each naming an entry of
    ``fn_dict`` that carries a ``percent``. Everything but the classifier is
    frozen, so a candidate's accuracy reflects the pruned representation
    rather than a full retraining of the backbone.
    """
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    n_blocks = len(model.blocks)
    if len(net_list) != n_blocks:
        raise ValueError(
            f"Chromosome has {len(net_list)} genes but {model_name} has {n_blocks} "
            f"blocks. Set QNAS.max_num_nodes to {n_blocks} in the config (or pass "
            f"--max_num_nodes {n_blocks}); extra genes would be silently ignored.")
    if len(alphas) != n_blocks:
        raise ValueError(
            f"Got alphas for {len(alphas)} blocks but {model_name} has {n_blocks}. "
            f"Regenerate them with run_darts_alphas.py --model_name {model_name}.")

    for i, (block, gene) in enumerate(zip(model.blocks, net_list)):
        kept = select_heads_by_alpha(alphas[i], fn_dict[gene]['params']['percent'])
        prune_attention_heads(block.attn, kept)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True
    return model
