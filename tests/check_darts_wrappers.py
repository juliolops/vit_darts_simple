"""Check that the DARTS wrappers compute exactly what timm computes.

With every alpha equal, ``N * softmax(alphas)`` is 1 for every head and neuron,
so the wrapped ViT must give the same outputs as the original timm ViT. The
check runs both attention paths (timm's fused ``scaled_dot_product_attention``
and the manual one), then confirms that the alphas receive gradients and that
changing them changes the output.

    python tests/check_darts_wrappers.py            # random weights, no download
    python tests/check_darts_wrappers.py --pretrained

Prints PASS/FAIL per check and exits with code 1 on any failure.
"""
import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.vit import create_vit  # noqa: E402
from vit_transformer_search import DartsAttentionWrapper, DartsMlpWrapper  # noqa: E402


def wrap(model):
    for block in model.blocks:
        block.attn = DartsAttentionWrapper(block.attn)
        block.mlp = DartsMlpWrapper(block.mlp)
    return model


def set_fused(model, fused):
    for block in model.blocks:
        if hasattr(block.attn, 'fused_attn'):
            block.attn.fused_attn = fused


def main(args):
    import timm
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"timm {timm.__version__} | torch {torch.__version__} | device {device}")

    original = create_vit(args.model, args.pretrained, args.num_classes).to(device).eval()
    # The alphas are created on the CPU, so move the wrapped model after wrapping
    # (build_darts_vit's callers do the same with model.to(device)).
    wrapped = wrap(copy.deepcopy(original)).to(device).eval()
    x = torch.randn(args.batch, 3, 224, 224, device=device)
    failures = 0

    def report(name, ok, detail):
        nonlocal failures
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1) Equal alphas -> same outputs as timm, on both attention paths.
    for fused in (True, False):
        set_fused(original, fused)
        set_fused(wrapped, fused)
        with torch.no_grad():
            diff = (original(x) - wrapped(x)).abs().max().item()
        report(f"igual ao timm ({'fused' if fused else 'manual'})", diff < args.atol,
               f"max |diff| = {diff:.2e} (tolerância {args.atol:.0e})")
    fused_default = getattr(create_vit(args.model, False, args.num_classes).blocks[0].attn,
                            'fused_attn', False)
    set_fused(wrapped, fused_default)
    print(f"       timm usa o caminho {'fused' if fused_default else 'manual'} por padrão nesta máquina")

    # 2) The alphas receive gradients.
    wrapped.train()
    wrapped.zero_grad()
    wrapped(x).sum().backward()
    head_grad = wrapped.blocks[0].attn.alphas.grad
    mlp_grad = wrapped.blocks[0].mlp.alphas.grad
    ok = head_grad is not None and mlp_grad is not None and \
        head_grad.abs().sum() > 0 and mlp_grad.abs().sum() > 0
    report("gradiente chega aos alphas", ok,
           f"cabeças {0 if head_grad is None else head_grad.abs().sum().item():.2e}, "
           f"neurônios {0 if mlp_grad is None else mlp_grad.abs().sum().item():.2e}")

    # 3) Changing the alphas changes the output.
    wrapped.eval()
    with torch.no_grad():
        before = wrapped(x)
        wrapped.blocks[0].attn.alphas[0] += 3.0
        wrapped.blocks[0].mlp.alphas[:100] -= 3.0
        change = (wrapped(x) - before).abs().max().item()
    report("alphas mudam a saída", change > 1e-4, f"max |mudança| = {change:.2e}")

    print("Resultado:", "tudo certo" if failures == 0 else f"{failures} verificação(ões) falharam")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='vit_base_patch16_224')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use the pretrained weights (downloads them if not cached).')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--atol', type=float, default=1e-4)
    main(parser.parse_args())
