import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vit import create_vit


def normalized_weights(alphas):
    """N * softmax(alphas): mean 1, so all-equal alphas leave the layer unchanged."""
    return alphas.numel() * F.softmax(alphas, dim=0)


def _self_attn_bias(seq_len, scores, attn_mask=None, is_causal=False):
    """Additive attention bias with ``F.scaled_dot_product_attention`` semantics
    (timm's ``resolve_self_attn_mask``), for the non-fused path."""
    if is_causal:
        return scores.new_full((seq_len, seq_len), float('-inf')).triu_(1)
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        bias = torch.zeros_like(attn_mask, dtype=scores.dtype)
        return bias.masked_fill_(~attn_mask, float('-inf'))
    return attn_mask


class DartsAttentionWrapper(nn.Module):
    """timm ``Attention`` with one DARTS alpha per head.

    Reuses the original module's layers (same pretrained weights, no copy) and
    follows timm's ``Attention.forward`` step by step: fused
    ``scaled_dot_product_attention`` when timm enables it, q/k norms, output
    norm and gate. The only addition is the head weighting
    ``N * softmax(alphas)`` on each head's output, before the heads are merged
    and projected. With all alphas equal it computes exactly what timm does.
    Attributes missing in older timm versions fall back to their no-op default.
    """

    def __init__(self, original_attn):
        super().__init__()
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.attn_dim = getattr(original_attn, 'attn_dim', self.num_heads * self.head_dim)
        self.scale = original_attn.scale
        self.fused_attn = getattr(original_attn, 'fused_attn', False)

        self.qkv = original_attn.qkv
        self.q_norm = getattr(original_attn, 'q_norm', nn.Identity())
        self.k_norm = getattr(original_attn, 'k_norm', nn.Identity())
        self.attn_drop = original_attn.attn_drop
        self.norm = getattr(original_attn, 'norm', nn.Identity())
        self.gate = getattr(original_attn, 'gate', None)
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop

        # Architecture parameters (alphas) weighting the heads.
        self.alphas = nn.Parameter(torch.ones(self.num_heads))

    def forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        gate = self.gate(x).sigmoid() if self.gate is not None else None
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
                is_causal=is_causal,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_bias = _self_attn_bias(N, attn, attn_mask, is_causal)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # DARTS: weight each head's output, shape (B, heads, N, head_dim).
        x = x * normalized_weights(self.alphas).view(1, -1, 1, 1)

        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        if gate is not None:
            x = x * gate
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DartsMlpWrapper(nn.Module):
    """timm ``Mlp`` with one DARTS alpha per hidden neuron.

    Reuses the original module's layers and follows timm's ``Mlp.forward``
    (fc1 -> act -> drop1 -> norm -> fc2 -> drop2). The only addition is the
    neuron weighting ``N * softmax(alphas)`` on the hidden activations, right
    before ``fc2``. With all alphas equal it computes exactly what timm does.
    """

    def __init__(self, original_mlp):
        super().__init__()
        self.fc1 = original_mlp.fc1
        self.act = original_mlp.act
        self.drop1 = original_mlp.drop1
        self.norm = getattr(original_mlp, 'norm', nn.Identity())
        self.fc2 = original_mlp.fc2
        self.drop2 = original_mlp.drop2

        # Architecture parameters (alphas) weighting the hidden neurons.
        self.alphas = nn.Parameter(torch.ones(self.fc1.out_features))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)

        # DARTS: weight each hidden neuron.
        x = x * normalized_weights(self.alphas)

        x = self.fc2(x)
        x = self.drop2(x)
        return x


def build_darts_vit(model_name='vit_base_patch16_224', num_classes=1000, head_classes=None):
    """Pretrained ViT with DARTS alphas on every attention head and MLP hidden neuron.

    Only the MLP weights and the classifier head are trained (``weight_params``);
    the attention layers, embeddings and LayerNorms keep their pretrained weights.
    ``head_classes`` keeps those outputs of the pretrained ImageNet classifier
    (ImageNet runs); otherwise the classifier is new (CIFAR-10).
    """
    model = create_vit(model_name, True, num_classes, head_classes)
    for param in model.parameters():
        param.requires_grad = False

    alpha_params = []
    weight_params = []

    for block in model.blocks:
        block.attn = DartsAttentionWrapper(block.attn)
        block.mlp = DartsMlpWrapper(block.mlp)
        alpha_params += [block.attn.alphas, block.mlp.alphas]

        for layer in (block.mlp.fc1, block.mlp.fc2):
            for param in layer.parameters():
                param.requires_grad = True
                weight_params.append(param)
            
    for param in model.head.parameters():
        param.requires_grad = True
        weight_params.append(param)
        
    return model, weight_params, alpha_params

def _clip(optimizer, max_norm):
    """Clip the gradient norm of every parameter handled by ``optimizer`` (no-op if max_norm <= 0)."""
    if max_norm and max_norm > 0:
        params = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, max_norm)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Mean loss and top-1 accuracy (%) of ``model`` over the whole ``loader``."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_darts_epoch(model, train_loader, val_loader, optimizer_w, optimizer_alpha, criterion, device,
                      update_alphas=True, grad_clip=5.0):
    """One DARTS epoch. With ``update_alphas=False`` (warm-up) only the MLP + classifier
    weights are trained and the alphas stay fixed. ``grad_clip`` caps the gradient norm
    of both the weights and the alphas, so a single bad batch cannot blow up the model."""
    model.train()
    
    val_iter = iter(val_loader)

    # Captura o total de steps (batches) na época atual
    total_steps = len(train_loader)
    
    # Variáveis para rastrear acertos Top-1 e Top-5
    trn_correct_top1 = 0
    trn_correct_top5 = 0
    trn_total = 0
    
    val_correct_top1 = 0
    val_correct_top5 = 0
    val_total = 0
    
    for step, (trn_X, trn_y) in enumerate(train_loader):
        trn_X, trn_y = trn_X.to(device), trn_y.to(device)
        
        try:
            val_X, val_y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            val_X, val_y = next(val_iter)
            
        val_X, val_y = val_X.to(device), val_y.to(device)

        # ---------------------------------------------------------
        # ETAPA 1: Atualizar Arquitetura (Alphas) com dados de Validação
        # ---------------------------------------------------------
        optimizer_alpha.zero_grad()
        if update_alphas:
            val_logits = model(val_X)
            loss_alpha = criterion(val_logits, val_y)
            loss_alpha.backward()
            _clip(optimizer_alpha, grad_clip)
            optimizer_alpha.step()
        else:
            # Warm-up: alphas frozen, the validation batch is only monitored.
            with torch.no_grad():
                val_logits = model(val_X)
                loss_alpha = criterion(val_logits, val_y)
        
        # Acurácia Top-1 Validação
        _, val_preds = torch.max(val_logits, 1)
        val_total += val_y.size(0)
        val_correct_top1 += (val_preds == val_y).sum().item()
        
        # Acurácia Top-5 Validação
        _, val_top5 = val_logits.topk(5, dim=1)
        val_correct_top5 += (val_top5 == val_y.view(-1, 1)).sum().item()

        # ---------------------------------------------------------
        # ETAPA 2: Atualizar Pesos (MLPs) com dados de Treino
        # ---------------------------------------------------------
        optimizer_w.zero_grad()
        trn_logits = model(trn_X)
        loss_w = criterion(trn_logits, trn_y)
        loss_w.backward()
        _clip(optimizer_w, grad_clip)
        optimizer_w.step()
        optimizer_alpha.zero_grad()  # drop the alpha gradients left by loss_w.backward()
        
        # Acurácia Top-1 Treino
        _, trn_preds = torch.max(trn_logits, 1)
        trn_total += trn_y.size(0)
        trn_correct_top1 += (trn_preds == trn_y).sum().item()
        
        # Acurácia Top-5 Treino
        _, trn_top5 = trn_logits.topk(5, dim=1)
        trn_correct_top5 += (trn_top5 == trn_y.view(-1, 1)).sum().item()
        
        # Exibição a cada 50 steps
        if step % 50 == 0:
            trn_acc1 = 100. * trn_correct_top1 / trn_total
            trn_acc5 = 100. * trn_correct_top5 / trn_total
            val_acc1 = 100. * val_correct_top1 / val_total
            val_acc5 = 100. * val_correct_top5 / val_total
            
            print(f"Step {step:03d}/{total_steps} | "
                  f"Treino -> Loss: {loss_w.item():.4f} Acc@1: {trn_acc1:.2f}% Acc@5: {trn_acc5:.2f}% | "
                  f"Validação -> Loss: {loss_alpha.item():.4f} Acc@1: {val_acc1:.2f}% Acc@5: {val_acc5:.2f}%")
