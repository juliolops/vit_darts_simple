import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DartsAttentionWrapper(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.qkv = original_attn.qkv
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop
        self.attn_drop = original_attn.attn_drop
        
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.scale = original_attn.scale

        # Parâmetros da arquitetura (Alphas) para ponderar as cabeças
        self.alphas = nn.Parameter(torch.ones(self.num_heads))

    def forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attn_mask, float('-inf'))
            else:
                attn = attn + attn_mask
                
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_heads = attn @ v 

        # DARTS: Ponderação das cabeças
        alpha_weights = F.softmax(self.alphas, dim=0).view(1, -1, 1, 1)
        x_heads = x_heads * alpha_weights

        x = x_heads.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def build_darts_vit(model_name='vit_base_patch16_224', num_classes=1000):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    alpha_params = []
    weight_params = []

    for block in model.blocks:
        block.attn = DartsAttentionWrapper(block.attn)
        
        for param in block.attn.qkv.parameters():
            param.requires_grad = False
        for param in block.attn.proj.parameters():
            param.requires_grad = False
            
        alpha_params.append(block.attn.alphas)
        
        for param in block.mlp.parameters():
            param.requires_grad = True
            weight_params.append(param)
            
    for param in model.head.parameters():
        param.requires_grad = True
        weight_params.append(param)
        
    return model, weight_params, alpha_params

def train_darts_epoch(model, train_loader, val_loader, optimizer_w, optimizer_alpha, criterion, device):
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
        val_logits = model(val_X)
        loss_alpha = criterion(val_logits, val_y)
        loss_alpha.backward()
        optimizer_alpha.step()
        
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
        optimizer_w.step()
        
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
