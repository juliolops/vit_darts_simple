import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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
    
    # Variáveis para rastrear acertos e total de amostras
    trn_correct = 0
    trn_total = 0
    val_correct = 0
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
        
        # Cálculo da acurácia de Validação no batch atual
        _, val_preds = torch.max(val_logits, 1)
        val_total += val_y.size(0)
        val_correct += (val_preds == val_y).sum().item()

        # ---------------------------------------------------------
        # ETAPA 2: Atualizar Pesos (MLPs) com dados de Treino
        # ---------------------------------------------------------
        optimizer_w.zero_grad()
        trn_logits = model(trn_X)
        loss_w = criterion(trn_logits, trn_y)
        loss_w.backward()
        optimizer_w.step()
        
        # Cálculo da acurácia de Treino no batch atual
        _, trn_preds = torch.max(trn_logits, 1)
        trn_total += trn_y.size(0)
        trn_correct += (trn_preds == trn_y).sum().item()
        
        # Exibição a cada 50 steps
        if step % 50 == 0:
            trn_acc = 100. * trn_correct / trn_total
            val_acc = 100. * val_correct / val_total
            print(f"Step {step:03d} | "
                  f"Treino -> Loss: {loss_w.item():.4f} Acc: {trn_acc:.2f}% | "
                  f"Validação -> Loss: {loss_alpha.item():.4f} Acc: {val_acc:.2f}%")

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Aceleração Apple Metal (MPS) ativada com sucesso!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Aceleração NVIDIA (CUDA) ativada!")
    else:
        device = torch.device("cpu")
        print("Aviso: Rodando na CPU.")

    print(f"Iniciando treinamento no dispositivo: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    train_size = len(full_train_dataset) // 2
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Correção da compatibilidade do pin_memory
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=use_pin_memory)

    model, weight_params, alpha_params = build_darts_vit(model_name='vit_base_patch16_224', num_classes=10)
    model = model.to(device)

    optimizer_w = torch.optim.AdamW(weight_params, lr=1e-3, weight_decay=1e-4)
    optimizer_alpha = torch.optim.Adam(alpha_params, lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    
    for epoch in range(num_epochs):
        print(f"\n--- Época {epoch+1}/{num_epochs} ---")
        
        train_darts_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_w=optimizer_w,
            optimizer_alpha=optimizer_alpha,
            criterion=criterion,
            device=device
        )
        
        analisar_alphas(model)

    torch.save(model.state_dict(), 'darts_vit_finetuned.pth')
    print("Treinamento concluído e modelo salvo!")

def analisar_alphas(model):
    print("\n[Distribuição dos Alphas por Bloco]")
    for i, block in enumerate(model.blocks):
        probs = F.softmax(block.attn.alphas.detach(), dim=0)
        probs_str = ", ".join([f"{p:.3f}" for p in probs.cpu().numpy()])
        print(f"Bloco {i:02d}: [{probs_str}]")

if __name__ == '__main__':
    main()
