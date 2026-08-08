import csv
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from core.cnn import input as cnn_input
from core.fairness.models import make_baseline_model, REGISTRY

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config_robust(args):
    """
    Manually build the train_spec dictionary to avoid QNAS-specific dependencies
    that cause crashes when running simple baselines.
    """
    # 1. Default Defaults
    train_spec = {
        'batch_size': 128,
        'eval_batch_size': 128,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'max_epochs': 10,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_augmentation': True,
        'split_seed': 2025,
        'loader_seed': 777,
        'train_split': 0.9,
        'dataset': 'custom_baseline', # Placeholder, will be updated by loader
    }

    # 2. Try to load the config file
    config_path = Path(args.config_file)
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                loaded_yaml = yaml.safe_load(f)
                
            # Case A: It's a full QNAS experiment config (has 'train' key)
            if loaded_yaml and 'train' in loaded_yaml:
                print(f"Loaded experiment config from {config_path}")
                train_spec.update(loaded_yaml['train'])
                # If QNAS config refers to a dataset config, ensure we point to it
                if 'config_path_dataset' not in train_spec:
                    pass
            # Case B: It's just a Dataset Spec (no 'train' key)
            else:
                print(f"Loaded dataset spec from {config_path}")
                # We assume the file passed IS the dataset config
                train_spec['config_path_dataset'] = str(config_path)
                
        except Exception as e:
            print(f"Warning: Could not parse config file as YAML ({e}). Assuming it is a dataset path.")
            train_spec['config_path_dataset'] = str(config_path)
    else:
        raise FileNotFoundError(f"Config file not found: {args.config_file}")

    # 3. Apply CLI Overrides (Highest Priority)
    if args.data_path:
        train_spec['data_path'] = args.data_path
    if args.batch_size:
        train_spec['batch_size'] = args.batch_size
    if args.max_epochs:
        train_spec['max_epochs'] = args.max_epochs
    if args.learning_rate:
        train_spec['learning_rate'] = args.learning_rate
    if args.device:
        train_spec['device'] = args.device
    if args.num_workers:
        train_spec['num_workers'] = args.num_workers
    if args.dataset:
        train_spec['dataset'] = args.dataset
    if args.limit_data:
        train_spec['limit_data'] = True
    if args.limit_data_value:
        train_spec['limit_data_value'] = args.limit_data_value
        train_spec['limit_data'] = True
    if args.results_csv:
        train_spec['results_csv'] = args.results_csv
    
    # Store other flags
    train_spec['freeze_backbone'] = args.freeze_backbone
    train_spec['from_scratch'] = args.from_scratch  # <--- NEW: Store flag in spec
    train_spec['experiment_path'] = args.experiment_path

    return train_spec

def train_one_model(arch, train_loader, val_loader, device, params, results_csv=None):
    print(f"\n--- Starting Training for [{arch}] ---")
    
    num_classes = params.get('num_classes', 2)
    
    # --- NEW: Check if we are training from scratch ---
    from_scratch = params.get('from_scratch', False)
    use_pretrained = not from_scratch
    
    if from_scratch:
        print(f"[{arch}] INITIALIZATION: Random Weights (Training from Scratch)")
    else:
        print(f"[{arch}] INITIALIZATION: ImageNet Pre-trained Weights")

    # Pass the pretrained argument to the factory function
    model = make_baseline_model(arch, num_classes=num_classes, pretrained=use_pretrained).to(device)
    
    if params.get('freeze_backbone', False):
        print(f"[{arch}] Freezing backbone and training only the head.")
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    lr = float(params.get('learning_rate', 1e-3))
    wd = float(params.get('weight_decay', 1e-4))
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    output_dir = Path(params['experiment_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = params.get('dataset', 'dataset')
    
    # Modify filename if training from scratch to avoid overwriting
    scratch_suffix = "_scratch" if from_scratch else ""
    checkpoint_path = output_dir / f"{dataset_name}_{arch}{scratch_suffix}.pt"
    
    best_val_acc = 0.0
    max_epochs = int(params.get('max_epochs', 10))

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} [{arch} Train]")
        for batch in pbar:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            if len(labels.shape) > 1 and labels.shape[1] == 1:
                labels = labels.squeeze()
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch
                
                inputs, labels = inputs.to(device), labels.to(device)
                if len(labels.shape) > 1 and labels.shape[1] == 1:
                    labels = labels.squeeze()
                labels = labels.long()
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / max(1, val_total)
        avg_train_loss = train_loss / max(1, train_total)
        print(f"Epoch {epoch} [{arch}]: Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> New best val acc: {best_val_acc:.2f}%. Saving model to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

    print(f"--- Finished Training for [{arch}]. Best model saved to {checkpoint_path} ---")
    
    if results_csv:
        results_path = Path(results_csv)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not results_path.exists()
        
        with open(results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['arch', 'dataset', 'best_val_acc', 'checkpoint_path'])
            writer.writerow([arch, dataset_name, f"{best_val_acc:.4f}", str(checkpoint_path.resolve())])
        print(f"Saved best accuracy result to {results_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models using QNAS GenericDataLoader and Config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Arguments ---
    parser.add_argument('--config_file', required=True, help="Path to the YAML config file")
    parser.add_argument('--experiment_path', required=True, help="Root directory for logs")
    parser.add_argument('--data_path', required=True, help="Root directory containing the datasets")
    
    parser.add_argument('--archs', type=str, default="resnet18,resnet50,efficientnet_v2_s",
                        help="Comma-separated torchvision archs to train.")
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="If set, train only the final classification head.")
    
    # --- NEW: Argument to disable pretrained weights ---
    parser.add_argument('--from_scratch', action='store_true',
                        help="If set, initializes models with random weights instead of ImageNet.")
                        
    parser.add_argument('--results_csv', type=str, default="checkpoints/baselines/baseline_results.csv",
                        help="Path to CSV file to save results.")
    
    # --- Data Limiting ---
    parser.add_argument('--limit_data', action='store_true', help="If set, enables dataset limiting.")
    parser.add_argument('--limit_data_value', type=int, default=None, help="Number of images to use.")

    # --- Overrides ---
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)

    args = parser.parse_args()

    print("Loading configuration...")
    # Use Robust Loader instead of core.config.ConfigParameters
    train_spec = load_config_robust(args)
    
    device_str = train_spec.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    set_seed(train_spec.get('split_seed', 42))
    
    print(f"Device: {device}")
    print(f"Dataset config: {train_spec.get('config_path_dataset')}")
    if train_spec.get('limit_data'):
        print(f"Data Limiting Enabled: {train_spec.get('limit_data_value')} samples")

    print("Initializing GenericDataLoader...")
    data_loader = cnn_input.GenericDataLoader(params=train_spec)
    train_loader, val_loader = data_loader.get_loader(for_train=True, pin_memory_device=device_str)
    
    train_spec['num_classes'] = data_loader.spec.num_classes
    # Ensure dataset name is set correctly for filenames
    if train_spec['dataset'] == 'custom_baseline':
        train_spec['dataset'] = data_loader.spec.name

    requested_archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    available_archs = [arch for arch in requested_archs if arch in REGISTRY]
    
    if not available_archs:
        print("No requested architectures are available.")
        sys.exit(1)

    for arch in available_archs:
        train_one_model(
            arch=arch,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            params=train_spec,
            results_csv=args.results_csv
        )

if __name__ == "__main__":
    main()