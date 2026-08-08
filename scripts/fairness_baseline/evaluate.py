# moq-nas/scripts/fairness_baseline/evaluate.py
import sys
import argparse
import torch
from pathlib import Path
import json
from types import SimpleNamespace 

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from core.cnn.metrics.fairness import FairnessMetric
from core.fairness.models import make_baseline_model

def print_detailed_results(arch, data):
    """
    Prints a formatted summary of model performance.
    """
    print(f"\n{'='*15} Results for [{arch}] {'='*15}")

    # --- Helper function to print a specific set of results ---
    def display_metrics(result_type, results):
        print(f"\n--- {result_type} Results ---")
        if 'per_group_tpr' in results:
            for group, tpr in results['per_group_tpr'].items():
                print(f"  - Group: {group:<20} | TPR: {tpr:.4f}")
        
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\n  Summary Metrics:")
            min_tpr = metrics.get('min_group_tpr', 'N/A')
            max_gap = metrics.get('max_min_gap', 'N/A')
            spd_sum = metrics.get('spd_sum', 'N/A')
            fairness = metrics.get('fairness_raw', 'N/A')
            
            # Helper to safely format floats or print string N/A
            fmt = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
            
            print(f"  - Min Group TPR : {fmt(min_tpr)}")
            print(f"  - Max-Min Gap   : {fmt(max_gap)}")
            print(f"  - SPD Sum       : {fmt(spd_sum)}")
            print(f"  - Fairness Score: {fmt(fairness)}")

    # Case 1: Nested structure ('soft'/'hard' results)
    if 'soft_results' in data or 'hard_results' in data:
        if 'soft_results' in data:
            display_metrics("Soft", data['soft_results'])
        if 'hard_results' in data:
            display_metrics("Hard", data['hard_results'])
    # Case 2: Flat structure
    elif 'per_group_tpr' in data:
        display_metrics("Overall", data)
    else:
        print("  No recognizable result structure found.")
    
    print(f"{'='*50}\n")


def evaluate_one_model(arch: str, ckpt_path: str, device: torch.device, args: argparse.Namespace):
    """
    Evaluates a single model checkpoint using the centralized FairnessMetric class.
    """
    print(f"\n--- Evaluating [{arch}] on [{args.dataset_name}] ---")
    print(f"Loading checkpoint: {ckpt_path}")

    # --- Step 1: Build the Model and Load its Weights ---
    # We assume binary classification for fairness baselines (2 classes)
    model = make_baseline_model(arch, num_classes=2)
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[Warning] Error loading state dict directly: {e}")
        return {}

    model.to(device)
    model.eval()
    
    # --- Step 2: Instantiate and Run the Fairness Metric ---
    fairness_metric = FairnessMetric(
        model=model,
        device=device,
        eval_dataset_name=args.dataset_name,
        eval_dataset_path=args.csv_path,
        optimization_objective="spd_sum",
        beta=args.beta,                # <--- CRITICAL: Passed correctly
        square_mode=args.resize_mode,  # <--- CRITICAL: Matches training (letterbox vs center_crop)
        img_size=args.img_size,        # <--- CRITICAL: Matches training size (e.g. 96)
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        positive_class_idx=1,
        eval_skintone_method='soft',
    )
    
    results = fairness_metric.compute()
    
    # --- Step 3: Print the Key Results ---
    print_detailed_results(arch, results)
        
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models for fairness using the core FairnessMetric.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Directory containing model checkpoints.")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['fairface', 'facet'], help="Name of the evaluation dataset.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the evaluation CSV file.")
    parser.add_argument('--filter', type=str, default=None, help="Optional: Only evaluate checkpoints containing this string.")
    
    # Settings
    parser.add_argument('--beta', type=float, default=0.2, help="Beta value for the fairness score calculation.")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None, help="e.g., 'cuda:0'. Default picks best available.")
    parser.add_argument('--cache_dir', type=str, default=None, help="Optional directory to cache cropped images for FACET.")

    # Image Transform Settings (Must match training)
    parser.add_argument('--img_size', type=int, default=224, help="Input size (e.g. 96, 224)")
    parser.add_argument('--resize_mode', type=str, default='letterbox', choices=['letterbox', 'center_crop'], 
                        help="Resize strategy used during training.")

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"[Error] Checkpoint directory not found: {ckpt_dir}")
        return

    all_results = {}
    checkpoints = sorted(list(ckpt_dir.glob("*.pt")))
    
    if not checkpoints:
        print(f"[Warning] No .pt files found in {ckpt_dir}")

    for ckpt_path in checkpoints:
        if args.filter and args.filter not in ckpt_path.name:
            continue

        try:
            fname = ckpt_path.stem
            # Robust architecture parsing: 
            # 1. Try splitting by '_data_' (old style)
            if '_data_' in fname:
                arch = fname.split('_data_')[1]
            # 2. Try parsing known architectures if split fails
            else:
                known_archs = ["resnet18", "resnet50", "efficientnet_v2_s", "mobilenet_v3_large", "convnext_tiny", "mnasnet1_0"]
                arch = "resnet18" # fallback
                for k in known_archs:
                    if k in fname:
                        arch = k
                        break
            
            results = evaluate_one_model(arch, str(ckpt_path), device, args)
            all_results[str(ckpt_path.name)] = results
            
        except Exception as e:
            print(f"\n[ERROR] Could not evaluate {ckpt_path.name}. Reason: {e}\n")
            import traceback
            traceback.print_exc()

    output_filename = f"fairness_results_{args.dataset_name}"
    if args.filter:
        output_filename += f"_{args.filter}"
    output_path = ckpt_dir / f"{output_filename}.json"
    
    if all_results:
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\n✅ Saved all results to {output_path}")
    else:
        print("\n[Warning] No results were generated.")

if __name__ == "__main__":
    main()