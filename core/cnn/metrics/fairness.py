# moq-nas/core/cnn/metrics/fairness.py
import torch
import numpy as np
from typing import Dict, Tuple
from collections import defaultdict
from torch.amp import autocast
from tqdm import tqdm

from .base import BaseMetric
from core.fairness.data import create_eval_loader

class FairnessMetric(BaseMetric):
    name = "FairnessMetric"

    @property
    def is_post_processing(self) -> bool:
        return True

    def __init__(self, **kwargs):
        super().__init__()
        self._init_args = kwargs

        self.model = self._init_args.get('model')
        device_str = self._init_args.get('device')
        self.device = torch.device(device_str) if device_str else torch.device('cpu')
        self.eval_dataset_name = self._init_args.get('eval_dataset_name')
        self.eval_dataset_path = self._init_args.get('eval_dataset_path')
        self.optimization_objective = self._init_args.get('optimization_objective', '').lower()
        self.beta = self._init_args.get('beta')
        self.cache_dir = self._init_args.get('cache_dir')
        self.batch_size = self._init_args.get('batch_size_fairness', 64)
        self.positive_class_idx = self._init_args.get('positive_class_idx', 1)
        self.eval_skintone_method = self._init_args.get('eval_skintone_method', 'soft').lower()
        self.img_size = self._init_args.get('img_size', 224)
        self.square_mode = self._init_args.get('square_mode', 'letterbox')
        # Injected by EvalPopulation from the run's precision policy
        # ('fp32'|'fp16'|'bf16'); defaults to fp32 when constructed directly.
        self.precision = self._init_args.get('precision', 'fp32')

        if not all([self.model, self.device, self.eval_dataset_name, self.beta]):
            raise ValueError(f"FairnessMetric is missing required arguments. Provided: {list(self._init_args.keys())}")

        # (Optional) helps on Ampere+ including 3090/L40S
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass

        self._results = {}

    def _autocast_kwargs(self):
        """
        Autocast settings derived from the run's precision policy, so
        fairness evaluation always uses the same dtype as training.
        bf16 hardware support acts only as a guard, never as a selector.
        Disabled on CPU and for fp32.
        """
        if self.device.type != 'cuda':
            return dict(device_type='cpu', enabled=False)
        if self.precision == 'fp32':
            return dict(device_type='cuda', enabled=False)
        if self.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"precision='bf16' was requested for fairness evaluation but "
                f"CUDA device '{torch.cuda.get_device_name(self.device)}' has "
                f"no native bfloat16 support. Use precision='fp16' or 'fp32'.")
        dtype = torch.bfloat16 if self.precision == 'bf16' else torch.float16
        return dict(device_type='cuda', dtype=dtype, enabled=True)

    def reset(self):
        self._results = {}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor = None):
        pass

    def compute(self, epoch_results=None) -> Dict[str, float]:
        if self._results:
            return self._results

        dataloader = create_eval_loader(
            dataset_name=self.eval_dataset_name,
            csv_path=self.eval_dataset_path,
            batch_size=self.batch_size,
            img_size=self.img_size,
            cache_dir=self.cache_dir,
            square_mode=self.square_mode
        )

        # Variables to hold results
        tpr_ = {}
        counts_ = {}

        # 1. Determine TPR and Counts based on the dataset
        if self.eval_dataset_name.lower() == 'facet':
            if self.eval_skintone_method == 'soft':
                tpr_, counts_ = self._compute_tpr_per_skintone_soft(dataloader)
            elif self.eval_skintone_method == 'hard':
                tpr_, counts_ = self._compute_tpr_per_skintone_hard(dataloader)
            else:
                raise ValueError(f"Unknown eval_skintone_method: {self.eval_skintone_method}")
        
        elif self.eval_dataset_name.lower() == 'fairface':
            tpr_, counts_ = self._compute_tpr_per_group(dataloader)
            
        else:
            raise ValueError(f"Unknown evaluation dataset: {self.eval_dataset_name}")

        # 2. Common Result Processing (Written only once)
        summary_metrics = self._compute_summary_metrics(tpr_, counts_)
        
        self._results["per_group_tpr"] = tpr_
        self._results["metrics"] = summary_metrics       # Nested (for your file output)
        self._results.update(summary_metrics)            # Flattened (for the optimizer)
        self._results["fairness_score"] = summary_metrics.get(self.optimization_objective, 0.0)

        del dataloader
        return self._results

    def _compute_tpr_per_skintone_hard(self, loader) -> Tuple[Dict[str, float], Dict[str, int]]:
        group_correct = defaultdict(int)
        group_total = defaultdict(int)

        with torch.inference_mode():
            for inputs, soft_labels in tqdm(loader, desc="Computing TPR (skintone, hard)"):
                inputs = inputs.to(self.device, non_blocking=True)
                with autocast(**self._autocast_kwargs()):
                    logits = self.model(inputs)
                logits = logits.detach().cpu() 
                preds = logits.argmax(dim=1)

                hard_labels = soft_labels.argmax(dim=1)  # on CPU already

                for i in range(len(preds)):
                    group_idx = int(hard_labels[i].item()) + 1  # 1..10
                    group_total[str(group_idx)] += 1
                    if int(preds[i].item()) == self.positive_class_idx:
                        group_correct[str(group_idx)] += 1
        
        per_tone_tpr = {
            tone: float(group_correct[tone] / group_total[tone]) if group_total[tone] > 0 else 0.0
            for tone in sorted(group_total.keys())
        }
        # Return TPR and Counts
        return per_tone_tpr, dict(group_total)

    def _compute_tpr_per_skintone_soft(self, loader) -> Tuple[Dict[str, float], Dict[str, float]]:
        denominator = defaultdict(float)
        numerator = defaultdict(float)

        with torch.inference_mode():
            for inputs, soft_labels in tqdm(loader, desc="Computing TPR (skintone, soft)"):
                inputs = inputs.to(self.device, non_blocking=True)
                with autocast(**self._autocast_kwargs()):
                    logits = self.model(inputs)
                logits = logits.detach().cpu()
                preds = logits.argmax(dim=1)

                # soft_labels is CPU; iterate without moving to GPU
                B, T = soft_labels.shape
                for i in range(B):
                    pred_i = int(preds[i].item())
                    row = soft_labels[i]
                    # accumulate weighted contributions
                    for tone_idx in range(T):
                        prob = float(row[tone_idx].item())
                        if prob > 0.0:
                            key = str(tone_idx + 1)
                            denominator[key] += prob
                            if pred_i == self.positive_class_idx:
                                numerator[key] += prob

        per_tone_tpr = {
            tone: float(numerator[tone] / denominator[tone]) if denominator[tone] > 0 else 0.0
            for tone in sorted(denominator.keys())
        }
        # Return TPR and Denominators (Weighted Counts)
        return per_tone_tpr, dict(denominator)

    def _compute_tpr_per_group(self, loader) -> Tuple[Dict[str, float], Dict[str, int]]:
        group_tpr = defaultdict(float)
        group_counts = defaultdict(int)
        label_map = {v: k for k, v in loader.dataset.race_to_idx.items()}

        with torch.inference_mode():
            for inputs, labels in tqdm(loader, desc="Computing TPR (per group)"):
                inputs = inputs.to(self.device, non_blocking=True)
                with autocast(**self._autocast_kwargs()):
                    logits = self.model(inputs)
                logits = logits.detach().cpu()
                preds = logits.argmax(dim=1)

                for i in range(len(labels)):
                    label_idx = int(labels[i].item())
                    group_name = label_map[label_idx]
                    if int(preds[i].item()) == self.positive_class_idx:
                        group_tpr[group_name] += 1.0
                    group_counts[group_name] += 1
        
        final_tpr = {}
        for group_name, total in group_counts.items():
            if total > 0:
                final_tpr[group_name] = float(group_tpr[group_name] / total)
        
        # Return TPR and Counts
        return dict(sorted(final_tpr.items())), dict(group_counts)

    def _compute_summary_metrics(self, per_group_tpr: Dict[str, float], group_counts: Dict[str, float]) -> Dict[str, float]:
        if not per_group_tpr or not group_counts:
            return {"min_group_tpr": 0.0, "max_min_gap": 0.0, "spd_sum": 0.0, "fairness_raw": 0.0}

        # 1. Identify the Minority Group (Smallest sample size N)
        # We look at group_counts to find the key with the minimum value
        minority_group_name = min(group_counts, key=group_counts.get)
        
        # 2. Get Acc_mino (The accuracy/TPR of that minority group)
        acc_mino = per_group_tpr.get(minority_group_name, 0.0)

        tprs = np.array(list(per_group_tpr.values()), dtype=np.float32)
        mean_tpr = float(np.mean(tprs))
        
        # 3. Calculate SPD Sum according to Equation (2)
        # Sum(|Acc_i - Acc_mino|)
        # Note: We must use absolute value because acc_mino is not necessarily the minimum TPR
        spd_sum = float(np.sum(np.abs(tprs - acc_mino)))
        
        fairness_score = max(0.0, (self.beta - spd_sum) / self.beta)

        return {
            "mean_tpr": mean_tpr,
            "spd_sum": spd_sum,
            "fairness_raw": fairness_score,
            "minority_group_acc": acc_mino # Useful for debugging
        }