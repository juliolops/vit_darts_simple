# evaluation_fairness_workers.py
from typing import Dict, Any, List, Tuple
import os, torch
import traceback
from utils.helpers import update_yaml_file

def fairness_worker_cuda(
    shard: List[Tuple[int, dict]],
    parallel_train_params: dict,
    fn_dict: dict,
    fairness_metric_names: List[str],
    fairness_params: dict | None,
    device_idx: int | None,
) -> Dict[int, Dict[str, float]]:
    from core.cnn import master
    from core.cnn.metrics.fairness import FairnessMetric

    # Thread caps to avoid BLAS oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

    # Select device inside the spawned child (safe)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        if device_idx is None:
            device_idx = 0
        device_idx = device_idx % torch.cuda.device_count()
        device = f"cuda:{device_idx}"
        torch.cuda.set_device(device_idx)
    else:
        device = "cpu"

    results: Dict[int, Dict[str, float]] = {}
    fairness_params = fairness_params or {}

    for i, spec in shard:
        # Default fallback for this candidate
        fallback_result = {name: 0.0 for name in fairness_metric_names}

        if not spec:
            results[i] = fallback_result
            continue

        net = None
        state = None
        evaluator = None
        try:
            params = {
                **parallel_train_params,
                "net_list": spec["decoded_net"],
                "fn_dict": fn_dict,
                **spec["decoded_params"],
                "device": device,
            }

            # Build & load on the target device
            net = master.create_model(params).to(device).eval()
            state = torch.load(spec["model_path"], map_location=device, weights_only=True)
            net.load_state_dict(state)

            evaluator = FairnessMetric(model=net, device=device, **fairness_params)
            
            # Compute returns a dict with keys like 'spd_sum', 'mean_tpr', 'fairness_raw'
            metrics_fairness = evaluator.compute()
            
            # Return the WHOLE dictionary so evaluation.py can pick what it wants
            results[i] = metrics_fairness
            
            patch = {
                "per_group_tpr": metrics_fairness.get("per_group_tpr", {}),
                "metrics_fairness": metrics_fairness.get("metrics", {}),
                "fairness_score": metrics_fairness.get("fairness_score", 0.0),
            }

            info_dir = spec.get("info_dir") or os.path.dirname(spec["model_path"])
            file_path = os.path.join(info_dir, "training_params.txt")   
            update_yaml_file(file_path, patch)

        except Exception as e:
            print(f"[fairness_worker_cuda] Model {i} failed: {e}")
            traceback.print_exc()
            results[i] = fallback_result

        finally:
            try:
                mpth = spec.get("model_path")
                if mpth and os.path.exists(mpth):
                    os.remove(mpth)
            except Exception:
                pass
            
            # Cleanup
            try: del evaluator
            except: pass
            try: del state
            except: pass
            try: del net
            except: pass
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    return results


def device_count_probe() -> int:
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0

def device_count_probe_runner(q):
    q.put(device_count_probe())

def fairness_queue_runner(q, shard, parallel_train_params, fn_dict, fairness_metric_names, fairness_params, device_idx):
    """Top-level picklable runner that executes the worker and puts results on a Queue."""
    res = fairness_worker_cuda(
        shard=shard,
        parallel_train_params=parallel_train_params,
        fn_dict=fn_dict,
        fairness_metric_names=fairness_metric_names, # Passed as list
        fairness_params=fairness_params,
        device_idx=device_idx,
    )
    q.put(res)