import argparse
import os
import pickle
import yaml
import multiprocessing as mp
import torch
import traceback
from concurrent.futures import ProcessPoolExecutor

from core.cnn import input, master
from utils.helpers import load_log_params_evolution, init_log, save_results_file


def parse_pareto_ids(exp_path: str, top_n: int | None = None, sort_by: str | None = None):
    """
    Load candidate IDs from the final Pareto front that exist on disk.
    Reads IDs from pareto_history.pkl, then filters them to ensure a
    corresponding directory exists in the 'archive' folder.
    """
    history_file = os.path.join(exp_path, "pareto_history.pkl")
    archive_dir = os.path.join(exp_path, "archive")
    ids = []

    try:
        if os.path.isfile(history_file):
            with open(history_file, "rb") as f:
                history = pickle.load(f)

            if history:
                last_gen = max(history.keys())
                front = history[last_gen].get(1, [])

                if sort_by:
                    is_reversed = sort_by not in ['params', 'inference_time']
                    front = sorted(front, key=lambda x: x.get(sort_by, 0), reverse=is_reversed)

                if os.path.isdir(archive_dir):
                    existing_dirs = set(os.listdir(archive_dir))
                else:
                    existing_dirs = set()

                all_front_ids = [rec.get("id") for rec in front if rec.get("id")]
                ids = [cid for cid in all_front_ids if cid in existing_dirs]

    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Warning: Could not load {history_file}. It may be corrupted. Error: {e}")

    if not ids and os.path.isdir(archive_dir):
        print("Warning: No valid IDs found in pareto_history.pkl. "
                "Falling back to alphabetical list of models in archive directory.")
        ids = sorted(d for d in os.listdir(archive_dir)
                    if os.path.isdir(os.path.join(archive_dir, d)))

    if top_n:
        ids = ids[:top_n]

    return ids


def load_candidate_params(archive_dir: str, cid: str, logger):
    """
    Safely loads network parameters, handling all potential errors.
    Returns (net_list, backbone_name, backbone_percentage, evolution_metrics).
    evolution_metrics contains objective values measured during evolution
    (e.g. total_flops, best_accuracy, fairness_spd) so they can be attached
    to the retrain results without recomputing them.
    """
    # Objective and hardware metrics measured during evolution that are worth
    # keeping in the retrain summary. Config params (batch_size, seed, etc.) excluded.
    _METRIC_KEYS = {
        'best_accuracy', 'total_flops', 'total_params', 'cuda_inference_time',
        'model_memory_usage', 'training_time', 'best_validation_loss',
        'fairness_spd', 'fairness_mean_tpr', 'fairness_score',
    }
    _SKIP = {'net_list', 'fn_dict', 'backbone_name', 'backbone_percentage',
             'generation', 'individual'}
    params_file = os.path.join(archive_dir, cid, "training_params.txt")
    try:
        with open(params_file, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.error(f"Content of {params_file} is not a valid dictionary for candidate {cid}. Skipping.")
            return None, None, None, {}
        net_list = data.get("net_list", [])
        backbone = data.get("backbone_name")
        backbone_pct = data.get("backbone_percentage", 0.0)
        evolution_metrics = {k: v for k, v in data.items()
                             if k in _METRIC_KEYS and v is not None}
        return net_list, backbone, backbone_pct, evolution_metrics
    except Exception as e:
        logger.error(f"Failed to load or parse {params_file} for candidate {cid}. Error: {e}. Skipping.")
        return None, None, None, {}


def worker(task_args):
    """
    A self-contained worker that takes a single tuple of arguments and returns a result.
    It no longer uses a queue. Any exception will be caught by the ProcessPoolExecutor.
    """
    # 1. Unpack all arguments
    cid, base_spec, fn_dict, args, device, log_file = task_args
    
    # 2. Each worker initializes its own logger, writing to the central log file
    logger = init_log(args.log_level, name=f"worker-{cid}", file_path=log_file)
    
    try:
        archive_dir = os.path.join(args.experiment_path, "archive")
        net_list, backbone, backbone_pct, evolution_metrics = load_candidate_params(archive_dir, cid, logger)

        if net_list is None:
            # If params fail to load, return an error status
            return cid, {"error": "Failed to load parameters"}

        params = dict(base_spec)
        params.update({
            "data_path": args.data_path,
            "dataset": args.dataset,
            "device": device,
            "phase": "retrain",
        })
        if backbone: params["backbone_name"] = backbone
        if backbone_pct: params["backbone_percentage"] = backbone_pct

        override_keys = [
            "max_epochs", "epochs_to_eval", "batch_size", "eval_batch_size",
            "limit_data", "lr_scheduler", "optimizer", "data_augmentation",
            "num_workers", "save_checkpoints_epochs", "patience_retrain", "delta_fraction"
        ]
        for k in override_keys:
            v = getattr(args, k, None)
            if v is not None: params[k] = v

        if args.network_config:
            params['network_config'] = args.network_config
        if not args.keep_metrics:
            params['metrics'] = [{'name': 'Accuracy'}]
            params.pop('artifacts', None)

        logger.info(f"Creating DataLoader for {cid} on {device}")
        loader = input.GenericDataLoader(params=params)
        train_loader, val_loader = loader.get_loader(pin_memory_device=device)
        test_loader = loader.get_loader(for_train=False, pin_memory_device=device)
        logger.info(f"Train loader: {len(train_loader.dataset)} samples, "
        f"Validation loader: {len(val_loader.dataset)} samples, "
        f"Test loader: {len(test_loader.dataset)} samples")
        results = {}
        for rep in range(args.num_repetitions):
            model_save_path = os.path.join(archive_dir, cid, f"retrain_parallel_{rep+1}")
            params["experiment_path"] = model_save_path
            
            logger.info(f"Starting retraining for {cid} repetition {rep+1} on {device}")
            
            res = master.retrain(params=params, fn_dict=fn_dict, net_list=net_list,
                                train_loader=train_loader, val_loader=val_loader,
                                test_loader=test_loader)
            if isinstance(res, dict):
                res['evolution_metrics'] = evolution_metrics
            results[f"retrain_{rep+1}"] = res
        
        # 3. Simply return the result tuple.
        return cid, results

    except Exception as e:
        # Catch any unexpected error and return it so the main process can log it.
        error_message = f"Worker {cid} crashed unexpectedly.\n{traceback.format_exc()}"
        return cid, {"error": error_message}


def main(arguments):
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, 'retrain_parallel.log')
    
    logger = init_log("INFO", name=__name__, file_path=log_file)

    try:
        config = load_log_params_evolution(arguments.experiment_path)
        train_spec = config['train_spec']
        fn_dict = config['fn_dict']

        candidate_ids = arguments.ids or parse_pareto_ids(
            arguments.experiment_path, arguments.top_n, arguments.sort_by
        )
        logger.info(f"Found {len(candidate_ids)} valid candidates to retrain: {candidate_ids}")
        if not candidate_ids:
            logger.error("No candidate IDs found to retrain.")
            return

        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not devices: devices = ["cpu"]
        
        if arguments.max_parallel_workers:
            num_workers = min(arguments.max_parallel_workers, len(candidate_ids))
            logger.info(f"Using {num_workers} parallel workers as specified by --max_parallel_workers.")
        else:
            num_workers = len(devices)

        tasks = []
        for i, cid in enumerate(candidate_ids):
            device = devices[i % len(devices)]
            tasks.append((cid, train_spec, fn_dict, arguments, device, log_file))

        final_results = {}
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_results = executor.map(worker, tasks)
            logger.info(f"Starting retraining for {len(tasks)} models on {len(devices)} devices...")
            for cid, res in future_results:
                if isinstance(res, dict) and "error" in res:
                    logger.error(f"--- Worker Error for Candidate {cid} ---\n{res['error']}\n"
                                f"-------------------------------------------")
                else:
                    logger.info(f"Successfully finished retraining for candidate {cid}.")
                    final_results[cid] = res

        # Merge with any existing results so multiple selection rules accumulate
        results_path = os.path.join(arguments.experiment_path, 'retrain_results_parallel.txt')
        if os.path.isfile(results_path):
            try:
                import json as _json
                with open(results_path) as _f:
                    existing = _json.load(_f)
                final_results = {**existing, **final_results}
            except Exception:
                pass  # corrupt/missing file: overwrite silently

        save_results_file(arguments.experiment_path, final_results,
                            file_name='retrain_results_parallel.txt')
        
    except Exception as e:
        logger.critical(f"A critical error occurred in the main process: {e}", exc_info=True)
    finally:
        logger.info("Retraining script finished.")


if __name__ == '__main__':
    # It's important to set the start method for CUDA + multiprocessing
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ids', nargs='+', default=None)
    parser.add_argument('--sort_by', type=str, default=None, choices=['accuracy', 'params', 'inference_time'])
    parser.add_argument('--max_parallel_workers', type=int, default=None,
                        help='Manually set the number of parallel workers. '
                            'Defaults to the number of available GPUs.')
    parser.add_argument('--top_n', type=int, default=None)
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='INFO')
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--epochs_to_eval', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--limit_data', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_repetitions', type=int, default=1)
    parser.add_argument('--lr_scheduler', type=str, default="multistep")
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--data_augmentation', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5)
    parser.add_argument('--patience_retrain', type=int, default=25)
    parser.add_argument('--delta_fraction', type=float, default=0.005)
    parser.add_argument('--network_config', type=str, default=None,
                        choices=['default', 'dense', 'backbone'],
                        help='Override network_config from the evolution config.')
    parser.add_argument('--keep_metrics', action='store_true',
                        help='Keep the full metric suite from the evolution config '
                             '(e.g. HardwareMetrics, FairnessMetric). '
                             'Default: use Accuracy only during retrain.')

    arguments = parser.parse_args()
    main(arguments)