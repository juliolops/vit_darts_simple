"""Phase 2: multi-objective search (accuracy vs FLOPs) over per-block head pruning.

    python run_all_evolution.py --config_file config.yaml \
        --experiment_path experiment_vit/teste1 --algorithm nsga2 \
        --population_size 4 --num_generations 3 --limit_data_value 500 --threads 1
"""
import argparse
import os

from algorithms.nsga import run_search
from core.config import load_config
from core.evaluation import EvalPopulation
from core.utils import init_log, set_global_seeds


def main(args: dict):
    set_global_seeds(args['seed'])
    logger = init_log(args['log_level'], name=__name__)
    logger.info(f"Global seed set to {args['seed']}")
    os.makedirs(args['experiment_path'], exist_ok=True)

    params = load_config(args)
    logger.info(f"Config: {params}")

    archive = run_search(
        args['algorithm'],
        EvalPopulation(params, log_level=args['log_level']),
        args['experiment_path'],
        percentages=params['percentages'],
        population_size=args['population_size'],
        num_generations=args['num_generations'],
        num_genes=params['num_genes'],
        crossover_rate=args['crossover_rate'],
        mutation_rate=args['mutation_rate'],
        seed=args['seed'],
        log_file=os.path.join(args['experiment_path'], 'log.txt'),
        log_level=args['log_level'],
    )
    logger.info("Evolution finished.")

    for cid, chrom, objectives in archive:
        obj_str = ", ".join(f"{name}={float(v):.4f}" for name, v in objectives.items())
        print(f"  {cid}: chrom={chrom}  ->  ({obj_str})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory for the log and pareto_history.pkl.')
    parser.add_argument('--algorithm', choices=['nsga2', 'nsga3'], default='nsga2')
    parser.add_argument('--population_size', type=int, default=20)
    parser.add_argument('--num_generations', type=int, default=50)
    parser.add_argument('--crossover_rate', type=float, default=0.9,
                        help='Probability that a pair of parents is recombined (SBX).')
    parser.add_argument('--mutation_rate', type=float, default=None,
                        help='Per-gene mutation probability (default: 1 / number of genes).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='INFO')
    # Optional overrides of the config file.
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--limit_data_value', type=int, default=None,
                        help='Images per candidate (train + val).')
    parser.add_argument('--threads', type=int, default=None,
                        help='Candidates evaluated concurrently.')
    main(vars(parser.parse_args()))
