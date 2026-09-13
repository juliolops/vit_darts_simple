"""Phase 2: multi-objective search (accuracy vs FLOPs) over per-block head pruning.

    python run_all_evolution.py --config_file config.yaml \
        --experiment_path experiment_vit/teste1 \
        --population_size 4 --num_generations 3 --limit_data_value 500 --threads 1
"""
import argparse
import os

from algorithms.nsga import NSGA
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

    engine = NSGA(
        EvalPopulation(params, log_level=args['log_level']),
        args['experiment_path'],
        percentages=params['percentages'],
        population_size=args['population_size'],
        num_generations=args['num_generations'],
        max_num_nodes=params['max_num_nodes'],
        crossover_rate=args['crossover_rate'],
        mutation_rate=args['mutation_rate'],
        log_file=os.path.join(args['experiment_path'], 'log.txt'),
        log_level=args['log_level'],
    )
    population, fitnesses = engine.evolve()
    logger.info("Evolution finished.")

    for i, (ind, fit) in enumerate(zip(population, fitnesses)):
        fit_str = ", ".join(f"{name}={float(v):.4f}" for name, v in zip(engine.objectives, fit))
        print(f"  Ind {i}: chrom={ind.tolist()}  ->  ({fit_str})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory for the log and pareto_history.pkl.')
    parser.add_argument('--population_size', type=int, default=20)
    parser.add_argument('--num_generations', type=int, default=50)
    parser.add_argument('--crossover_rate', type=float, default=0.9)
    parser.add_argument('--mutation_rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='INFO')
    # Optional overrides of the config file.
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--limit_data_value', type=int, default=None,
                        help='Images per candidate (train + val).')
    parser.add_argument('--threads', type=int, default=None,
                        help='Candidates evaluated concurrently.')
    main(vars(parser.parse_args()))
