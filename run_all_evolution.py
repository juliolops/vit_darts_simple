import os
import argparse
from typing import Tuple

from core import config as cfg
from core import evaluation
from core.eval_cache import CachedEvaluator, compute_fingerprint
from utils.helpers import check_files, init_log, download_dataset
from utils.seeding import set_global_seeds

from algorithms.qnas.moqnas import MOQNAS
from algorithms.ga import nsga2, nsga3
from algorithms.ga import moead


def _bootstrap(logger, args) -> Tuple[object, object, str]:
    """Shared setup for all algorithms: config, dataset, EvalPopulation."""
    if not os.path.exists(args['experiment_path']):
        logger.info(f"Creating {args['experiment_path']} ...")
        os.makedirs(args['experiment_path'])

    phase = 'evolution' if not args['continue_path'] else 'continue_evolution'
    if phase == 'continue_evolution':
        logger.info(f"Continue evolution from: {args['continue_path']}. Checking files ...")
        check_files(args['continue_path'])

    if args['early_stopping']:
        logger.info(f"Early stopping is enabled. Patience: {args['patience']} generations.")

    if args["use_cache"]:
        logger.info("Using cached evaluations to speed up runs.")
    
    if args['gpu_list'] is not None:
        logger.info(f"Using GPU devices: {args['gpu_list']}")
    
    if args.get('multi_objective', False):
        logger.info("Multi-objective optimization is enabled.")

    logger.info(f"Getting parameters from {args['config_file']} ...")
    config = cfg.ConfigParameters(args, phase=phase)
    config.get_parameters()

    # moqnas reads generations from the config; let the CLI flag win.
    # Must happen before save_params_logfile so continue_evolution (which
    # reloads the saved params) sees the overridden value.
    if args.get('num_generations') is not None and args.get('algo', '').lower() == 'moqnas':
        logger.info(f"Overriding config max_generations ({config.QNAS_spec.get('max_generations')}) "
                    f"with --num_generations {args['num_generations']}")
        config.QNAS_spec['max_generations'] = args['num_generations']

    logger.info(f"Saving parameters for {config.phase} phase ...")
    config.save_params_logfile()

    precision = config.train_spec.get('precision', 'fp32')
    if precision != 'fp32':
        logger.info(f"Using mixed precision training (precision={precision}) ...")

    dataset_status = download_dataset(params=config.train_spec)
    logger.info("Dataset is already downloaded." if dataset_status else "Dataset downloaded successfully.")

    eval_pop = evaluation.EvalPopulation(
        params=config.train_spec,
        fn_dict=config.fn_dict,
        log_level=config.train_spec['log_level']
    )
    if args.get('use_cache'):
        # Unified cache (core/eval_cache.py) wraps the evaluator for EVERY
        # algorithm, moqnas included; the legacy per-algorithm caches stay off.
        noop_names = frozenset(
            name for name, spec in config.fn_dict.items()
            if spec.get('function') == 'NoOp'
        )
        eval_pop = CachedEvaluator(eval_pop, config.train_spec,
                                   noop_names=noop_names,
                                   log_level=config.train_spec['log_level'])
        logger.info(f"Unified evaluation cache enabled at {eval_pop.cache_path}")
    return config, eval_pop, phase


def _setup_checkpoint(engine, config, args, logger):
    """Inject checkpoint identity and, with --resume, restore the saved state.

    The config block stored in every checkpoint carries the evaluation
    fingerprint (the same notion of evaluation identity used by the cache),
    the precision and the seed. Without --resume an existing checkpoint is
    ignored and the run starts from generation 0 (safe default).
    """
    from algorithms.checkpoint import checkpoint_path, load_checkpoint
    engine.checkpoint_extra = {
        'fingerprint': compute_fingerprint(config.train_spec),
        'precision': config.train_spec.get('precision', 'fp32'),
        'seed': args.get('seed', 42),
    }
    engine.checkpoint_keep_every = config.train_spec.get('checkpoint_keep_every', 0)
    ckpt = checkpoint_path(engine)
    if args.get('resume'):
        completed = load_checkpoint(engine)
        logger.info(f"Resumed from {ckpt}: generation {completed} completed, "
                    f"continuing at {completed + 1}.")
    elif os.path.exists(ckpt):
        logger.info(f"Checkpoint found at {ckpt} but --resume was not given; "
                    f"starting from generation 0 (checkpoint will be overwritten).")


def main(**args):
    set_global_seeds(args.get('seed', 42))
    logger = init_log(args['log_level'], name=__name__)
    logger.info(f"Global seed set to {args.get('seed', 42)}")
    config, eval_pop, _ = _bootstrap(logger, args)

    algo = args.get('algo', 'nsga2').lower()
    logger.info(f"Selected algorithm: {algo}")

    if args.get('resume') and algo not in ('moqnas', 'nsga2', 'nsga3', 'moead'):
        raise ValueError(f"--resume is not supported for --algo {algo}.")

    if args.get('num_generations') is None:
        args['num_generations'] = 50  # nsga2/nsga3/moead default; moqnas uses the config value

    # If fn_list / max_num_nodes weren't set via CLI, fall back to the config.
    # max_num_nodes is the chromosome length, which the ViT space ties to the
    # transformer block count — a stale CLI default would silently add genes
    # that decode to nothing.
    if args.get('fn_list') is None:
        args['fn_list'] = config.QNAS_spec.get('fn_list')
    if args.get('max_num_nodes') is None:
        args['max_num_nodes'] = config.QNAS_spec['max_num_nodes']

    # -------- Instantiate the engine --------
    if algo == 'nsga3':
        if nsga3 is None:
            raise ImportError("algorithms.ga.nsga3 not found.")
        logger.info("Using NSGA-III.")
        engine = nsga3.NSGA3(
            eval_pop,
            config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
            use_cache=False,  # unified cache wraps eval_pop (core/eval_cache.py)
            ref_divisions=args.get('ref_divisions', None),
        )

    elif algo == 'nsga2':
        if nsga2 is None:
            raise ImportError("algorithms.ga.nsga2 not found.")
        logger.info("Using NSGA-II.")
        engine = nsga2.NSGA2(
            eval_func=eval_pop,
            experiment_path=config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
            use_cache=False,  # unified cache wraps eval_pop (core/eval_cache.py)
        )
    elif algo == 'moead':
        if moead is None:
            raise ImportError("algorithms.ga.moead not found.")
        logger.info("Using MOEA/D.")
        engine = moead.MOEAD(
            eval_func=eval_pop,
            experiment_path=config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
            use_cache=False,  # unified cache wraps eval_pop (core/eval_cache.py)
            divisions=args.get('ref_divisions', None),          # reuse NSGA-III arg name
            T=args.get('moead_T', 20),
            scalar_method=args.get('moead_scalar', 'tchebycheff'),
            prob_neighbor_mating=args.get('moead_pneighbor', 0.9),
        )
    elif algo == 'moqnas':
        if MOQNAS is None:
            raise ImportError("algorithms.qnas.moqnas.MOQNAS not found.")
        logger.info("Using MO-QNAS.")
        engine = MOQNAS(
            eval_func=eval_pop,
            experiment_path=config.train_spec['experiment_path'],
            objectives=config.train_spec['objectives'],
            log_file=config.files_spec['log_file'],
            log_level=config.train_spec['log_level'],
            data_file=config.files_spec['data_file'],
        )
        # Special initializer for MO-QNAS
        engine.initialize_moqnas(**config.QNAS_spec)

        _setup_checkpoint(engine, config, args, logger)

        logger.info("Starting MO-QNAS evolution ...")
        engine.evolve()
        logger.info("MO-QNAS evolution finished.")
        return

    else:
        raise ValueError("Unknown --algo. Choose from: nsga2, nsga3, moead, moqnas.")

    # -------- Shared GA-style init (NSGA-II / NSGA-III / MOEA-D) --------
    engine.initialize_ga(
        population_size=args['population_size'],
        num_generations=args['num_generations'],
        max_num_nodes=args['max_num_nodes'],
        fn_list=args['fn_list'],
        crossover_rate=args['crossover_rate'],
        mutation_rate=args['mutation_rate'],
        early_stopping=args['early_stopping'],
        elitism=args['elitism'],
        patience=args['patience'],
        params_ranges=config.QNAS_spec['params_ranges'],
        mutation_strategy=args['mutation_strategy']
    )

    # Checkpoint/resume wiring for the GA family (after initialize_ga, before
    # evolve, so the restored state overrides the fresh initial population).
    _setup_checkpoint(engine, config, args, logger)

    # -------- Run evolution --------
    logger.info("Starting evolution ...")
    population, fitnesses = engine.evolve()
    logger.info("Evolution finished.")
    if population is not None and fitnesses is not None:
        # Print generically over the configured objectives (any number/type),
        # instead of assuming a fixed 3-objective accuracy/params/time layout.
        obj_names = config.train_spec['objectives']
        for i, (ind, fit) in enumerate(zip(population, fitnesses)):
            chrom = ind.tolist() if hasattr(ind, "tolist") else ind
            fit_str = ", ".join(f"{name}={float(v):.4f}" for name, v in zip(obj_names, fit))
            print(f"  Ind {i}: chrom={chrom}  →  ({fit_str})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where to write logs and model files.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input data.')
    parser.add_argument('--config_path_dataset', type=str, required=True,
                        help='Path to dataset config file (YAML).')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10'],
                        help='Dataset name.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Configuration file name.')
    parser.add_argument('--continue_path', type=str, default='',
                        help='If resuming a previous evolution, point to its experiment path.')
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging verbosity.')
    parser.add_argument('--seed', type=int, default=int(os.environ.get('REFACTOR_SEED', 42)),
                        help='Global RNG seed (random/numpy/torch). Defaults to $REFACTOR_SEED or 42.')

    # Training / pipeline toggles (kept from your scripts)
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['RMSProp', 'Adam', 'AdamW', 'SGD'])
    parser.add_argument('--fitness_metric', type=str, default='best_accuracy',
                        choices=['best_accuracy', 'best_loss', 'scalar_multi_objective'])
    parser.add_argument('--data_augmentation', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--en_pop_crossover', action='store_true')
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5)
    parser.add_argument('--limit_data_value', type=int, default=10000)
    parser.add_argument('--backbone_name', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large',
                                'mobilenet_v2', 'resnet18', 'resnet50'])
    parser.add_argument('--network_config', type=str, default='default',
                        choices=['default', 'dense', 'backbone'])

    # Algorithm selector
    parser.add_argument('--algo', type=str, default='nsga2',
                        choices=['nsga2', 'nsga3', 'moead', 'moqnas'],
                        help='Which multi-objective evolutionary algorithm to run.')

    # NSGA-II/III and MOEA/D params (shared GA-style init)
    parser.add_argument('--population_size', type=int, default=20)
    parser.add_argument('--num_generations', type=int, default=None,
                        help='Number of generations. Defaults: 50 for nsga2/nsga3/moead; '
                             'config max_generations for moqnas (CLI value overrides it).')
    parser.add_argument('--max_num_nodes', type=int, default=None,
                        help='Chromosome length. Defaults to QNAS.max_num_nodes '
                             'from the config file.')
    parser.add_argument('--fn_list', nargs='+', type=str, default=None,
                        help='Layer/op names used to decode chromosomes (fallback to config).')
    parser.add_argument('--crossover_rate', type=float, default=0.9)
    parser.add_argument('--mutation_rate', type=float, default=0.05)
    parser.add_argument('--elitism', action='store_true')
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--mutation_strategy', type=str, default='random',
                        choices=['random', 'standard', 'swap', 'block', 'neighbor'],
                        help='Strategy for mutation: random (mixed), standard (gene replacement), swap, block, or neighbor.')
    # ---------------------------

    # NSGA-III specific
    parser.add_argument('--ref_divisions', type=int, default=None,
                        help='NSGA-III lattice divisions p (auto if None).')
    
    # MOEA/D specific
    parser.add_argument('--moead_T', type=int, default=20,
                        help='Neighborhood size T for MOEA/D.')
    parser.add_argument('--moead_scalar', type=str, default='tchebycheff',
                        choices=['tchebycheff', 'weighted_sum'],
                        help='Scalarization method for MOEA/D.')
    parser.add_argument('--moead_pneighbor', type=float, default=0.9,
                        help='Probability of mating within neighborhood in MOEA/D.')

    # QNAS/MO-QNAS extras (kept for compatibility)
    parser.add_argument('--elite_mode', type=str, default='global_k',
                        choices=['single', 'global_k', 'bootstrap_k', 'old', 'moead_topk'])
    parser.add_argument('--ref_dir_method', type=str, default='das-dennis',
                        choices=['das-dennis', 'dirichlet'])
    parser.add_argument('--no-truncate-after-noop', action='store_true', dest='truncate_after_noop',
                        help='Disable truncating architectures after the first no-op.')
    parser.add_argument('--no-avoid-consecutive-pool', action='store_true', dest='avoid_consecutive_pool',
                        help='Disable the rule preventing consecutive pooling layers.')
    parser.add_argument('--no-enforce-noop-in-update', action='store_true', dest='enforce_noop_in_update',
                        help='Disable enforcing no-op rules during the quantum update.')
    
    parser.add_argument('--multi_objective', action='store_true', default=False,
                        help='Enable multi-objective optimization (MO-QNAS).')

    # Cache
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume a moqnas run from <experiment_path>/checkpoint.pkl. '
                             'Without this flag an existing checkpoint is ignored.')
    parser.add_argument('--use_cache', action='store_true', default=False,
                        help='Use cached evaluations to speed up runs.')

    parser.add_argument('--gpu_list', type=str, default=None, help='String of CUDA devices to be used (e.g., "0,1")')
    parser.add_argument('--workers_per_gpu', type=int, default=None,
                        help='Candidates evaluated concurrently per visible GPU. '
                             'Overrides the config value; takes precedence over threads.')
    parser.add_argument('--threads', type=int, default=None,
                        help='Total evaluation worker processes (legacy; overrides the config).')
    parser.add_argument('--train_timeout', type=int, default=None,
                        help='Per-candidate training timeout in seconds. Overrides the config '
                             'value and the global TRAIN_TIMEOUT in settings.py.')

    args = parser.parse_args()
    main(**vars(args))