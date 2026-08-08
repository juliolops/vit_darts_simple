""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Q-NAS configuration.
"""

import inspect
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import yaml
import re

from settings import CFG_OBJ_PATH
from core.precision import resolve_precision
from .cnn import model
from utils.helpers import load_yaml, load_pkl, natural_key

# Output names each metric plugin contributes to the evaluation results.
# The trainer itself always provides the *builtin* names regardless of plugins.
# Used by _check_objectives to fail fast on objectives nothing will produce.
METRIC_PROVIDES = {
    'Accuracy': {'accuracy'},
    'MedMNIST_Metrics': {'auc_score', 'acc_medmnist'},
    'HardwareMetrics': {'cuda_inference_time', 'total_params', 'total_flops',
                        'model_memory_usage'},
    'ScalarizedFitness': {'scalar_multi_objective'},
    'ValidationLossFitness': {'fitness_val_loss'},
    'FairnessMetric': {'fairness_spd', 'fairness_mean_tpr', 'fairness_score'},
}
TRAINER_BUILTIN_METRICS = {'best_accuracy', 'best_loss'}


class ConfigParameters(object):
    """Handles the loading, validation, and organization of all configuration parameters."""

    def __init__(self, args: dict, phase: str):
        """Initializes the ConfigParameters object.

        Args:
            args (dict): A dictionary of command-line arguments.
            phase (str): The current operational phase, which must be one of
                        'evolution', 'continue_evolution', or 'retrain'.
        """
        self.phase = phase
        self.args = args
        self.QNAS_spec = {}
        self.train_spec = {}
        self.files_spec = {}
        self.fn_dict = {}
        self.previous_params_file = None
        self.data_info = None
        self.evolved_params = None

    def _check_vars(self, config_file: dict):
        """Validates the structure and types of parameters in the configuration file.

        This method ensures that all required keys are present in the config file
        and that their corresponding values have the correct data types. It also
        performs value-range checks for specific parameters like learning rates
        and function probabilities.

        Args:
            config_file (dict): The loaded configuration from the YAML file.

        Raises:
            KeyError: If a required variable is missing.
            TypeError: If a variable has an incorrect type.
            ValueError: If a parameter's value is out of its allowed bounds.
        """

        def check_params_ranges():
            """Checks if hyperparameter search ranges are within safe, predefined limits."""
            ranges = config_file['QNAS']['params_ranges']
            allowed = {'decay': (1e-6, 1.0), 'learning_rate': (1e-6, 1.0),
                        'momentum': (0.0, 1.0), 'weight_decay': (1e-10, 1e-1),
                        'backbone_percentage': (0.1, 1.0)}

            for key, value in ranges.items():
                limit = allowed.get(key)
                if not limit: continue
                
                low, high = (value[0], value[1]) if isinstance(value, list) else (value, value)
                if not (limit[0] <= low and high <= limit[1]):
                    raise ValueError(f'{key} value out of bound {limit}!')

        def check_fn_dict():
            """Validates the function dictionary, checking for valid function names and probabilities."""
            available_fn = [c[0] for c in inspect.getmembers(model, inspect.isclass)]
            fn_dict = config_file['QNAS']['function_dict']
            probs = []

            for name, definition in fn_dict.items():
                if definition['function'] not in available_fn:
                    raise ValueError(f"{definition['function']} is not a valid function!")
                for param in definition['params'].values():
                    if not isinstance(param, int) or param < 0:
                        raise ValueError(f"{name} has an invalid parameter: {definition['params']}!")
                
                prob_val = definition['prob']
                probs.append(eval(prob_val) if isinstance(prob_val, str) else prob_val)

            if any(p is not None for p in probs):
                prob_sum = np.sum([p for p in probs if p is not None])
                if not np.isclose(prob_sum, 1.0):
                    raise ValueError(f"Function probabilities should sum to 1.0, but sum to {prob_sum}.")

        vars_dict = {
            'QNAS': [('crossover_rate', float), ('max_generations', int), ('max_num_nodes', int),
                    ('num_quantum_ind', int), ('penalize_number', int), ('repetition', int),
                    ('replace_method', str), ('quantum_update_config', dict), ('update_quantum_gen', int),
                    ('params_ranges', dict), ('patience', int),
                    ('crossover_frequency', int), ('pop_crossover_rate', float), ('pop_crossover_method', list),
                    ('mo_crossover_strategy', str), ('elite_mode', str), ('k_elites', int), ('pool_factor', int),
                    ('ema_beta', float), ('rank_weighting', bool),
                    ('initial_prob_distribution', str), ('function_dict', dict),
                    ('terminal_op_name', str), ('pool_op_name', str), ('min_active_len', int),
                    ('truncate_after_noop', bool), ('avoid_consecutive_pool', bool), ('enforce_noop_in_update', bool),
                    ('noop_max_prob', float), ('noop_ramp_cap', bool)],
            
            'train': [('batch_size', int), ('eval_batch_size', int), ('max_epochs', int),
                    ('epochs_to_eval', int), ('optimizer', str), ('device', str),
                    ('dataset', str),
                    ('objectives', list), ('multi_objective', bool), ('metrics', list),
                    ('data_augmentation', bool), ('subtract_mean', bool), ('artifacts', list),
                    ('limit_data', bool), ('limit_data_value', int), ('backbone_name', str),
                    ('network_config', str), ('threads', int),
                    ('train_split', float), ('split_seed', int), ('loader_seed', int), ('download', bool),
                    ('stats_max_batches', int), ('num_workers', int),
                    ]
        }

        for config, items in vars_dict.items():
            for var_name, var_type in items:
                var = config_file[config].get(var_name)
                if var is None:
                    raise KeyError(f"Variable \"{config}:{var_name}\" not found in config file.")
                if not isinstance(var, var_type):
                    raise TypeError(f"Variable {var_name} should be {var_type} but is {type(var)}")

        # Precision: accept either explicit 'precision' (str) or legacy 'mixed_precision' (bool).
        train = config_file['train']
        precision_val = train.get('precision')
        mixed_val = train.get('mixed_precision')
        if precision_val is None and mixed_val is None:
            raise KeyError("train config must have 'precision' (fp32|fp16|bf16) "
                           "or legacy 'mixed_precision' (bool).")
        if precision_val is not None and precision_val not in ('fp32', 'fp16', 'bf16'):
            raise ValueError(f"train:precision must be fp32|fp16|bf16, got {precision_val!r}.")
        
        check_params_ranges()
        check_fn_dict()

        if config_file['train']['epochs_to_eval'] >= config_file['train']['max_epochs']:
            raise ValueError('Invalid epochs_to_eval! It should be < max_epochs.')

        # Optional; defaults to 'max' (current behavior) when absent.
        agg = config_file['train'].get('eval_window_agg', 'max')
        if agg not in ('max', 'mean', 'last'):
            raise ValueError(f"Invalid eval_window_agg '{agg}'! Use 'max', 'mean' or 'last'.")

    def _get_evolution_params(self):
        """Loads and organizes parameters for a new evolution run."""
        config_file = load_yaml(self.args['config_file'])
        self._check_vars(config_file)

        self.train_spec = dict(config_file['train'])
        self.QNAS_spec = dict(config_file['QNAS'])
        self.files_spec['config_file'] = self.args['config_file']

        ranges = self._get_ranges(config_file)
        self.QNAS_spec['params_ranges'] = OrderedDict(sorted(ranges.items()))
        self.QNAS_spec['early_stopping'] = self.args.get('early_stopping')
        self.QNAS_spec['en_pop_crossover'] = self.args.get('en_pop_crossover')
        self.QNAS_spec['elite_mode'] = self.args.get('elite_mode', 'global_k')
        self.QNAS_spec['truncate_after_noop'] = self.args.get('truncate_after_noop', True)
        self.QNAS_spec['avoid_consecutive_pool'] = self.args.get('avoid_consecutive_pool', True)
        self.QNAS_spec['enforce_noop_in_update'] = self.args.get('enforce_noop_in_update', True)
        if self.train_spec['multi_objective']:
            self.QNAS_spec['ref_dir_method'] = self.args.get('ref_dir_method', 'das-dennis')
        self._get_fn_spec()

        train_override_keys = [
            'optimizer', 'data_augmentation', 'network_config',
            'backbone_name', 'dataset', 'data_path', 'limit_data_value',
            'multi_objective', 'objectives', 'config_path_dataset', 'gpu_list',
            'seed', 'workers_per_gpu', 'threads', 'train_timeout',
        ]
        for key in train_override_keys:
            val = self.args.get(key)
            if val is not None:
                self.train_spec[key] = val

        self.train_spec['experiment_path'] = self.args['experiment_path']

        # Normalize the precision policy once; downstream code reads
        # train_spec['precision'] ('fp32'|'fp16'|'bf16').
        if 'precision' not in self.train_spec and self.train_spec.get('mixed_precision', False):
            logging.getLogger(__name__).info(
                "train.precision not set; derived 'fp16' from the legacy "
                "mixed_precision flag. Set precision: fp16|bf16|fp32 explicitly.")
        self.train_spec['precision'] = resolve_precision(self.train_spec)

        self._check_objectives()

    def _check_objectives(self):
        """Validate ``train.objectives`` at parse time, before any training.

        Two checks, both fatal:

        1. Sense resolution: every objective name must match exactly one key
           of ``dataset_configs/cfg_obj.json`` under the same substring rule
           the algorithms use (``key in objective``). Zero matches used to be
           a silently-ignored warning in NSGA2/MOQNAS that left
           ``objective_senses`` shorter than the fitness matrix, flipping the
           wrong columns; more than one match is ambiguous.
        2. Producibility: every objective must be provided by the trainer
           builtins or by one of the configured metric plugins (per
           METRIC_PROVIDES). Skipped if the config declares a metric this
           table does not know about (forward compatibility).

        Raises
        ------
        ValueError
            Naming the offending objective and listing the valid options.
        """
        objectives = self.train_spec.get('objectives') or []
        with open(CFG_OBJ_PATH, 'r') as f:
            senses = json.load(f)['objectives']

        for obj in objectives:
            matches = [k for k in senses if k in obj]
            if len(matches) == 0:
                raise ValueError(
                    f"Objective '{obj}' matches no sense rule in {CFG_OBJ_PATH}. "
                    f"Available sense keys: {sorted(senses)}")
            if len(matches) > 1:
                raise ValueError(
                    f"Objective '{obj}' is ambiguous: it matches multiple sense "
                    f"rules {matches} in {CFG_OBJ_PATH}. Rename the objective or "
                    f"the rules so exactly one applies.")

        metrics_cfg = self.train_spec.get('metrics') or []
        metric_names = [m.get('name') for m in metrics_cfg]
        if not metrics_cfg:
            # Legacy single-objective configs declare no metrics; the trainer
            # fills hardware-style names with constant 0.0 defaults.
            non_builtin = [o for o in objectives if o not in TRAINER_BUILTIN_METRICS]
            if non_builtin:
                logging.getLogger(__name__).warning(
                    "Config declares no metrics; objectives %s will evaluate "
                    "as the trainer's 0.0 defaults.", non_builtin)
        elif all(name in METRIC_PROVIDES for name in metric_names):
            providable = set(TRAINER_BUILTIN_METRICS)
            for name in metric_names:
                providable |= METRIC_PROVIDES[name]
            missing = [o for o in objectives if o not in providable]
            if missing:
                raise ValueError(
                    f"Objectives {missing} are not produced by the configured "
                    f"metrics {metric_names} nor by the trainer builtins. "
                    f"Producible names: {sorted(providable)}")

    def _get_fn_spec(self):
        """
        Organize function specifications for QNAS.
        
        This method now supports a new parameter `initial_prob_distribution`
        in the QNAS config, which can be 'from_config' or 'uniform'.
        """
        self.QNAS_spec['fn_list'] = sorted(
            self.QNAS_spec['function_dict'].keys(), key=natural_key
        )
        self.fn_dict = self.QNAS_spec.pop('function_dict')
        self.QNAS_spec['initial_probs'] = []
        self.QNAS_spec['reducing_fns_list'] = []

        # Get the desired distribution method. Default to 'from_config' if not specified.
        prob_distribution_method = self.QNAS_spec.get('initial_prob_distribution', 'from_config')
        
        self.QNAS_spec.pop('initial_prob_distribution', None)

        # Conditionally populate the initial_probs list based on the chosen method.
        if prob_distribution_method == 'from_config':
            print("INFO: Initializing probabilities from the configuration file.")
            for fn in self.QNAS_spec['fn_list']:
                if type(self.fn_dict[fn]['prob']) == str:
                    prob = eval(self.fn_dict[fn]['prob'])
                else:
                    prob = self.fn_dict[fn]['prob']
                
                if prob is not None:
                    self.QNAS_spec['initial_probs'].append(prob)
        
        elif prob_distribution_method == 'uniform':
            # By leaving `initial_probs` as an empty list, we signal the QPopulationNetwork
            # class to create its own uniform distribution.
            print("INFO: Using a uniform initial probability distribution for all operators.")
            pass # Keep self.QNAS_spec['initial_probs'] as []

        else:
            raise ValueError(f"Unknown initial_prob_distribution: '{prob_distribution_method}'. "
                            f"Please use 'from_config' or 'uniform'.")

        # The rest of the function remains the same.
        # It populates the reducing functions list independently of the probabilities.
        for fn in self.QNAS_spec['fn_list']:
            strides = self.fn_dict[fn]['params'].get('strides')
            if strides and strides > 1:
                self.QNAS_spec['reducing_fns_list'].append(fn)
                
        for item in self.fn_dict.values():
            del item['prob']

    def _get_ranges(self, config_file):
        """  Get the ranges of the numerical parameters to be evolved.

        Args:
            config_file: dict holding the parameters in the config file.

        Returns:
            dict containing the extracted ranges.
        """

        if self.train_spec['optimizer'] == 'Momentum':
            ranges = {key: val for key, val in config_file['QNAS']['params_ranges'].items()
                    if key != 'decay' and type(val) == list}
        else:
            ranges = {key: val for key, val in config_file['QNAS']['params_ranges'].items()
                    if type(val) == list}

        # If user provided a value instead of a range, parameter will not be evolved.
        for key, value in config_file['QNAS']['params_ranges'].items():
            if type(value) != list:
                self.train_spec[key] = value

        return ranges

    def _get_continue_params(self):
        """ Get parameters for the continue evolution phase. The evolution parameters are loaded
            from previous evolution configuration, except from the maximum number of generations
            (*max_generations*).
        """

        self.files_spec['continue_path'] = self.args['continue_path']
        self.files_spec['previous_QNAS_params'] = os.path.join(
            self.files_spec['continue_path'], 'log_params_evolution.txt')

        self.files_spec['previous_data_file'] = os.path.join(self.args['continue_path'],
                                                            'data_QNAS.pkl')
        self.load_old_params()
        self.QNAS_spec['max_generations'] = load_yaml(
                self.args['config_file'])['QNAS']['max_generations']

        self.train_spec['experiment_path'] = self.args['experiment_path']

    def _get_retrain_params(self):
        """ Get specific parameters for the retrain phase. The keys in *self.train_spec* that
            exist in self.args are overwritten.
        """

        self.files_spec['previous_QNAS_params'] = os.path.join(self.args['experiment_path'],
                                                            'log_params_evolution.txt')
        self.load_old_params()

        for key in self.args.keys():
            self.train_spec[key] = self.args[key]

        self.train_spec['experiment_path'] = os.path.join(self.train_spec['experiment_path'],
                                                        self.args['retrain_folder'])
        del self.args['retrain_folder']

    def _get_common_params(self):
        """ Get parameters that are combined/calculated the same way for all phases. """

        self.train_spec['data_path'] = self.args['data_path']
        #self.data_info = self.get_data_info()

        # if not self.train_spec['eval_batch_size']:
        #     self.train_spec['eval_batch_size'] = self.data_info.num_valid_ex

        # Calculating parameters based on steps
        #self._calculate_step_params()

        self.train_spec['phase'] = self.phase
        self.train_spec['log_level'] = self.args['log_level']

        self.files_spec['log_file'] = os.path.join(self.args['experiment_path'], 'log_QNAS.txt')
        self.files_spec['data_file'] = os.path.join(self.args['experiment_path'],
                                                    'data_QNAS.pkl')
        
    def get_parameters(self):
        """ Organize dicts combining the command-line and config_file parameters,
            joining all the necessary information for each *phase* of the program.
        """

        if self.phase == 'evolution':
            self._get_evolution_params()
        elif self.phase == 'continue_evolution':
            self._get_continue_params()
        else:
            self._get_retrain_params()
        self._get_common_params()
        # Older saved params files lack the precision key; resolve it for
        # every phase so the trainer always finds a normalized value.
        self.train_spec['precision'] = resolve_precision(self.train_spec)

    def load_old_params(self):
        """ Load parameters from *self.files_spec['previous_QNAS_params']* and replace
            *self.train_spec*, *self.QNAS_spec*, and *self.fn_dict* with the file values.
        """

        previous_params_file = load_yaml(self.files_spec['previous_QNAS_params'])

        self.train_spec = dict(previous_params_file['train'])
        self.QNAS_spec = dict(previous_params_file['QNAS'])
        self.QNAS_spec['params_ranges'] = eval(self.QNAS_spec['params_ranges'])
        self.fn_dict = previous_params_file['fn_dict']
    
    def load_evolved_data(self, experiment_path: str):
        """
        Loads evolved data from the specified experiment path.

        Parameters:
        - experiment_path (str): The path to the experiment folder containing evolved data.

        Returns:
        None

        This method reads the evolved data from the best-performing experiment folder within the specified path.
        It extracts information such as neural network details, generation, and individual from the 'training_params.txt' file.

        If the data is in an old format (generation and individual not specified in 'training_params.txt'),
        it attempts to extract them from the folder name using a regular expression.

        The extracted information is stored in the 'evolved_params' attribute of the class.

        Note: This method assumes a specific folder and file structure for evolved data.
        """
        
        best_so_far_link = os.path.join(experiment_path, 'best_so_far')
        
        if os.path.islink(best_so_far_link):
            best_result_folder = os.readlink(best_so_far_link)
        else:
            experiment_folders = [f.name for f in os.scandir(experiment_path) if f.is_dir()]
            best_result_folder = [name for name in experiment_folders if name[0].isdigit()]
            best_result_folder = os.path.join(experiment_path, best_result_folder[0])
            
        with open(os.path.join(best_result_folder, 'training_params.txt'), 'r') as file:
                best_individual_info = yaml.safe_load(file)
        net_list = best_individual_info.get('net_list', [])
        generation = best_individual_info.get('generation', 0)
        individual = best_individual_info.get('individual', 0)
        backbone_name = best_individual_info.get('backbone_name', None)
        backbone_percentage = best_individual_info.get('backbone_percentage', 0)
            
        if generation == 0 and individual == 0: # only for old format
                matches = re.search(r'(\d+)_(\d+)$', best_result_folder)
                generation = int(matches.group(1))
                individual = int(matches.group(2))

        self.evolved_params = {'params': None, 'net': net_list, 'generation': generation, 
                                'individual': individual, 'backbone_name':backbone_name, 
                                'backbone_percentage':backbone_percentage}

    def override_train_params(self, new_params_dict):
        """ Override *self.train_spec* parameters with the ones in *new_params_dict*. Update
            step parameters, in case a epoch parameter was modified.

        Args:
            new_params_dict: dict containing parameters to override/add to self.train_spec.
        """

        self.train_spec.update(new_params_dict)

        # Recalculating parameters based on steps
        self._calculate_step_params()

    def params_to_logfile(self, params, text_file, nested_level=0):
        """ Print dictionary *params* to a txt file with nested level formatting.

        Args:
            params: dictionary with parameters.
            text_file: file object.
            nested_level: level of nested dictionary.
        """

        spacing = '    '
        if type(params) == dict:
            for key, value in OrderedDict(sorted(params.items())).items():
                if type(value) == dict:
                    if nested_level < 2:
                        print(f'{nested_level * spacing}{key}:', file=text_file)
                        self.params_to_logfile(value, text_file, nested_level + 1)
                    else:
                        print(f'{nested_level * spacing}{key}: {value}', file=text_file)
                else:
                    if type(value) == float:
                        if value < 1e-3:
                            print(f'{nested_level * spacing}{key}: {value:.2E}', file=text_file)
                        else:
                            print(f'{nested_level * spacing}{key}: {value:.4f}', file=text_file)
                    else:
                        print(f'{nested_level * spacing}{key}: {value}', file=text_file)
                if nested_level == 0:
                    print('', file=text_file)

    def save_params_logfile(self):
        """ Helper function to save the parameters in a txt file. """
        # data_dict = {key: value for key, value in self.data_info.__dict__.items()
        #              if key != 'mean_image'}
        
        if self.train_spec['phase'] == 'retrain':
            phase = 'retrain'
            params_dict = {'evolved_params': self.evolved_params,
                            'train': self.train_spec,
                            'files': self.files_spec}
                            #'train_data_info': data_dict}
        else:
            phase = 'evolution'
            params_dict = {'QNAS': self.QNAS_spec,
                            'train': self.train_spec,
                            'files': self.files_spec,
                            'fn_dict': self.fn_dict}
                            #'train_data_info': data_dict}

        params_file_path = os.path.join(self.train_spec['experiment_path'],
                                        f'log_params_{phase}.txt')

        with open(params_file_path, mode='w') as text_file:
            self.params_to_logfile(params_dict, text_file)