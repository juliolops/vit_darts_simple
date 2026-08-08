# utils/helpers.py — temporary compatibility facade (Block C of the refactor
# roadmap). All implementations now live in the topic modules below; existing
# `from utils.helpers import X` consumers keep working unchanged. Formal
# deprecation happens in F.3 — import from the topic modules in new code.
# Note: the old standalone load_evolved_data was removed on purpose (F.2);
# the canonical implementation is core.config.ConfigParameters.load_evolved_data.
import warnings

warnings.warn(
    "utils.helpers is deprecated. Import from utils.io, utils.logging_utils, "
    "utils.experiment, utils.dataset or utils.visualization instead.",
    DeprecationWarning, stacklevel=2,
)

from utils.io import (
    load_yaml, save_pkl, load_pkl, create_info_file, save_results_file,
    load_log_params_evolution, load_pareto_history, load_history_from_json,
    save_history_to_json, backup_cache, load_cache, update_yaml_file,
    load_retrain_results,
)
from utils.logging_utils import init_log
from utils.experiment import (
    natural_key, check_file_exists, check_files,
    delete_old_dirs, delete_old_dirs_v2, calculate_time,
)
from utils.dataset import download_dataset, setup_dataset_info, dataset_cache
from utils.visualization import (
    plot_confusion_matrix, agg_results, test_acc_mean_std,
    plot_training_history, get_gpu_memory, compute_hypervolume_mixed,
    plot_hypervolume_over_epochs, plot_hypervolume_comparison,
    plot_pareto_evolution, load_data_for_pareto,
)
