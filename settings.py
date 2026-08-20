"""Project-wide settings and filesystem paths for MoQ-NAS.

Central place for constants shared across modules (``algorithms/ga/nsga2.py``,
``algorithms/qnas/moqnas.py``, ``core/training/trainer.py``). Paths are anchored
to the repository root so they work regardless of the current working
directory.
"""
import os

# Absolute path to the repository root (directory containing this file).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset YAML configs (cifar10_vit.yaml, plus the shared cfg_obj.json senses file).
DATASET_CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'dataset_configs')

# Multi-objective definition file (objective names/senses) read by the
# NSGA-II and MO-QNAS engines.
CFG_OBJ_PATH = os.path.join(DATASET_CONFIGS_DIR, 'cfg_obj.json')

# Hard limit, in seconds, for training a single candidate during evolution
# before it is aborted (1.5 h). Used by core/training/trainer.py.
TRAIN_TIMEOUT = 5400
