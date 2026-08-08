"""Project-wide settings and filesystem paths for MoQ-NAS.

Central place for constants that were previously hardcoded across modules
(``algorithms/ga/nsga2.py``, ``algorithms/qnas/moqnas.py``,
``core/cnn/trainer.py``). Paths are anchored to the repository root so they
work regardless of the current working directory.

Notes
-----
Directory names point to the CURRENT layout on purpose; they will be
updated in lockstep when the directories are renamed (stages A.5-A.7 of
the refactor roadmap).
"""
import os

# Absolute path to the repository root (directory containing this file).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset YAML configs (cifar10.yaml, pathmnist.yaml, ...).
DATASET_CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'dataset_configs')

# Multi-objective definition file (objective names/senses) read by the
# NSGA-II and MO-QNAS engines.
CFG_OBJ_PATH = os.path.join(DATASET_CONFIGS_DIR, 'cfg_obj.json')

# Hard limit, in seconds, for training a single candidate during evolution
# before it is aborted (1.5 h). Used by core/cnn/trainer.py.
TRAIN_TIMEOUT = 5400
