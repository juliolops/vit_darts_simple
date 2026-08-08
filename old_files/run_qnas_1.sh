#!/usr/bin/env bash
# QNAS runner — now a thin wrapper over the experiment-matrix launcher.
# Configure algorithms / configs / repeats / GPUs in the matrix file, not here:
#   experiment_matrices/qfamily.yaml
# Extra flags pass through, e.g.:  ./run_qnas_1.sh --dry-run
exec python launch.py experiment_matrices/qfamily.yaml "$@"
