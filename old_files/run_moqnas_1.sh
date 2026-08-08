#!/usr/bin/env bash
# MO-QNAS runner — now a thin wrapper over the experiment-matrix launcher.
# Configure algorithms / configs / repeats / GPUs in the matrix file, not here:
#   experiment_matrices/qfamily.yaml
# Extra flags pass through, e.g.:  ./run_moqnas_1.sh --dry-run
exec python launch.py experiment_matrices/qfamily.yaml "$@"
