#!/usr/bin/env bash
# GA/NSGA-II/NSGA-III/MOEA-D runner — now a thin wrapper over the experiment-matrix launcher.
# Configure algorithms / configs / repeats / GPUs in the matrix file, not here:
#   experiment_matrices/ea.yaml
# Extra flags pass through, e.g.:  ./run_ea_1.sh --dry-run
exec python launch.py experiment_matrices/ea.yaml "$@"
