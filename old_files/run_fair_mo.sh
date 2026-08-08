#!/usr/bin/env bash
# ===============================
# run_qnas_family.sh
# QNAS / MO-QNAS runner
# ===============================

# —— Which algorithms to run (choose any of: qnas, moqnas) ——
#algos=("qnas" "moqnas")
algos=("moqnas")
# —— Experiment settings ——
dataset="personbin"
data_path="datasets/${dataset}_data_96"
config_dir="experiment_configs/fairness"
exp_root="experiment_${dataset}_qfamily"
log_level="INFO"
network_config="default"
backbone_name="resnet18"           # used only if network_config="backbone"
fitness_metric="best_accuracy"
dataset_sample_size=10000

# NEW: dataset YAML path derived from dataset name
config_path_dataset="dataset_configs/person_bin_96.yaml"

# —— Common toggles (kept from your scripts) ——
use_cache=false                     
early_stopping=false                
en_pop_crossover=true               

# —— QNAS-specific ——
elite_mode_qnas="global_k"          # "single" | "global_k" | "bootstrap_k" | "old"
# Architecture rule toggles (enabled by default in your repo).
# Set to false to APPEND the corresponding '--no-...' flag (matching your scripts).
truncate_after_noop=false           
avoid_consecutive_pool=true         
enforce_noop_in_update=true         

# —— MO-QNAS-specific ——
optimizer="AdamW"                   
save_checkpoints_epochs=5           
data_augmentation=false             
elite_mode_moqnas="moead_topk"      # "single"|"global_k"|"bootstrap_k"|"old"|"moead_topk"  
ref_dir_method="das-dennis"         # "das-dennis"|"dirichlet"                             
continue_path=""                    # resume path, keep empty if not resuming             

# —— dataset size & repeats ——
configs=("config0.yaml")
exps=("exp1")
cuda_devices=("0,1")
num_repeats=3

# --------------- Runner ---------------
for ((j=0; j<${#configs[@]}; j++)); do
  cfg="${config_dir}/${configs[$j]}"
  exp="${exps[$j]}"
  cuda="${cuda_devices[$j]}"

  for algo in "${algos[@]}"; do
    echo "=== ${algo} | ${dataset} | ${configs[$j]} | CUDA=${cuda} ==="
    for ((i=1; i<=num_repeats; i++)); do
      exp_path="${exp_root}/${algo}/${exp}_repeat_${i}"
      mkdir -p "${exp_path}"

      # Common args
      COMMON_ARGS=(
        --experiment_path      "${exp_path}"
        --config_file          "${cfg}"
        --data_path            "${data_path}"
        --dataset              "${dataset}"
        --config_path_dataset  "${config_path_dataset}"
        --limit_data_value     "${dataset_sample_size}"
        --fitness_metric       "${fitness_metric}"
        --network_config       "${network_config}"
        --backbone_name        "${backbone_name}"
        --log_level            "${log_level}"
        $($use_cache && echo --use_cache)
        $($early_stopping && echo --early_stopping)
        $($en_pop_crossover && echo --en_pop_crossover)
        $($truncate_after_noop && echo --no-truncate-after-noop)
        $($avoid_consecutive_pool && echo --no-avoid-consecutive-pool)
        $($enforce_noop_in_update && echo --no-enforce-noop-in-update)
      )

      if [[ "${algo}" == "qnas" ]]; then
        CUDA_VISIBLE_DEVICES="${cuda}" python run_all_evolution.py \
          --algo qnas "${COMMON_ARGS[@]}" \
          --elite_mode "${elite_mode_qnas}"

      elif [[ "${algo}" == "moqnas" ]]; then
        CUDA_VISIBLE_DEVICES="${cuda}" python run_all_evolution.py \
          --algo moqnas "${COMMON_ARGS[@]}" \
          --optimizer "${optimizer}" \
          --save_checkpoints_epochs "${save_checkpoints_epochs}" \
          --elite_mode "${elite_mode_moqnas}" \
          --ref_dir_method "${ref_dir_method}" \
          $( [[ -n "${continue_path}" ]] && echo --continue_path "${continue_path}" ) \
          $( $data_augmentation && echo --data_augmentation )
      fi
    done
  done
done
