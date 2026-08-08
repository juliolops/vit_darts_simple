#!/bin/bash

# Define variables para el experimento de retrain
dataset="cifar10"
network_config="default"

exp="exp19"
# Loop para repetir tres veces
for repeat in 1; do
    echo "Starting $exp F13 repeat $repeat"
    exp_path="experiment_${dataset}_qfamily/qnas/${exp}_repeat_${repeat}"

    # Retrain model
    CUDA_VISIBLE_DEVICES=0 python retrain_model.py \
        --experiment_path "$exp_path" \
        --data_path "datasets/${dataset}_data" \
        --dataset "$dataset" \
        --retrain_folder retrain \
        --config_code F13 \
        --log_level INFO \
        --max_epochs 300 \
        --epochs_to_eval 300 \
        --patience_retrain 300 \
        --batch_size 256 \
        --eval_batch_size 256 \
        --device cuda:0 \
        --num_repetitions 3 \
        --lr_scheduler "multistep" \
        --data_augmentation \
        --network_config "$network_config" \
        --optimizer "AdamW"

    # Verificar si el comando anterior fue exitoso
    if [ $? -ne 0 ]; then
        echo "Error: Retrain model script failed for repeat $repeat."
        exit 1
    fi
done

echo "Retrain model script completed successfully."
