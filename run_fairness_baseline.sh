#!/bin/bash

# --- 🎮 GPU SELECTION ---
# Case A: One GPU -> Runs SEQUENTIALLY
CUDA_DEVICES=("1") 

# Case B: Two GPUs -> Runs in PARALLEL
# CUDA_DEVICES=("0" "1")

# --- CONFIGURATION ---
ARCHS="resnet18 resnet50 efficientnet_v2_s convnext_tiny mobilenet_v3_large mnasnet1_0"

# Image settings
IMG_SIZE=96
RESIZE_MODE="letterbox"

# Config files
CONFIG_PERSON="dataset_configs/person_bin_96.yaml"
CONFIG_FACE="dataset_configs/face_bin_96.yaml"

# Data paths
DATA_PERSON="datasets/personbin_data_96"
DATA_FACE="datasets/facebin_data_96"

# ---------------------

# 1. Define Defaults
OPTIONAL_ARGS=""
OUTPUT_DIR="checkpoints/baseline_limit_96"
MODE_DESC="FULL-TRAINING (Pretrained)"

# Default Epochs (Standard Fine-tuning / Head Only)
MAX_EPOCHS=10

# --- 🔄 ARGUMENT PARSING ---
for arg in "$@"; do
    case $arg in
        --head_only)
            OPTIONAL_ARGS="$OPTIONAL_ARGS --freeze_backbone"
            # Update directory logic
            if [[ "$OUTPUT_DIR" == *"scratch"* ]]; then
                OUTPUT_DIR="checkpoints/baseline_scratch_freeze_limit_96"
            else
                OUTPUT_DIR="checkpoints/baseline_freeze_limit_96"
            fi
            MODE_DESC="HEAD-ONLY (Frozen Backbone)"
            
            # Set Epochs for Head Only
            MAX_EPOCHS=10
            ;;
            
        --from_scratch)
            OPTIONAL_ARGS="$OPTIONAL_ARGS --from_scratch"
            # Update directory logic
            if [[ "$OUTPUT_DIR" == *"freeze"* ]]; then
                OUTPUT_DIR="checkpoints/baseline_scratch_freeze_limit_96"
            else
                OUTPUT_DIR="checkpoints/baseline_scratch_limit_96"
            fi
            MODE_DESC="FULL-TRAINING (From Scratch)"
            
            # Set Epochs for From Scratch
            MAX_EPOCHS=50
            ;;
    esac
done

echo "✅ Running Mode: $MODE_DESC"
echo "✅ Max Epochs: $MAX_EPOCHS"
echo "✅ Optional Args: $OPTIONAL_ARGS"

# Create directories
LOG_DIR="logs"
NUM_GPUS=${#CUDA_DEVICES[@]}
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

echo "✅ Log files will be saved in '$LOG_DIR'"
echo "✅ Checkpoints will be saved in '$OUTPUT_DIR'"
echo "✅ GPU Setup: ${NUM_GPUS} device(s): ${CUDA_DEVICES[*]}"
echo "------------------------------------------------------"


for arch in $ARCHS; do
    echo "🚀 Processing ARCHITECTURE: $arch"
    
    LOG_FILE_FACEBIN="$LOG_DIR/facebin_${arch}.log"
    LOG_FILE_PERSONBIN="$LOG_DIR/personbin_${arch}.log"

    if [ "$NUM_GPUS" -ge 2 ]; then
        # ================= PARALLEL MODE =================
        GPU_A=${CUDA_DEVICES[0]}
        GPU_B=${CUDA_DEVICES[1]}

        echo "⚡ Mode: PARALLEL"
        echo "   --> [GPU $GPU_A] Facebin training..."
        CUDA_VISIBLE_DEVICES="$GPU_A" python scripts/fairness_baseline/train.py \
            --config_file $CONFIG_FACE \
            --data_path $DATA_FACE \
            --experiment_path $OUTPUT_DIR \
            --results_csv "$OUTPUT_DIR/facebin_results_acc.csv" \
            --archs $arch \
            --max_epochs $MAX_EPOCHS \
            --limit_data \
            --limit_data_value 10000 \
            $OPTIONAL_ARGS > "$LOG_FILE_FACEBIN" 2>&1 &
            
        echo "   --> [GPU $GPU_B] Personbin training..."
        CUDA_VISIBLE_DEVICES="$GPU_B" python scripts/fairness_baseline/train.py \
            --config_file $CONFIG_PERSON \
            --data_path $DATA_PERSON \
            --experiment_path $OUTPUT_DIR \
            --results_csv "$OUTPUT_DIR/personbin_results_acc.csv" \
            --archs $arch \
            --max_epochs $MAX_EPOCHS \
            --limit_data \
            --limit_data_value 10000 \
            $OPTIONAL_ARGS > "$LOG_FILE_PERSONBIN" 2>&1 &
            
        wait
    else
        # ================= SEQUENTIAL MODE =================
        GPU_A=${CUDA_DEVICES[0]}
        
        echo "🐢 Mode: SEQUENTIAL (using GPU $GPU_A)"
        
        echo "   --> [1/2] Facebin training..."
        CUDA_VISIBLE_DEVICES="$GPU_A" python scripts/fairness_baseline/train.py \
            --config_file $CONFIG_FACE \
            --data_path $DATA_FACE \
            --experiment_path $OUTPUT_DIR \
            --results_csv "$OUTPUT_DIR/facebin_results_acc.csv" \
            --archs $arch \
            --max_epochs $MAX_EPOCHS \
            --limit_data \
            --limit_data_value 10000 \
            $OPTIONAL_ARGS > "$LOG_FILE_FACEBIN" 2>&1
            
        echo "   --> [2/2] Personbin training..."
        CUDA_VISIBLE_DEVICES="$GPU_A" python scripts/fairness_baseline/train.py \
            --config_file $CONFIG_PERSON \
            --data_path $DATA_PERSON \
            --experiment_path $OUTPUT_DIR \
            --results_csv "$OUTPUT_DIR/personbin_results_acc.csv" \
            --archs $arch \
            --max_epochs $MAX_EPOCHS \
            --limit_data \
            --limit_data_value 10000 \
            $OPTIONAL_ARGS > "$LOG_FILE_PERSONBIN" 2>&1
    fi

    echo "✅ Finished ARCHITECTURE: $arch"
    echo "------------------------------------------------------"
done

# --- Evaluation ---
echo "🚀 Starting evaluation..."
CKPT_DIR="$OUTPUT_DIR"
EVAL_GPU=${CUDA_DEVICES[0]}

echo "Evaluating on FairFace..."
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir $CKPT_DIR \
    --dataset_name fairface \
    --csv_path datasets/FairFace/0.25/fairface_val.csv \
    --filter face \
    --img_size $IMG_SIZE \
    --resize_mode $RESIZE_MODE \
    --beta 0.2

echo "Evaluating on FACET..."
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir $CKPT_DIR \
    --dataset_name facet \
    --csv_path datasets/facet_data/facet_eval.csv \
    --filter person \
    --img_size $IMG_SIZE \
    --resize_mode $RESIZE_MODE \
    --beta 0.2 \
    --cache_dir .cache/facet_crops

echo "✅ Evaluation complete."