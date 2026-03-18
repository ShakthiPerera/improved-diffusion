#!/bin/bash
# Run the full reverse-path analysis pipeline for ONE model.
#
# Usage:
#   bash scripts/run_full_pipeline.sh <MODEL_ID> <CKPT_PATH> <PREDICT_XSTART> <GPU_ID> [BATCH_SIZE]
#
# Arguments:
#   MODEL_ID        e.g. M0_ddpm_eps
#   CKPT_PATH       e.g. logs/exp01/ddpm_eps/ema_0.9999_150000.pt
#   PREDICT_XSTART  "False" for eps-prediction models, "True" for x0-prediction models
#   GPU_ID          e.g. 0, 1, 2
#   BATCH_SIZE      (optional) default 128; increase to 256/512 if GPU has headroom
#
# Examples:
#   bash scripts/run_full_pipeline.sh M0_ddpm_eps logs/exp01/ddpm_eps/ema_0.9999_150000.pt False 0
#   bash scripts/run_full_pipeline.sh M1_ddpm_x0  logs/exp01/ddpm_x0/ema_0.9999_250000.pt  True  0 256
#
# Check free VRAM before picking BATCH_SIZE:
#   nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader -i <GPU_ID>

set -e

MODEL_ID=$1
CKPT_PATH=$2
PREDICT_XSTART=$3
GPU_ID=$4
BATCH_SIZE=${5:-128}

if [ -z "$MODEL_ID" ] || [ -z "$CKPT_PATH" ] || [ -z "$PREDICT_XSTART" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: bash scripts/run_full_pipeline.sh <MODEL_ID> <CKPT_PATH> <PREDICT_XSTART> <GPU_ID> [BATCH_SIZE]"
    exit 1
fi

OUTPUT_DIR="reverse_path_analysis/${MODEL_ID}"

# Model and diffusion flags — must match what was used during training
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3
             --learn_sigma False --dropout 0.1 --attention_resolutions 16,8
             --use_scale_shift_norm True --num_heads 4"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear
                 --rescale_timesteps True"

echo "============================================"
echo "Model:         ${MODEL_ID}"
echo "Checkpoint:    ${CKPT_PATH}"
echo "predict_xstart: ${PREDICT_XSTART}"
echo "GPU:           ${GPU_ID}"
echo "batch_size:    ${BATCH_SIZE}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "============================================"

# ── Step 1: Reverse path data collection ──────────────────────────────────
echo ""
echo "[1/3] Running reverse_path_analysis.py ..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/reverse_path_analysis.py \
    --model_path "${CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --z_T_path   reverse_path_analysis/shared/z_T.pt \
    --num_samples 10000 \
    --predict_xstart ${PREDICT_XSTART} \
    --batch_size ${BATCH_SIZE} \
    ${MODEL_FLAGS} \
    ${DIFFUSION_FLAGS}

# ── Step 2: Image-space metrics ────────────────────────────────────────────
echo ""
echo "[2/3] Running compute_image_metrics.py ..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/compute_image_metrics.py \
    --model_dir             "${OUTPUT_DIR}" \
    --model_id              "${MODEL_ID}" \
    --real_features_path    reverse_path_analysis/shared/real_inception_features.pt \
    --prdc_max_t            200 \
    --batch_size            64

# ── Step 3: Compile CSVs ───────────────────────────────────────────────────
echo ""
echo "[3/3] Running compile_model_csvs.py ..."
python scripts/compile_model_csvs.py \
    --model_dir "${OUTPUT_DIR}" \
    --model_id  "${MODEL_ID}"

# ── Step 4: Write TensorBoard logs ────────────────────────────────────────
echo ""
echo "[4/4] Writing TensorBoard logs ..."
python scripts/to_tensorboard.py \
    --model_dir "${OUTPUT_DIR}" \
    --model_id  "${MODEL_ID}" \
    --tb_dir    reverse_path_analysis/tensorboard

echo ""
echo "============================================"
echo "${MODEL_ID} COMPLETE"
echo "  ${OUTPUT_DIR}/${MODEL_ID}_scalars.csv"
echo "  ${OUTPUT_DIR}/${MODEL_ID}_eigenspectra.csv"
echo "  TensorBoard: reverse_path_analysis/tensorboard/${MODEL_ID}"
echo ""
echo "  View all models: tensorboard --logdir reverse_path_analysis/tensorboard"
echo "============================================"
