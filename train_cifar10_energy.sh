#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# CIFAR-10 data
DATA_DIR="${DATA_DIR:-${REPO_DIR}/datasets/cifar_train}"

# Energy regularization.
ENERGY_LAMBDA="${ENERGY_LAMBDA:-0.0}"
ENERGY_MODE="${ENERGY_MODE:-batch_mean}"

#logging
OPENAI_LOGDIR="${OPENAI_LOGDIR:-${REPO_DIR}/logs/cifar10_${ENERGY_LAMBDA}_${ENERGY_MODE}_1}"
export OPENAI_LOGDIR

# Launch mode.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Training hyperparameters.
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-128}"
MICROBATCH="${MICROBATCH:--1}"
LR="${LR:-0.0002}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-linear}"
EMA_RATE="${EMA_RATE:-0.9999}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-800000}"
USE_FP16="${USE_FP16:-False}"

# Resume support.
# RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/storage2/Diffusion/repos/improved-diffusion/logs/cifar10_0.0_batch_mean/model670000.pt}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"


# Optional dataset prep.
PREPARE_DATA="${PREPARE_DATA:-0}"

if [[ "${PREPARE_DATA}" == "1" ]]; then
  "${PYTHON_BIN}" datasets/cifar10.py
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Data directory not found: ${DATA_DIR}"
  echo "Create it with:"
  echo "  ${PYTHON_BIN} datasets/cifar10.py"
  exit 1
fi

mkdir -p "${OPENAI_LOGDIR}"

CMD=(
  "${PYTHON_BIN}" scripts/image_train.py
  --data_dir "${DATA_DIR}"
  --image_size 32
  --num_channels 128
  --num_res_blocks 3
  --learn_sigma False
  --dropout 0.1
  --diffusion_steps "${DIFFUSION_STEPS}"
  --noise_schedule "${NOISE_SCHEDULE}"
  --lr "${LR}"
  --batch_size "${PER_GPU_BATCH_SIZE}"
  --microbatch "${MICROBATCH}"
  --ema_rate "${EMA_RATE}"
  --log_interval "${LOG_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
  --lr_anneal_steps "${LR_ANNEAL_STEPS}"
  --use_fp16 "${USE_FP16}"
  --energy_lambda "${ENERGY_LAMBDA}"
  --energy_mode "${ENERGY_MODE}"
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

echo "Repo dir:           ${REPO_DIR}"
echo "Python:             ${PYTHON_BIN}"
echo "Data dir:           ${DATA_DIR}"
echo "Log dir:            ${OPENAI_LOGDIR}"
echo "Visible GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "Processes/node:     ${NPROC_PER_NODE}"
echo "Per-GPU batch size: ${PER_GPU_BATCH_SIZE}"
echo "Energy lambda:      ${ENERGY_LAMBDA}"
echo "Energy mode:        ${ENERGY_MODE}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint:  ${RESUME_CHECKPOINT}"
else
  echo "Resume checkpoint:  <none>"
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" "${CMD[@]:1}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${CMD[@]}"
fi
