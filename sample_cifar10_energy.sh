#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Checkpoint to sample from. Prefer an EMA checkpoint.
MODEL_PATH="${MODEL_PATH:-/scratch1/peramorphiq-branch-prediction/temp/ndiff/improved-diffusion/logs/cifar10_0.3_batch_mean_minsnr5.0/ema_0.9999_800000.pt}"

# Logging/output directory.
MODEL_STEM="$(basename "${MODEL_PATH:-unset}")"
MODEL_STEM="${MODEL_STEM%.pt}"
OPENAI_LOGDIR="${OPENAI_LOGDIR:-${REPO_DIR}/logs/sample_cifar10_0.3_5.0_2${MODEL_STEM}}"
export OPENAI_LOGDIR

# Launch mode.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Sampling hyperparameters.
NUM_SAMPLES="${NUM_SAMPLES:-8000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CLIP_DENOISED="${CLIP_DENOISED:-True}"
USE_DDIM="${USE_DDIM:-False}"
TIMESTEP_RESPACING="${TIMESTEP_RESPACING:-}"
SEED="${SEED:--1}"
SAVE_PNG="${SAVE_PNG:-True}"
PNG_DIR="${PNG_DIR:-}"

# CIFAR-10 model hyperparameters. Keep these aligned with training.
IMAGE_SIZE="${IMAGE_SIZE:-32}"
NUM_CHANNELS="${NUM_CHANNELS:-128}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-3}"
LEARN_SIGMA="${LEARN_SIGMA:-False}"
DROPOUT="${DROPOUT:-0.1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-linear}"

# Energy settings are included for config parity with training.
ENERGY_LAMBDA="${ENERGY_LAMBDA:-0.3}"
ENERGY_MODE="${ENERGY_MODE:-batch_mean}"
MIN_SNR_GAMMA="${MIN_SNR_GAMMA:-5.0}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required."
  echo "Example:"
  echo "  MODEL_PATH=${REPO_DIR}/logs/cifar10_0.3_batch_mean_5.0/ema_0.9999_800000.pt bash sample_cifar10_energy.sh"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model checkpoint not found: ${MODEL_PATH}"
  exit 1
fi

mkdir -p "${OPENAI_LOGDIR}"

CMD=(
  "${PYTHON_BIN}" scripts/image_sample.py
  --model_path "${MODEL_PATH}"
  --image_size "${IMAGE_SIZE}"
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --learn_sigma "${LEARN_SIGMA}"
  --dropout "${DROPOUT}"
  --diffusion_steps "${DIFFUSION_STEPS}"
  --noise_schedule "${NOISE_SCHEDULE}"
  --num_samples "${NUM_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --clip_denoised "${CLIP_DENOISED}"
  --use_ddim "${USE_DDIM}"
  --seed "${SEED}"
  --save_png "${SAVE_PNG}"
  --energy_lambda "${ENERGY_LAMBDA}"
  --energy_mode "${ENERGY_MODE}"
  --min_snr_gamma "${MIN_SNR_GAMMA}"
)

if [[ -n "${TIMESTEP_RESPACING}" ]]; then
  CMD+=(--timestep_respacing "${TIMESTEP_RESPACING}")
fi

if [[ -n "${PNG_DIR}" ]]; then
  CMD+=(--png_dir "${PNG_DIR}")
fi

echo "Repo dir:           ${REPO_DIR}"
echo "Python:             ${PYTHON_BIN}"
echo "Model path:         ${MODEL_PATH}"
echo "Log dir:            ${OPENAI_LOGDIR}"
echo "Visible GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "Processes/node:     ${NPROC_PER_NODE}"
echo "Num samples:        ${NUM_SAMPLES}"
echo "Batch size:         ${BATCH_SIZE}"
echo "Use DDIM:           ${USE_DDIM}"
echo "Timestep respacing: ${TIMESTEP_RESPACING:-<default>}"
echo "Seed:               ${SEED}"
echo "Save PNG:           ${SAVE_PNG}"
echo "PNG dir:            ${PNG_DIR:-${OPENAI_LOGDIR}/png_samples}"
echo "Energy lambda:      ${ENERGY_LAMBDA}"
echo "Energy mode:        ${ENERGY_MODE}"
echo "Min-SNR gamma:      ${MIN_SNR_GAMMA}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" "${CMD[@]:1}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${CMD[@]}"
fi
