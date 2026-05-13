#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# CIFAR-10 real image directory.
REAL_DIR="${REAL_DIR:-/home/shakthip/improved-diffusion/datasets/cifar_train}"

# Generated output directory. This can be either:
#   1. the sampling log directory containing png_samples/, or
#   2. the image directory itself.
GENERATED_DIR="${1:-${GENERATED_DIR:-${REPO_DIR}/logs/sample_cifar10_0.3_5.0_1ema_0.9999_800000}}"

IMAGE_SIZE="${IMAGE_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -d "${GENERATED_DIR}/png_samples" ]]; then
  GEN_IMAGE_DIR="${GENERATED_DIR}/png_samples"
else
  GEN_IMAGE_DIR="${GENERATED_DIR}"
fi

METRICS_CSV="${METRICS_CSV:-${GENERATED_DIR}/metrics.csv}"

if [[ ! -d "${REAL_DIR}" ]]; then
  echo "Real image directory not found: ${REAL_DIR}"
  exit 1
fi

if [[ ! -d "${GEN_IMAGE_DIR}" ]]; then
  echo "Generated image directory not found: ${GEN_IMAGE_DIR}"
  echo "Set GENERATED_DIR to the sampling log dir or the directory with generated PNG/JPG images."
  exit 1
fi

mkdir -p "$(dirname "${METRICS_CSV}")"

LOG_FILE="$(mktemp)"
cleanup() {
  rm -f "${LOG_FILE}"
}
trap cleanup EXIT

echo "Repo dir:            ${REPO_DIR}"
echo "Python:              ${PYTHON_BIN}"
echo "Real dir:            ${REAL_DIR}"
echo "Generated dir:       ${GENERATED_DIR}"
echo "Generated image dir: ${GEN_IMAGE_DIR}"
echo "Metrics CSV:         ${METRICS_CSV}"
echo "Visible GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "Image size:          ${IMAGE_SIZE}"
echo "Batch size:          ${BATCH_SIZE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" evaluation.py \
  --real_dir "${REAL_DIR}" \
  --gen_dir "${GEN_IMAGE_DIR}" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" | tee "${LOG_FILE}"

"${PYTHON_BIN}" - "${LOG_FILE}" "${METRICS_CSV}" "${REAL_DIR}" "${GEN_IMAGE_DIR}" "${IMAGE_SIZE}" "${BATCH_SIZE}" <<'PY'
import csv
import glob
import os
import re
import sys

log_path, out_csv, real_dir, gen_dir, image_size, batch_size = sys.argv[1:]

metrics = {
    "fid": "",
    "inception_score_mean": "",
    "inception_score_std": "",
    "precision": "",
    "recall": "",
    "density": "",
    "coverage": "",
}

with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    for raw_line in f:
        line = raw_line.strip()
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", line)
        if line.startswith("FID:") and numbers:
            metrics["fid"] = numbers[0]
        elif line.startswith("Inception Score:") and numbers:
            metrics["inception_score_mean"] = numbers[0]
            if len(numbers) > 1:
                metrics["inception_score_std"] = numbers[1]
        elif line.startswith("Precision:") and numbers:
            metrics["precision"] = numbers[0]
        elif line.startswith("Recall:") and numbers:
            metrics["recall"] = numbers[0]
        elif line.startswith("Density:") and numbers:
            metrics["density"] = numbers[0]
        elif line.startswith("Coverage:") and numbers:
            metrics["coverage"] = numbers[0]

missing = [key for key, value in metrics.items() if value == ""]
if missing:
    raise SystemExit(f"Could not parse metrics from evaluation output: {', '.join(missing)}")

valid_exts = {".jpg", ".jpeg", ".png"}
real_count = sum(
    1 for path in glob.glob(os.path.join(real_dir, "*"))
    if os.path.splitext(path)[1].lower() in valid_exts
)
gen_count = sum(
    1 for path in glob.glob(os.path.join(gen_dir, "*"))
    if os.path.splitext(path)[1].lower() in valid_exts
)

row = {
    "real_dir": real_dir,
    "generated_image_dir": gen_dir,
    "image_size": image_size,
    "batch_size": batch_size,
    "real_image_count": real_count,
    "generated_image_count": gen_count,
    **metrics,
}

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)

print(f"Saved metrics CSV: {out_csv}")
PY
