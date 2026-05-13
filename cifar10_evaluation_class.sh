#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# CIFAR-10 real image directory. Expected filenames start with class names,
# for example: dog_24495.png, plane_06858.png.
REAL_DIR="${REAL_DIR:-/home/shakthip/improved-diffusion/datasets/cifar_train}"

# Generated output directory. This can be either:
#   1. the sampling log directory containing png_samples/, or
#   2. the image directory itself.
GENERATED_DIR="${1:-${GENERATED_DIR:-${REPO_DIR}/logs/sample_cifar10_class_conditional_0.3_ema_0.9999_800000}}"

IMAGE_SIZE="${IMAGE_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SELECTION_SEED="${SELECTION_SEED:-}"

if [[ -d "${GENERATED_DIR}/png_samples" ]]; then
  GEN_IMAGE_DIR="${GENERATED_DIR}/png_samples"
else
  GEN_IMAGE_DIR="${GENERATED_DIR}"
fi

METRICS_CSV="${METRICS_CSV:-${GENERATED_DIR}/metrics_class_balanced.csv}"

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
BALANCED_REAL_DIR="$(mktemp -d)"
CLASS_COUNTS_JSON="${BALANCED_REAL_DIR}/class_counts.json"

cleanup() {
  rm -f "${LOG_FILE}"
  rm -rf "${BALANCED_REAL_DIR}"
}
trap cleanup EXIT

echo "Repo dir:                ${REPO_DIR}"
echo "Python:                  ${PYTHON_BIN}"
echo "Real dir:                ${REAL_DIR}"
echo "Generated dir:           ${GENERATED_DIR}"
echo "Generated image dir:     ${GEN_IMAGE_DIR}"
echo "Balanced real temp dir:  ${BALANCED_REAL_DIR}"
echo "Metrics CSV:             ${METRICS_CSV}"
echo "Visible GPUs:            ${CUDA_VISIBLE_DEVICES}"
echo "Image size:              ${IMAGE_SIZE}"
echo "Batch size:              ${BATCH_SIZE}"
echo "Selection seed:          ${SELECTION_SEED:-<sorted deterministic>}"

"${PYTHON_BIN}" - "${REAL_DIR}" "${GEN_IMAGE_DIR}" "${BALANCED_REAL_DIR}" "${CLASS_COUNTS_JSON}" "${SELECTION_SEED}" <<'PY'
import json
import os
import random
import sys
from collections import Counter, defaultdict

real_dir, gen_dir, balanced_real_dir, counts_json, seed = sys.argv[1:]

classes = ("bird", "car", "cat", "deer", "dog", "frog", "horse", "plane", "ship", "truck")
class_set = set(classes)
valid_exts = {".jpg", ".jpeg", ".png"}

def image_files(path):
    return [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if os.path.splitext(name)[1].lower() in valid_exts
    ]

def class_name(path):
    prefix = os.path.basename(path).split("_", 1)[0]
    return prefix if prefix in class_set else None

real_by_class = defaultdict(list)
for path in image_files(real_dir):
    cls = class_name(path)
    if cls is not None:
        real_by_class[cls].append(path)

gen_counts = Counter()
unknown_gen = []
for path in image_files(gen_dir):
    cls = class_name(path)
    if cls is None:
        unknown_gen.append(os.path.basename(path))
    else:
        gen_counts[cls] += 1

if unknown_gen:
    preview = ", ".join(unknown_gen[:10])
    extra = "" if len(unknown_gen) <= 10 else f", ... ({len(unknown_gen)} total)"
    raise SystemExit(
        "Generated images must have CIFAR-10 class-name prefixes "
        f"({', '.join(classes)}). Unknown filenames: {preview}{extra}"
    )

if not gen_counts:
    raise SystemExit(f"No generated images found in {gen_dir}")

rng = random.Random(int(seed)) if seed else None
selected_counts = {}

for cls in classes:
    needed = gen_counts[cls]
    available = list(real_by_class[cls])
    if needed == 0:
        selected_counts[cls] = 0
        continue
    if len(available) < needed:
        raise SystemExit(
            f"Not enough real images for class {cls}: need {needed}, found {len(available)}"
        )
    if rng is not None:
        rng.shuffle(available)
    selected = available[:needed]
    selected_counts[cls] = len(selected)
    for index, src in enumerate(selected):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(balanced_real_dir, f"{cls}_{index:06d}{ext}")
        os.symlink(src, dst)

with open(counts_json, "w", encoding="utf-8") as f:
    json.dump(
        {
            "generated_class_counts": {cls: gen_counts[cls] for cls in classes},
            "selected_real_class_counts": selected_counts,
        },
        f,
        sort_keys=True,
    )

print("Class counts used for evaluation:")
for cls in classes:
    print(f"  {cls}: generated={gen_counts[cls]} real_selected={selected_counts[cls]}")
print(f"Selected {sum(selected_counts.values())} real images into {balanced_real_dir}")
PY

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" evaluation.py \
  --real_dir "${BALANCED_REAL_DIR}" \
  --gen_dir "${GEN_IMAGE_DIR}" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" | tee "${LOG_FILE}"

"${PYTHON_BIN}" - "${LOG_FILE}" "${METRICS_CSV}" "${REAL_DIR}" "${BALANCED_REAL_DIR}" "${GEN_IMAGE_DIR}" "${IMAGE_SIZE}" "${BATCH_SIZE}" "${CLASS_COUNTS_JSON}" <<'PY'
import csv
import glob
import json
import os
import re
import sys

log_path, out_csv, real_dir, balanced_real_dir, gen_dir, image_size, batch_size, counts_json = sys.argv[1:]

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
balanced_real_count = sum(
    1 for path in glob.glob(os.path.join(balanced_real_dir, "*"))
    if os.path.splitext(path)[1].lower() in valid_exts
)
gen_count = sum(
    1 for path in glob.glob(os.path.join(gen_dir, "*"))
    if os.path.splitext(path)[1].lower() in valid_exts
)

with open(counts_json, "r", encoding="utf-8") as f:
    class_counts = json.load(f)

row = {
    "real_dir": real_dir,
    "balanced_real_dir": balanced_real_dir,
    "generated_image_dir": gen_dir,
    "image_size": image_size,
    "batch_size": batch_size,
    "real_image_count": balanced_real_count,
    "generated_image_count": gen_count,
    "generated_class_counts": json.dumps(class_counts["generated_class_counts"], sort_keys=True),
    "selected_real_class_counts": json.dumps(class_counts["selected_real_class_counts"], sort_keys=True),
    **metrics,
}

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)

print(f"Saved metrics CSV: {out_csv}")
PY
