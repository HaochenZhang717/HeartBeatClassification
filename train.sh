#!/usr/bin/env bash
# Run from workspace/:  bash train.sh
#
# Env overrides:
#   CONDA_ENV    conda env to activate (default: vlm)
#   DATA_ROOT    processed_data directory holding mitdb/ svdb/ edb/ stdb/ nsrdb/
#                (default: /mnt/unites8/playpen/haochenz/Time_Series_Datasets/heart_beat/processed_data)
#   INDEX        path to beats_index.npz (default: heartbeat_dataset/beats_index.npz)
#   REBUILD_INDEX=1   force rebuild even if INDEX exists and matches DATA_ROOT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-vlm}"
DATA_ROOT="${DATA_ROOT:-/mnt/unites8/playpen/haochenz/Time_Series_Datasets/heart_beat/processed_data}"
INDEX="${INDEX:-heartbeat_dataset/beats_index.npz}"
OUT_DIR="${OUT_DIR:-runs/nsv_baseline}"
# GPU 0 is often busy with another user on this box; default to GPU 1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PRELOAD_FLAG="${PRELOAD_FLAG:---preload}"   # set PRELOAD_FLAG="" to disable

# --- activate conda env ---------------------------------------------------- #
CONDA_BASE="$(conda info --base 2>/dev/null || echo /playpen-shared/haochenz/miniconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo " 1D-CNN  N/S/V 3-class training"
echo " conda env       : $CONDA_ENV   ($(which python))"
echo " CUDA_VISIBLE    : $CUDA_VISIBLE_DEVICES"
echo " data root       : $DATA_ROOT"
echo " index           : $INDEX"
echo " out_dir         : $OUT_DIR"
echo " batch / workers : $BATCH_SIZE / $NUM_WORKERS  preload=${PRELOAD_FLAG:-off}"
echo " started         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# --- sanity-check data + (re)build index if needed ------------------------- #
if [ ! -d "$DATA_ROOT" ]; then
    echo "[error] DATA_ROOT does not exist: $DATA_ROOT" >&2
    echo "        Set DATA_ROOT=/path/to/processed_data and rerun." >&2
    exit 1
fi
for ds in mitdb svdb edb stdb nsrdb; do
    if [ ! -d "$DATA_ROOT/$ds" ]; then
        echo "[warn] missing dataset subdir: $DATA_ROOT/$ds (will be skipped)" >&2
    fi
done

# Rebuild index if missing, or if its stored paths don't point under DATA_ROOT,
# or if the user forced REBUILD_INDEX=1.
need_rebuild=0
if [ ! -f "$INDEX" ]; then
    need_rebuild=1
elif [ "${REBUILD_INDEX:-0}" = "1" ]; then
    need_rebuild=1
else
    first_path="$(python - <<PY
import numpy as np
try:
    p = str(np.load("$INDEX", allow_pickle=False)["record_path"][0])
    print(p)
except Exception:
    print("")
PY
)"
    case "$first_path" in
        "$DATA_ROOT"/*) ;;                  # index already matches DATA_ROOT
        "") need_rebuild=1 ;;               # unreadable -> rebuild
        *) echo "[info] index paths point elsewhere ($first_path)"
           echo "       rebuilding for DATA_ROOT=$DATA_ROOT"
           need_rebuild=1 ;;
    esac
fi

if [ "$need_rebuild" = "1" ]; then
    echo "[info] building beats index from $DATA_ROOT ..."
    python -m heartbeat_dataset.build_index \
        --processed-root "$DATA_ROOT" \
        --out "$INDEX" \
        --datasets mitdb svdb edb stdb nsrdb
fi

# --- train ---------------------------------------------------------------- #
python -m heartbeat_classifier.train \
    --index-path   "$INDEX"          \
    --classes      N S V             \
    --out-dir      "$OUT_DIR"        \
    --epochs       30                \
    --batch-size   "$BATCH_SIZE"     \
    --lr           1e-3              \
    --weight-decay 1e-4              \
    --gamma        2.0               \
    --val-frac     0.1               \
    --patience     8                 \
    --seed         0                 \
    --num-workers  "$NUM_WORKERS"    \
    $PRELOAD_FLAG

echo ""
echo "=========================================="
echo " Training done. Running DS2 evaluation..."
echo "=========================================="

python -m heartbeat_classifier.evaluate \
    --index-path  "$INDEX"              \
    --ckpt        "$OUT_DIR/best.pt"    \
    --batch-size  512                   \
    --num-workers "$NUM_WORKERS"        \
    $PRELOAD_FLAG

echo ""
echo "All done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results → $OUT_DIR/"
