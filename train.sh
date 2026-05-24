#!/usr/bin/env bash
# Run from workspace/:  bash train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INDEX="heartbeat_dataset/beats_index.npz"
OUT_DIR="runs/nsv_baseline"

echo "=========================================="
echo " 1D-CNN  N/S/V 3-class training"
echo " out_dir : $OUT_DIR"
echo " started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

python -m heartbeat_classifier.train \
    --index-path  "$INDEX"      \
    --classes     N S V         \
    --out-dir     "$OUT_DIR"    \
    --epochs      30            \
    --batch-size  128           \
    --lr          1e-3          \
    --weight-decay 1e-4         \
    --gamma       2.0           \
    --val-frac    0.1           \
    --patience    8             \
    --seed        0             \
    --num-workers 0

echo ""
echo "=========================================="
echo " Training done. Running DS2 evaluation..."
echo "=========================================="

python -m heartbeat_classifier.evaluate \
    --index-path  "$INDEX"              \
    --ckpt        "$OUT_DIR/best.pt"    \
    --batch-size  256                   \
    --num-workers 0

echo ""
echo "All done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results → $OUT_DIR/"
