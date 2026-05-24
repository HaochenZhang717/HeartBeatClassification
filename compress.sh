#!/usr/bin/env bash
# Run from workspace/:  bash compress.sh
# Requires a trained checkpoint at runs/nsv_baseline/best.pt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CKPT="runs/nsv_baseline/best.pt"
INDEX="heartbeat_dataset/beats_index.npz"
OUT_DIR="runs/nsv_baseline/compressed"

if [[ ! -f "$CKPT" ]]; then
    echo "[error] checkpoint not found: $CKPT"
    echo "        Run train.sh first."
    exit 1
fi

echo "=========================================="
echo " HeartbeatCNN compression pipeline"
echo " ckpt    : $CKPT"
echo " out_dir : $OUT_DIR"
echo " started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

python -m heartbeat_classifier.compress \
    --ckpt          "$CKPT"      \
    --index-path    "$INDEX"     \
    --out-dir       "$OUT_DIR"   \
    --batch-size    256          \
    --cal-batches   50           \
    --prune-amount  0.30         \
    --num-workers   0            \
    --qbackend      qnnpack

echo ""
echo "All done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results → $OUT_DIR/"
