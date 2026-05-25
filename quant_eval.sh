#!/usr/bin/env bash
# 一键评估 fp32 / 8-bit / 4-bit 权重量化精度
# 用法:  bash quant_eval.sh
#
# 可通过环境变量覆盖:
#   CONDA_ENV   conda env (default: vlm)
#   CKPT        checkpoint (default: runs/nsv_baseline/best.pt)
#   INDEX       beats index (default: heartbeat_dataset/beats_index.npz)
#   OUT         JSONL 输出路径 (default: runs/nsv_baseline/compressed/quant_results.jsonl)
#   BITS        要评估的 bit 宽 (default: "8 4")
#   BATCH_SIZE  (default: 512)
#   NUM_WORKERS (default: 4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-vlm}"
CKPT="${CKPT:-runs/nsv_baseline/best.pt}"
INDEX="${INDEX:-heartbeat_dataset/beats_index.npz}"
OUT="${OUT:-runs/nsv_baseline/compressed/quant_results.jsonl}"
BITS="${BITS:-8 4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ ! -f "$CKPT" ]]; then
    echo "[error] checkpoint not found: $CKPT" >&2
    echo "        先跑 train.sh 拿到 best.pt" >&2
    exit 1
fi

# --- 激活 conda env ------------------------------------------------------- #
CONDA_BASE="$(conda info --base 2>/dev/null || echo /playpen-shared/haochenz/miniconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$OUT")"

echo "=========================================="
echo " HeartbeatCNN 量化精度评估 (fp32 + ${BITS}-bit)"
echo " ckpt    : $CKPT"
echo " index   : $INDEX"
echo " out     : $OUT"
echo " started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 把所有参数通过环境变量传给 Python，避免 heredoc 引号嵌套
export QE_CKPT="$CKPT" QE_INDEX="$INDEX" QE_OUT="$OUT" QE_BITS="$BITS"
export QE_BATCH_SIZE="$BATCH_SIZE" QE_NUM_WORKERS="$NUM_WORKERS"

python - <<'PY'
"""
评估 HeartbeatCNN 在 DS2 测试集上的精度:
  - fp32 基线
  - 每个 BITS 列出的位宽,做 per-channel 对称权重 fake-quantization
    (量化->dequant 回 fp32 推理,只反映精度损失)

每个 config 一行写入 JSONL,包含:
  config / bits / classes
  accuracy / macro_f1 / macro_precision / macro_recall
  per_class_accuracy (= recall, 每类 TP/(TP+FN))
  per_class_precision / per_class_recall / per_class_f1 / per_class_support
  confusion_matrix
"""
import copy, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from heartbeat_classifier.data import build_datasets
from heartbeat_classifier.model import HeartbeatCNN
from heartbeat_classifier.utils import compute_metrics, pick_device


ckpt_path   = Path(os.environ["QE_CKPT"])
index_path  = Path(os.environ["QE_INDEX"])
out_path    = Path(os.environ["QE_OUT"])
bits_list   = [int(b) for b in os.environ["QE_BITS"].split()]
batch_size  = int(os.environ["QE_BATCH_SIZE"])
num_workers = int(os.environ["QE_NUM_WORKERS"])

device = pick_device()
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
classes = ckpt["args"].get("classes", ["N", "S", "V"])
n_classes = len(classes)
print(f"[setup] classes={classes}  device={device}")

fp32 = HeartbeatCNN(in_channels=2, n_classes=n_classes)
fp32.load_state_dict(ckpt["state_dict"])
fp32.eval()

_, _, test_ds = build_datasets(
    index_path=index_path, classes=classes, return_torch=True, preload=True,
)
loader_kw = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"))
if num_workers > 0:
    loader_kw["persistent_workers"] = True
    loader_kw["prefetch_factor"] = 4
test_ldr = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kw)
print(f"[data]  DS2 test: {len(test_ds)} beats  class_counts={test_ds.class_counts()}")


# ── per-channel 对称 fake-quant ─────────────────────────────────────────────
@torch.no_grad()
def fake_quantize(w: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1                   # 8-bit: 127, 4-bit: 7
    flat = w.reshape(w.shape[0], -1)
    amax = flat.abs().amax(dim=1).clamp(min=1e-12)
    scale = (amax / qmax).view([-1] + [1] * (w.dim() - 1))
    q = torch.round(w / scale).clamp(-qmax - 1, qmax)
    return (q * scale).to(w.dtype)


def quantize_weights(src: nn.Module, bits: int) -> nn.Module:
    m = copy.deepcopy(src)
    for mod in m.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            mod.weight.data.copy_(fake_quantize(mod.weight.data, bits))
    return m


# ── 评估 ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module) -> dict:
    model.to(device).eval()
    preds, trues = [], []
    for x, y in test_ldr:
        logits = model(x.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    m = compute_metrics(y_true, y_pred, n_classes=n_classes)
    labels = list(range(n_classes))
    p_macro, r_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = np.asarray(m["confusion_matrix"], dtype=np.int64)
    row_sums = cm.sum(axis=1)
    per_class_acc = np.where(row_sums > 0, np.diag(cm) / np.maximum(row_sums, 1), 0.0)
    m["macro_precision"] = float(p_macro)
    m["macro_recall"] = float(r_macro)
    m["per_class_accuracy"] = per_class_acc.astype(float).tolist()
    return m


# ── 跑全套 + 写 JSONL ────────────────────────────────────────────────────────
configs = [("fp32", None, fp32)]
for b in bits_list:
    configs.append((f"int{b}_weight_only", b, quantize_weights(fp32, b)))

print(f"\n[eval] writing → {out_path}")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as fh:
    for tag, bits, model in configs:
        print(f"\n── {tag} ──")
        m = evaluate(model)
        row = {
            "config": tag,
            "bits": bits,
            "classes": classes,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "per_class_accuracy": m["per_class_accuracy"],
            "per_class_precision": m["per_class_precision"],
            "per_class_recall": m["per_class_recall"],
            "per_class_f1": m["per_class_f1"],
            "per_class_support": m["per_class_support"],
            "confusion_matrix": m["confusion_matrix"],
            "ckpt": str(ckpt_path),
        }
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(f"  acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  "
              f"macroP={m['macro_precision']:.4f}  macroR={m['macro_recall']:.4f}")
        for c, a, p, r, f1, supp in zip(
            classes, m["per_class_accuracy"], m["per_class_precision"],
            m["per_class_recall"], m["per_class_f1"], m["per_class_support"],
        ):
            print(f"  {c}: acc={a:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  n={supp}")

print(f"\n[done] {len(configs)} configs → {out_path}")
PY

echo ""
echo "All done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results → $OUT"
