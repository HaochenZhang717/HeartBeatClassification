"""Model compression pipeline for HeartbeatCNN.

Evaluates five configurations on the MIT-BIH DS2 test set (all on CPU):
  fp32           — original float32 model (baseline)
  dyn_int8       — dynamic INT8 (Linear layer only)
  static_int8    — static INT8 (Conv1d + Linear, fused Conv-BN-ReLU)
  pruned30_fp32  — 30% L1 structured pruning of Conv1d filters, fp32
  pruned30_int8  — pruned30 + static INT8

Metrics: Accuracy, Macro-F1, per-class F1/Precision/Recall, model size (KB),
         single-sample CPU inference time (ms/beat).

Usage:
    python -m heartbeat_classifier.compress \\
        --ckpt runs/nsv_baseline/best.pt \\
        --index-path heartbeat_dataset/beats_index.npz \\
        --out-dir runs/nsv_baseline/compressed
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune_utils
from torch.utils.data import DataLoader

from heartbeat_classifier.data import build_datasets
from heartbeat_classifier.model import HeartbeatCNN
from heartbeat_classifier.utils import compute_metrics


# ── quantizable wrapper ───────────────────────────────────────────────────────

class _QHeartbeatCNN(nn.Module):
    """HeartbeatCNN with QuantStub/DeQuantStub for static PTQ.

    The input transpose happens *before* QuantStub to avoid any
    potential issues with quantized-tensor reshape operations.
    """

    def __init__(self, src: HeartbeatCNN):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.features = copy.deepcopy(src.features)
        self.gap = copy.deepcopy(src.gap)
        self.head = copy.deepcopy(src.head)
        # In-place ReLU must be replaced for static quantization
        for block in self.features:
            if hasattr(block, "relu"):
                block.relu = nn.ReLU(inplace=False)

    def fuse(self) -> "_QHeartbeatCNN":
        """Fuse Conv1d + BatchNorm1d + ReLU in every ConvBlock."""
        for block in self.features:
            torch.quantization.fuse_modules(
                block, ["conv", "bn", "relu"], inplace=True
            )
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)   # (B,T,C) → (B,C,T) before quantization
        x = self.quant(x)
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        x = self.head(x)
        return self.dequant(x)


# ── utilities ─────────────────────────────────────────────────────────────────

def _file_size_kb(model: nn.Module) -> float:
    """Serialize state_dict to a temp file and return size in KB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024.0
    os.unlink(path)
    return size


@torch.no_grad()
def _bench_ms(model: nn.Module, window: int = 250, leads: int = 2,
               n_runs: int = 1000, warmup: int = 100) -> float:
    """Mean single-sample CPU inference time in ms/beat."""
    model = model.cpu().eval()
    x = torch.randn(1, window, leads)
    for _ in range(warmup):
        model(x)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    return (time.perf_counter() - t0) / n_runs * 1000.0


@torch.no_grad()
def _eval_cpu(model: nn.Module, loader: DataLoader, n_classes: int) -> dict:
    model.cpu().eval()
    preds, trues = [], []
    for x, y in loader:
        logits = model(x.cpu())
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return compute_metrics(
        np.concatenate(trues), np.concatenate(preds), n_classes=n_classes
    )


def _static_ptq(
    src: HeartbeatCNN,
    cal_loader: DataLoader,
    n_cal_batches: int,
    backend: str,
) -> nn.Module:
    """Build a static INT8 quantized model via calibration."""
    torch.backends.quantized.engine = backend
    qm = _QHeartbeatCNN(src).cpu().eval()
    qm.fuse()
    qm.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(qm, inplace=True)
    with torch.no_grad():
        for i, (x, _) in enumerate(cal_loader):
            if i >= n_cal_batches:
                break
            qm(x.cpu())
    torch.quantization.convert(qm, inplace=True)
    return qm


def _structured_prune(src: HeartbeatCNN, amount: float) -> HeartbeatCNN:
    """L1 structured pruning on all Conv1d output channels (permanent)."""
    m = copy.deepcopy(src)
    for module in m.modules():
        if isinstance(module, nn.Conv1d):
            prune_utils.ln_structured(
                module, name="weight", amount=amount, n=1, dim=0
            )
            prune_utils.remove(module, "weight")
    return m


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="HeartbeatCNN compression: PTQ + structured pruning"
    )
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to best.pt or last.pt")
    ap.add_argument("--index-path", type=Path,
                    default=Path("heartbeat_dataset/beats_index.npz"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: <ckpt_dir>/compressed)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--cal-batches", type=int, default=50,
                    help="Val batches used for static PTQ calibration")
    ap.add_argument("--prune-amount", type=float, default=0.30,
                    help="Fraction of Conv1d output filters to prune (L1)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--qbackend", type=str, default="qnnpack",
                    choices=["qnnpack", "fbgemm"],
                    help="Quantization backend: qnnpack (ARM/Apple) or fbgemm (x86)")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.ckpt.parent / "compressed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["args"].get("classes", ["N", "S", "V"])
    n_classes = len(classes)
    print(f"[setup] classes={classes}  backend={args.qbackend}")
    print(f"[setup] out_dir={out_dir}")

    fp32 = HeartbeatCNN(in_channels=2, n_classes=n_classes)
    fp32.load_state_dict(ckpt["state_dict"])
    fp32.eval()

    # ── datasets ─────────────────────────────────────────────────────────────
    _, val_ds, test_ds = build_datasets(
        index_path=args.index_path, classes=classes, return_torch=True
    )
    test_ldr = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    cal_ldr = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"[data]  test={len(test_ds)}  val(cal)={len(val_ds)}")

    # ── evaluate one config ───────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    def run(tag: str, model: nn.Module):
        print(f"\n── {tag} {'─'*(50 - len(tag))}")
        m = _eval_cpu(model, test_ldr, n_classes)
        size_kb = _file_size_kb(model)
        t_ms = _bench_ms(model)
        all_results[tag] = {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "per_class_f1": m["per_class_f1"],
            "per_class_precision": m["per_class_precision"],
            "per_class_recall": m["per_class_recall"],
            "per_class_support": m["per_class_support"],
            "confusion_matrix": m["confusion_matrix"],
            "size_kb": size_kb,
            "inference_ms": t_ms,
        }
        print(f"  acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}")
        for c, p, r, f1 in zip(classes, m["per_class_precision"],
                                m["per_class_recall"], m["per_class_f1"]):
            print(f"  {c}: P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
        print(f"  size={size_kb:.1f} KB   inference={t_ms:.3f} ms/beat")

    # ── 1. float32 baseline ───────────────────────────────────────────────────
    run("fp32", fp32)

    # ── 2. dynamic INT8 (Linear only) ────────────────────────────────────────
    dyn = torch.quantization.quantize_dynamic(
        copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8
    )
    run("dyn_int8", dyn)
    torch.save(dyn.state_dict(), out_dir / "dyn_int8.pt")

    # ── 3. static INT8 (Conv1d + Linear) ─────────────────────────────────────
    try:
        static = _static_ptq(fp32, cal_ldr, args.cal_batches, args.qbackend)
        run("static_int8", static)
        torch.save(static, out_dir / "static_int8_full.pt")
    except Exception as exc:
        print(f"[warn] static PTQ failed: {exc}")
        all_results["static_int8"] = {"error": str(exc)}

    # ── 4. structured pruning 30%, fp32 ──────────────────────────────────────
    pct = int(args.prune_amount * 100)
    pruned = _structured_prune(fp32, args.prune_amount)
    run(f"pruned{pct}_fp32", pruned)
    torch.save(pruned.state_dict(), out_dir / f"pruned{pct}_fp32.pt")

    # ── 5. pruned + static INT8 ───────────────────────────────────────────────
    try:
        pruned_int8 = _static_ptq(pruned, cal_ldr, args.cal_batches, args.qbackend)
        run(f"pruned{pct}_int8", pruned_int8)
        torch.save(pruned_int8, out_dir / f"pruned{pct}_int8_full.pt")
    except Exception as exc:
        print(f"[warn] pruned + PTQ failed: {exc}")
        all_results[f"pruned{pct}_int8"] = {"error": str(exc)}

    # ── summary table ─────────────────────────────────────────────────────────
    _print_table(all_results, classes)

    # ── persist ───────────────────────────────────────────────────────────────
    with open(out_dir / "compression_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    _plot(all_results, classes, out_dir)
    print(f"\n[done] results → {out_dir}/")


# ── reporting ─────────────────────────────────────────────────────────────────

def _print_table(results: dict, classes: list[str]):
    sep = "═" * 82
    print(f"\n{sep}")
    hdr = (f"{'Config':<20} {'Acc':>6} {'MacroF1':>8} "
           + " ".join(f"{c+'-F1':>8}" for c in classes)
           + f"  {'KB':>7}  {'ms/beat':>8}")
    print(hdr)
    print("─" * 82)
    for tag, r in results.items():
        if "error" in r:
            print(f"  {tag:<18}  ERROR: {r['error'][:55]}")
            continue
        f1s = " ".join(f"{f:8.4f}" for f in r["per_class_f1"])
        print(f"  {tag:<18} {r['accuracy']:6.4f} {r['macro_f1']:8.4f} "
              f"{f1s}  {r['size_kb']:7.1f}  {r['inference_ms']:8.3f}")
    print(sep)


def _plot(results: dict, classes: list[str], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return
    tags = list(valid.keys())
    colors = [plt.cm.tab10(i) for i in range(len(tags))]

    # ── bar chart: macro-F1 / size / inference time ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(tags))
    bar_kw = dict(edgecolor="gray", linewidth=0.4)

    def annotate(ax, vals, fmt="{:.2f}"):
        ymax = ax.get_ylim()[1]
        for i, v in enumerate(vals):
            ax.text(i, v + ymax * 0.01, fmt.format(v),
                    ha="center", va="bottom", fontsize=7)

    mf1 = [valid[t]["macro_f1"] * 100 for t in tags]
    axes[0].bar(x, mf1, color=colors, **bar_kw)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("Macro F1 (%)")
    axes[0].set_title("Classification Performance")
    axes[0].set_ylim(max(0, min(mf1) - 6), 105)
    annotate(axes[0], mf1)

    sizes = [valid[t]["size_kb"] for t in tags]
    axes[1].bar(x, sizes, color=colors, **bar_kw)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("Model Size (KB)")
    axes[1].set_title("Model Size (state_dict on disk)")
    annotate(axes[1], sizes, "{:.0f}")

    times = [valid[t]["inference_ms"] for t in tags]
    axes[2].bar(x, times, color=colors, **bar_kw)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    axes[2].set_ylabel("ms / beat")
    axes[2].set_title("CPU Inference Time (batch=1)")
    axes[2].axhline(100, color="red", linestyle="--", lw=1, alpha=0.7,
                    label="100 ms limit (Aim 2a)")
    axes[2].legend(fontsize=8)
    annotate(axes[2], times, "{:.3f}")

    fig.tight_layout()
    fig.savefig(out_dir / "compression_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── per-class F1 grouped bar ──────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 5))
    n = len(tags)
    w = 0.75 / n
    offsets = np.linspace(-(n - 1) * w / 2, (n - 1) * w / 2, n)
    for i, (tag, offs) in enumerate(zip(tags, offsets)):
        f1s = [v * 100 for v in valid[tag]["per_class_f1"]]
        ax.bar(np.arange(len(classes)) + offs, f1s, w,
               label=tag, color=colors[i], edgecolor="gray", linewidth=0.4)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Per-Class F1 Across Compression Configs")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 108)
    fig2.tight_layout()
    fig2.savefig(out_dir / "per_class_f1_compression.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plots] saved to {out_dir}/")


if __name__ == "__main__":
    main()
