"""Evaluate a trained checkpoint on the mitdb DS2 held-out test set.

Usage:
    python -m heartbeat_classifier.evaluate --ckpt runs/baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from heartbeat_classifier.data import build_datasets
from heartbeat_classifier.model import HeartbeatCNN
from heartbeat_classifier.utils import (
    AAMI_CLASSES,
    compute_metrics,
    pick_device,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-path", type=Path,
                    default=Path("heartbeat_dataset/beats_index.npz"))
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to best.pt or last.pt")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write test_metrics.json + confusion matrix "
                         "png; defaults to the checkpoint's directory")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--preload", action="store_true",
                    help="Materialize all DS2 windows into RAM (~0.1 GB).")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = args.out_dir or args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()

    # Read classes from the checkpoint so evaluate always matches training.
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["args"].get("classes", ["N", "S", "V", "F", "Q"])
    n_classes = len(classes)
    print(f"[setup] classes={classes} (loaded from checkpoint)")

    _, _, test_ds = build_datasets(
        index_path=args.index_path, classes=classes, return_torch=True,
        preload=args.preload)
    print(f"[data] DS2 test set: {len(test_ds)} beats")
    print(f"[data] class counts: {test_ds.class_counts()}")

    loader_kw: dict = dict(num_workers=args.num_workers,
                           pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 4
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        **loader_kw)
    model = HeartbeatCNN(in_channels=2, n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(y.numpy())
    metrics = compute_metrics(np.concatenate(trues), np.concatenate(preds),
                              n_classes=n_classes)

    print(f"\n[test] accuracy = {metrics['accuracy']:.4f}")
    print(f"[test] macro F1 = {metrics['macro_f1']:.4f}")
    print("[test] per-class:")
    for c, p, r, f1, supp in zip(classes,
                                  metrics["per_class_precision"],
                                  metrics["per_class_recall"],
                                  metrics["per_class_f1"],
                                  metrics["per_class_support"]):
        print(f"  {c}: P={p:.4f}  R={r:.4f}  F1={f1:.4f}  n={supp}")

    with (out_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    _plot_confusion(metrics["confusion_matrix"], out_dir / "confusion_matrix.png",
                    classes)
    print(f"\n[done] wrote {out_dir}/test_metrics.json and confusion_matrix.png")


def _plot_confusion(cm: list[list[int]], out_path: Path,
                    classes: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("DS2 confusion matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
