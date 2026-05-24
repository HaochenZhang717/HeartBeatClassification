"""Train the 1D-CNN heartbeat classifier.

Run from the project root (where heartbeat_dataset/ and heartbeat_classifier/
both live):

    python -m heartbeat_classifier.train \
        --index-path heartbeat_dataset/beats_index.npz \
        --out-dir runs/baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from heartbeat_classifier.data import build_datasets, make_balanced_sampler
from heartbeat_classifier.model import HeartbeatCNN
from heartbeat_classifier.utils import (
    AAMI_CLASSES,
    CSVLogger,
    FocalLoss,
    compute_metrics,
    pick_device,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-path", type=Path,
                    default=Path("heartbeat_dataset/beats_index.npz"))
    ap.add_argument("--classes", nargs="+", default=["N", "S", "V"],
                    help="AAMI classes to train on, e.g. --classes N S V")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/baseline"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="Focal loss focusing parameter")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8,
                    help="Early stopping patience on val macro-F1")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, n_classes):
    model.eval()
    total_loss, n = 0.0, 0
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(y.cpu().numpy())
    metrics = compute_metrics(np.concatenate(trues), np.concatenate(preds),
                              n_classes=n_classes)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def main():
    args = parse_args()
    classes = args.classes
    n_classes = len(classes)
    set_seed(args.seed)
    device = pick_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] device={device}  classes={classes}  out_dir={args.out_dir}")

    train_ds, val_ds, _ = build_datasets(
        index_path=args.index_path,
        classes=classes,
        val_frac=args.val_frac,
        seed=args.seed,
        return_torch=True,
    )
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")
    print(f"[data] train class counts: {train_ds.class_counts()}")
    print(f"[data] val   class counts: {val_ds.class_counts()}")

    sampler = make_balanced_sampler(train_ds)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    model = HeartbeatCNN(in_channels=2, n_classes=n_classes).to(device)
    loss_fn = FocalLoss(gamma=args.gamma, alpha=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    fieldnames = ["epoch", "lr", "train_loss", "val_loss",
                  "val_acc", "val_macro_f1"]
    fieldnames += [f"val_f1_{c}" for c in classes]

    best_f1 = -1.0
    epochs_since_best = 0

    with CSVLogger(args.out_dir / "metrics.csv", fieldnames) as logger:
        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_loss_sum, tr_n = 0.0, 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                tr_loss_sum += loss.item() * y.size(0)
                tr_n += y.size(0)
            train_loss = tr_loss_sum / max(tr_n, 1)

            val_m = evaluate(model, val_loader, device, loss_fn, n_classes)
            scheduler.step()

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_m["loss"],
                "val_acc": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
            }
            for c, f1 in zip(classes, val_m["per_class_f1"]):
                row[f"val_f1_{c}"] = f1
            logger.log(row)

            print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  "
                  f"val_loss={val_m['loss']:.4f}  "
                  f"val_acc={val_m['accuracy']:.4f}  "
                  f"val_macroF1={val_m['macro_f1']:.4f}")

            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_metrics": val_m,
                "args": vars(args),
            }
            torch.save(ckpt, args.out_dir / "last.pt")
            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                epochs_since_best = 0
                torch.save(ckpt, args.out_dir / "best.pt")
                print(f"  ↑ new best val macro-F1 = {best_f1:.4f}")
            else:
                epochs_since_best += 1
                if epochs_since_best >= args.patience:
                    print(f"  early stop after {epochs_since_best} epochs "
                          f"without improvement")
                    break

    _plot_curves(args.out_dir / "metrics.csv", args.out_dir, classes)
    print(f"[done] best val macro-F1 = {best_f1:.4f}")


def _plot_curves(csv_path: Path, out_dir: Path, classes: list[str]) -> None:
    import csv as _csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(_csv.DictReader(csv_path.open()))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    axes[0].plot(ep, [float(r["val_loss"]) for r in rows], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(ep, [float(r["val_acc"]) for r in rows])
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Val accuracy")
    axes[2].plot(ep, [float(r["val_macro_f1"]) for r in rows])
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("macro F1")
    axes[2].set_title("Val macro-F1")
    fig.savefig(out_dir / "curves.png", dpi=120, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(7, 4))
    for c in classes:
        ax.plot(ep, [float(r[f"val_f1_{c}"]) for r in rows], label=c)
    ax.set_xlabel("epoch"); ax.set_ylabel("F1"); ax.legend()
    ax.set_title("Per-class val F1")
    fig2.savefig(out_dir / "per_class_f1.png", dpi=120, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
