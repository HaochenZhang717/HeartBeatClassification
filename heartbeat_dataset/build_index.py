"""Scan processed_data/ and build a flat per-beat index.

Run once after process_ecg.py has produced the per-record .npz files:

    python -m heartbeat_dataset.build_index \
        --processed-root processed_data \
        --out heartbeat_dataset/beats_index.npz \
        --datasets mitdb svdb edb stdb

Output (single .npz file):
    record_path  : <U256, absolute path to the per-record .npz
    dataset      : <U16, dataset name (mitdb / svdb / ...)
    record_name  : <U16, record id (also serves as patient id for these dbs)
    sample_idx   : int64, R-peak sample index in the resampled (250 Hz) signal
    label_str    : <U1, AAMI symbol ('N','S','V','F','Q')
    label_int    : int8, AAMI class id (see AAMI_CLASSES order)

afdb is excluded by default — its beats are not AAMI-annotated, all labels were
defaulted to 'N' in process_ecg.py, which would heavily skew training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

AAMI_CLASSES = ("N", "S", "V", "F", "Q")
AAMI_TO_INT = {c: i for i, c in enumerate(AAMI_CLASSES)}

DEFAULT_DATASETS = ("mitdb", "svdb", "edb", "stdb", "nsrdb")


def scan_dataset(processed_root: Path, dataset: str) -> list[dict]:
    """Return one row per beat for every record in `dataset`."""
    rows: list[dict] = []
    ds_dir = processed_root / dataset
    if not ds_dir.exists():
        print(f"  [warn] {ds_dir} does not exist — skipping")
        return rows
    for npz_path in sorted(ds_dir.glob("*.npz")):
        d = np.load(npz_path, allow_pickle=True)
        beat_idx = d["beat_sample_idx"]
        labels = d["beat_aami_label"]
        if len(beat_idx) == 0:
            continue
        record_name = npz_path.stem
        for s, lab in zip(beat_idx.tolist(), labels.tolist()):
            if lab not in AAMI_TO_INT:
                continue
            rows.append({
                "record_path": str(npz_path.resolve()),
                "dataset": dataset,
                "record_name": record_name,
                "sample_idx": int(s),
                "label_str": str(lab),
                "label_int": AAMI_TO_INT[lab],
            })
        print(f"  [ok] {dataset}/{record_name}: {len(beat_idx)} beats")
    return rows


def build_index(processed_root: Path, datasets: list[str]) -> dict[str, np.ndarray]:
    all_rows: list[dict] = []
    for ds in datasets:
        print(f"\n=== Scanning {ds} ===")
        all_rows.extend(scan_dataset(processed_root, ds))

    n = len(all_rows)
    print(f"\nTotal beats indexed: {n}")
    if n == 0:
        raise RuntimeError("No beats indexed — did you run process_ecg.py first?")

    out = {
        "record_path": np.array([r["record_path"] for r in all_rows], dtype="<U256"),
        "dataset":     np.array([r["dataset"]     for r in all_rows], dtype="<U16"),
        "record_name": np.array([r["record_name"] for r in all_rows], dtype="<U16"),
        "sample_idx":  np.array([r["sample_idx"]  for r in all_rows], dtype=np.int64),
        "label_str":   np.array([r["label_str"]   for r in all_rows], dtype="<U1"),
        "label_int":   np.array([r["label_int"]   for r in all_rows], dtype=np.int8),
    }

    # Class histogram for sanity.
    print("\nClass histogram:")
    for c, ci in AAMI_TO_INT.items():
        cnt = int((out["label_int"] == ci).sum())
        print(f"  {c} ({ci}): {cnt}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", type=Path,
                    default=Path("processed_data"),
                    help="Directory containing per-dataset subfolders of .npz files")
    ap.add_argument("--out", type=Path,
                    default=Path("heartbeat_dataset/beats_index.npz"),
                    help="Output path for the beat index .npz")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                    help=f"Datasets to include (default: {DEFAULT_DATASETS})")
    args = ap.parse_args()

    idx = build_index(args.processed_root, args.datasets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **idx)
    print(f"\nWrote index → {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
