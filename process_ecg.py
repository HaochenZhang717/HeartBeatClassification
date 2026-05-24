"""Process PhysioNet ECG datasets (mitdb, svdb, nsrdb, edb, stdb, afdb) into a
unified NPZ format.

Output per record (one .npz file):
  signal              : float32 array, shape (n_samples, 2), resampled to 250 Hz
  fs                  : int, always 250
  beat_sample_idx     : int64 array of R-peak sample indices (in resampled signal)
  beat_aami_label     : <U1 array of AAMI 5-class labels ('N','S','V','F','Q')
  beat_wfdb_symbol    : <U2 array of original WFDB symbols
  rhythm_start        : int64 array of rhythm episode start sample indices
  rhythm_end          : int64 array of rhythm episode end sample indices
  rhythm_label        : object array of rhythm strings (e.g. 'N','AFIB','VT')
  meta                : dict (dataset, record_name, original_fs, duration_s,
                        leads, beat_ann_source)
"""

from __future__ import annotations

import os
from pathlib import Path
from fractions import Fraction

import numpy as np
import wfdb
from scipy.signal import resample_poly

RAW_ROOT = Path("/Users/zhc/Documents/Time_Series_Datasets/ECG_Datasets/data_source")
OUT_ROOT = Path("/Users/zhc/Documents/Time_Series_Datasets/ECG_Datasets/workspace/processed_data")
TARGET_FS = 250

# AAMI EC57 5-class mapping of WFDB beat symbols.
AAMI_MAP = {
    # N - Normal and bundle-branch blocks / escape from supra-ventricular origin
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S - Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V - Ventricular ectopic
    "V": "V", "E": "V",
    # F - Fusion of ventricular and normal
    "F": "F",
    # Q - Unknown / paced
    "/": "Q", "f": "Q", "Q": "Q", "?": "Q",
}
BEAT_SYMBOLS = set(AAMI_MAP.keys())

# Per-dataset config.
DATASETS = {
    "mitdb": {
        "ann_ext": "atr",
        "exclude_records": {"102", "104", "107", "217"},  # AAMI paced records
    },
    "svdb": {"ann_ext": "atr", "exclude_records": set()},
    "nsrdb": {"ann_ext": "atr", "exclude_records": set()},
    "edb":   {"ann_ext": "atr", "exclude_records": set()},
    "stdb":  {"ann_ext": "atr", "exclude_records": set()},
    "afdb":  {
        "ann_ext": "atr",  # rhythm
        "beat_ext_preferred": "qrsc",  # prefer corrected, fall back to qrs
        "beat_ext_fallback": "qrs",
        "exclude_records": {"00735", "03665"},  # missing .dat
    },
}


def resample_signal(sig: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return sig.astype(np.float32, copy=False)
    ratio = Fraction(fs_out, fs_in).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    out = resample_poly(sig, up=up, down=down, axis=0).astype(np.float32)
    return out


def map_indices(idx: np.ndarray, fs_in: int, fs_out: int, n_out: int) -> np.ndarray:
    """Rescale sample indices after resampling and clip to [0, n_out-1]."""
    if fs_in == fs_out:
        scaled = idx
    else:
        scaled = np.round(idx.astype(np.float64) * fs_out / fs_in).astype(np.int64)
    return np.clip(scaled, 0, n_out - 1)


def parse_rhythm_segments(samples: np.ndarray, symbols: list, aux_notes: list,
                          n_total: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Walk through annotations, build rhythm episodes from '+' markers.

    A '+' annotation with an aux_note like '(AFIB' opens a rhythm episode that
    runs until the next '+' marker (or end of record).
    """
    starts, ends, labels = [], [], []
    cur_label, cur_start = None, None
    for s, sym, aux in zip(samples, symbols, aux_notes):
        if sym != "+":
            continue
        aux_clean = (aux or "").strip().lstrip("(").rstrip("\x00").strip()
        if not aux_clean:
            continue
        # Close previous episode.
        if cur_label is not None:
            starts.append(cur_start)
            ends.append(int(s))
            labels.append(cur_label)
        cur_label = aux_clean
        cur_start = int(s)
    if cur_label is not None:
        starts.append(cur_start)
        ends.append(n_total)
        labels.append(cur_label)
    return (np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
            labels)


def extract_beats(samples: np.ndarray, symbols: list) -> tuple[np.ndarray, list, list]:
    """Keep only annotations whose symbol is a known WFDB beat symbol."""
    keep = [i for i, sym in enumerate(symbols) if sym in BEAT_SYMBOLS]
    if not keep:
        return (np.empty(0, dtype=np.int64), [], [])
    keep_idx = np.asarray(keep)
    return (samples[keep_idx],
            [symbols[i] for i in keep],
            [AAMI_MAP[symbols[i]] for i in keep])


def read_records_list(dataset: str) -> list[str]:
    rec_file = RAW_ROOT / dataset / "1.0.0" / "RECORDS"
    return [ln.strip() for ln in rec_file.read_text().splitlines() if ln.strip()]


def process_record(dataset: str, rec_name: str, cfg: dict) -> dict | None:
    base = RAW_ROOT / dataset / "1.0.0" / rec_name
    dat_path = base.with_suffix(".dat")
    if not dat_path.exists():
        print(f"  [skip] {dataset}/{rec_name}: missing .dat")
        return None

    rec = wfdb.rdrecord(str(base))
    sig = rec.p_signal  # (n, n_leads), float
    fs_in = int(rec.fs)
    if sig.shape[1] < 2:
        print(f"  [skip] {dataset}/{rec_name}: <2 leads")
        return None
    sig = sig[:, :2]  # always keep first 2 leads
    leads = rec.sig_name[:2]

    # Resample to target.
    sig_rs = resample_signal(sig, fs_in, TARGET_FS)
    n_out = sig_rs.shape[0]

    # Read rhythm annotations (always from .atr).
    atr_path = base.with_suffix(".atr")
    if atr_path.exists():
        ann = wfdb.rdann(str(base), "atr")
        rhythm_start, rhythm_end, rhythm_label = parse_rhythm_segments(
            ann.sample, ann.symbol, ann.aux_note, n_total=rec.sig_len)
        rhythm_start = map_indices(rhythm_start, fs_in, TARGET_FS, n_out)
        rhythm_end = map_indices(rhythm_end, fs_in, TARGET_FS, n_out)
    else:
        ann = None
        rhythm_start = np.empty(0, dtype=np.int64)
        rhythm_end = np.empty(0, dtype=np.int64)
        rhythm_label = []

    # Read beat annotations.
    if dataset == "afdb":
        # Prefer .qrsc, fall back to .qrs. Symbols are not real AAMI classes
        # (afdb does not annotate ectopics); default all to 'N'.
        beat_ext = cfg.get("beat_ext_preferred")
        if not base.with_suffix(f".{beat_ext}").exists():
            beat_ext = cfg.get("beat_ext_fallback")
        beat_ann = wfdb.rdann(str(base), beat_ext)
        beat_samples = beat_ann.sample
        beat_syms = list(beat_ann.symbol)
        # afdb beat-position annotation: all default to 'N' (no AAMI info available)
        aami_labels = ["N"] * len(beat_samples)
        beat_source = beat_ext
    else:
        if ann is None:
            beat_samples = np.empty(0, dtype=np.int64)
            beat_syms = []
            aami_labels = []
        else:
            beat_samples, beat_syms, aami_labels = extract_beats(ann.sample, ann.symbol)
        beat_source = "atr"

    beat_samples = map_indices(np.asarray(beat_samples, dtype=np.int64),
                               fs_in, TARGET_FS, n_out)

    meta = {
        "dataset": dataset,
        "record_name": rec_name,
        "original_fs": fs_in,
        "duration_s": float(n_out) / TARGET_FS,
        "leads": list(leads),
        "beat_ann_source": beat_source,
    }

    return {
        "signal": sig_rs,
        "fs": np.int32(TARGET_FS),
        "beat_sample_idx": beat_samples,
        "beat_aami_label": np.asarray(aami_labels, dtype="<U1"),
        "beat_wfdb_symbol": np.asarray(beat_syms, dtype="<U2"),
        "rhythm_start": rhythm_start,
        "rhythm_end": rhythm_end,
        "rhythm_label": np.asarray(rhythm_label, dtype=object),
        "meta": meta,
    }


def save_npz(out_dir: Path, rec_name: str, data: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec_name}.npz"
    np.savez_compressed(
        out_path,
        signal=data["signal"],
        fs=data["fs"],
        beat_sample_idx=data["beat_sample_idx"],
        beat_aami_label=data["beat_aami_label"],
        beat_wfdb_symbol=data["beat_wfdb_symbol"],
        rhythm_start=data["rhythm_start"],
        rhythm_end=data["rhythm_end"],
        rhythm_label=data["rhythm_label"],
        meta=np.array(data["meta"], dtype=object),
    )


def process_dataset(dataset: str, cfg: dict) -> dict:
    print(f"\n=== Processing {dataset} ===")
    out_dir = OUT_ROOT / dataset
    records = read_records_list(dataset)
    excluded = cfg["exclude_records"]

    n_done = n_skipped = n_failed = 0
    for rec_name in records:
        if rec_name in excluded:
            print(f"  [exclude] {rec_name}")
            n_skipped += 1
            continue
        try:
            data = process_record(dataset, rec_name, cfg)
        except Exception as e:
            print(f"  [fail] {rec_name}: {e}")
            n_failed += 1
            continue
        if data is None:
            n_skipped += 1
            continue
        save_npz(out_dir, rec_name, data)
        n_done += 1
        print(f"  [ok] {rec_name}  n_samples={data['signal'].shape[0]}  "
              f"beats={len(data['beat_sample_idx'])}  "
              f"rhythms={len(data['rhythm_label'])}")
    summary = {"dataset": dataset, "done": n_done,
               "skipped": n_skipped, "failed": n_failed}
    print(f"--- {dataset}: done={n_done} skipped={n_skipped} failed={n_failed}")
    return summary


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset, cfg in DATASETS.items():
        summaries.append(process_dataset(dataset, cfg))
    print("\n=== Summary ===")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()