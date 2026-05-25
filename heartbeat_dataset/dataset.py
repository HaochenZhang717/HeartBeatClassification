"""HeartbeatDataset: paired (waveform_window, label) samples for AAMI 5-class
beat classification.

Typical usage
-------------
    from heartbeat_dataset import HeartbeatDataset, MITDB_DS1, MITDB_DS2

    train_ds = HeartbeatDataset(
        index_path="heartbeat_dataset/beats_index.npz",
        records=MITDB_DS1, dataset="mitdb",
        pre_samples=100, post_samples=150,
        normalize="zscore-window",
        return_torch=True,
    )
    val_ds = HeartbeatDataset(
        index_path="heartbeat_dataset/beats_index.npz",
        records=MITDB_DS2, dataset="mitdb",
        pre_samples=100, post_samples=150,
        normalize="zscore-window",
        return_torch=True,
    )

    wave, label = train_ds[0]   # wave: (window, 2) float32; label: int

The dataset loads per-record signals lazily and keeps an LRU cache so that
sequential / record-grouped sampling stays fast. Each beat that doesn't fit
fully inside the record (would require samples beyond the signal boundary)
is padded with edge values.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")
AAMI_TO_INT: dict[str, int] = {c: i for i, c in enumerate(AAMI_CLASSES)}
INT_TO_AAMI: dict[int, str] = {i: c for c, i in AAMI_TO_INT.items()}


class _SignalCache:
    """Tiny LRU cache from record_path -> signal ndarray."""

    def __init__(self, max_records: int = 8):
        self.max_records = max_records
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: str) -> np.ndarray:
        if path in self._store:
            self._store.move_to_end(path)
            return self._store[path]
        sig = np.load(path, allow_pickle=False)["signal"]  # (n, 2) float32
        self._store[path] = sig
        if len(self._store) > self.max_records:
            self._store.popitem(last=False)
        return sig


class HeartbeatDataset:
    """AAMI heartbeat classification dataset.

    Parameters
    ----------
    index_path : path to beats_index.npz produced by build_index.py
    records : optional whitelist of record_name values. If given, only beats
        from these records are used. Combine with `dataset` to disambiguate
        record-name collisions across datasets.
    dataset : optional whitelist for a single source database (e.g. 'mitdb').
    classes : iterable of AAMI labels to include (default: all 5)
    pre_samples / post_samples : window is [r - pre, r + post), so length =
        pre_samples + post_samples. Defaults assume fs=250 Hz: 100+150 = 250
        samples ~= 1.0 s, centered slightly post-R (covers full PQRST).
    leads : which channels to return. 'both' -> (window, 2). 0 or 1 ->
        (window, 1). default 'both'.
    normalize : 'none' | 'zscore-window' | 'zscore-record'
        - 'zscore-window': per-sample z-score over the returned window
        - 'zscore-record': use the parent record's global mean/std (cached)
    indices : optional subset of row indices (into the beats_index) — used by
        splits. If None, all rows pass the records/dataset/classes filter.
    return_torch : if True, return torch.Tensor instead of np.ndarray
    cache_records : number of full record signals to keep in memory
    """

    def __init__(
        self,
        index_path: str | Path,
        records: Iterable[str] | None = None,
        dataset: str | None = None,
        classes: Sequence[str] = AAMI_CLASSES,
        pre_samples: int = 100,
        post_samples: int = 150,
        leads: str | int = "both",
        normalize: str = "zscore-window",
        indices: np.ndarray | None = None,
        return_torch: bool = False,
        cache_records: int = 8,
        preload: bool = False,
    ) -> None:
        self.index_path = Path(index_path)
        self.pre_samples = int(pre_samples)
        self.post_samples = int(post_samples)
        self.window = self.pre_samples + self.post_samples
        self.leads = leads
        self.normalize = normalize
        self.return_torch = return_torch
        self._cache = _SignalCache(max_records=cache_records)
        self._record_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._windows: np.ndarray | None = None

        if normalize not in {"none", "zscore-window", "zscore-record"}:
            raise ValueError(f"unknown normalize={normalize!r}")
        if leads not in {"both", 0, 1}:
            raise ValueError(f"leads must be 'both', 0, or 1; got {leads!r}")

        raw = np.load(self.index_path, allow_pickle=False)
        idx = {k: raw[k] for k in raw.files}

        # Filter rows: dataset → records → classes → user-supplied indices.
        mask = np.ones(len(idx["sample_idx"]), dtype=bool)
        if dataset is not None:
            mask &= (idx["dataset"] == dataset)
        if records is not None:
            mask &= np.isin(idx["record_name"], list(records))
        class_ids = np.array([AAMI_TO_INT[c] for c in classes], dtype=np.int8)
        mask &= np.isin(idx["label_int"], class_ids)
        selected = np.nonzero(mask)[0]
        if indices is not None:
            selected = np.intersect1d(selected, np.asarray(indices, dtype=np.int64))

        self._rows = {k: v[selected] for k, v in idx.items()}
        self._n = len(selected)

        if return_torch:
            import torch  # noqa: F401  (raises a clear error if missing)
            self._torch = __import__("torch")
        else:
            self._torch = None

        if preload:
            self._preload_windows()

    # ----- introspection ------------------------------------------------ #
    def __len__(self) -> int:
        return self._n

    @property
    def num_classes(self) -> int:
        return len(AAMI_CLASSES)

    def class_counts(self) -> dict[str, int]:
        out = {c: 0 for c in AAMI_CLASSES}
        unique, counts = np.unique(self._rows["label_int"], return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            out[INT_TO_AAMI[int(u)]] = int(c)
        return out

    def class_weights(self, method: str = "inv_freq") -> np.ndarray:
        """Return per-class weights of shape (num_classes,).

        method='inv_freq' -> w_c = N_total / (num_classes * N_c).
        Classes that don't appear at all get weight 0.
        """
        counts = np.zeros(len(AAMI_CLASSES), dtype=np.float64)
        for c, n in self.class_counts().items():
            counts[AAMI_TO_INT[c]] = n
        weights = np.zeros_like(counts)
        present = counts > 0
        if method == "inv_freq":
            n_total = counts.sum()
            weights[present] = n_total / (present.sum() * counts[present])
        else:
            raise ValueError(f"unknown method={method!r}")
        return weights.astype(np.float32)

    # ----- core access -------------------------------------------------- #
    def _get_record_stats(self, path: str, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if path not in self._record_stats:
            mu = sig.mean(axis=0, keepdims=True)
            sd = sig.std(axis=0, keepdims=True) + 1e-6
            self._record_stats[path] = (mu.astype(np.float32), sd.astype(np.float32))
        return self._record_stats[path]

    def _window(self, sig: np.ndarray, center: int) -> np.ndarray:
        """Extract sig[center - pre : center + post, :], edge-padding when
        the window runs off either end of the record."""
        n = sig.shape[0]
        start = center - self.pre_samples
        end = center + self.post_samples
        pad_left = max(0, -start)
        pad_right = max(0, end - n)
        s = max(0, start)
        e = min(n, end)
        win = sig[s:e]
        if pad_left or pad_right:
            win = np.pad(
                win,
                ((pad_left, pad_right), (0, 0)),
                mode="edge",
            )
        return win  # (window, 2) float32

    def _preload_windows(self) -> None:
        """Materialize every (window, leads_out) sample into a single ndarray.

        Pre-sorts beats by record_path so each per-record .npz is loaded once.
        Memory: ~ n * window * leads_out * 4 bytes (≈5 GB for the full train pool).
        """
        n = self._n
        leads_out = 2 if self.leads == "both" else 1
        arr = np.empty((n, self.window, leads_out), dtype=np.float32)

        paths = self._rows["record_path"]
        centers = self._rows["sample_idx"]
        order = np.argsort(paths, kind="stable")

        last_path: str | None = None
        sig: np.ndarray | None = None
        print(f"[preload] materializing {n} windows ...")
        for j, i in enumerate(order):
            path = str(paths[i])
            if path != last_path:
                sig = np.load(path, allow_pickle=False)["signal"]
                last_path = path
            win = self._window(sig, int(centers[i])).astype(np.float32, copy=False)
            if self.normalize == "zscore-window":
                mu = win.mean(axis=0, keepdims=True)
                sd = win.std(axis=0, keepdims=True) + 1e-6
                win = (win - mu) / sd
            elif self.normalize == "zscore-record":
                mu, sd = self._get_record_stats(path, sig)
                win = (win - mu) / sd
            if self.leads == 0:
                win = win[:, 0:1]
            elif self.leads == 1:
                win = win[:, 1:2]
            arr[i] = win
            if (j + 1) % 250_000 == 0:
                print(f"[preload]   {j + 1}/{n}")
        self._windows = arr
        # The per-record signal cache is no longer needed once windows are in RAM.
        self._cache = _SignalCache(max_records=1)
        print(f"[preload] done — {arr.nbytes / 1e9:.2f} GB resident")

    def __getitem__(self, i: int):
        label = int(self._rows["label_int"][i])

        if self._windows is not None:
            win = self._windows[i]
            if self._torch is not None:
                return (
                    self._torch.from_numpy(win),
                    self._torch.tensor(label, dtype=self._torch.long),
                )
            return win, label

        path = str(self._rows["record_path"][i])
        center = int(self._rows["sample_idx"][i])

        sig = self._cache.get(path)
        win = self._window(sig, center).astype(np.float32, copy=False)

        if self.normalize == "zscore-window":
            mu = win.mean(axis=0, keepdims=True)
            sd = win.std(axis=0, keepdims=True) + 1e-6
            win = (win - mu) / sd
        elif self.normalize == "zscore-record":
            mu, sd = self._get_record_stats(path, sig)
            win = (win - mu) / sd

        if self.leads == 0:
            win = win[:, 0:1]
        elif self.leads == 1:
            win = win[:, 1:2]

        if self._torch is not None:
            return (
                self._torch.from_numpy(win.copy()),
                self._torch.tensor(label, dtype=self._torch.long),
            )
        return win, label

    # ----- convenience -------------------------------------------------- #
    def get_meta(self, i: int) -> dict:
        return {
            "dataset": str(self._rows["dataset"][i]),
            "record_name": str(self._rows["record_name"][i]),
            "sample_idx": int(self._rows["sample_idx"][i]),
            "label_str": str(self._rows["label_str"][i]),
            "label_int": int(self._rows["label_int"][i]),
        }
