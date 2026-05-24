"""Inter-patient splits for heartbeat classification.

`MITDB_DS1` / `MITDB_DS2` follow the de Chazal et al. (2004) split, which is
the standard inter-patient protocol for MIT-BIH Arrhythmia Database. The two
sets share no patients, so a model trained on DS1 and evaluated on DS2 cannot
exploit subject-specific morphology.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# de Chazal et al. 2004 inter-patient split for MIT-BIH Arrhythmia (mitdb).
# Records 102, 104, 107, 217 (paced) are excluded by the AAMI standard and
# are already filtered out by process_ecg.py.
MITDB_DS1: tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
)
MITDB_DS2: tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
)


def filter_index_by_records(
    index: dict[str, np.ndarray],
    records: Iterable[str],
    dataset: str | None = None,
) -> np.ndarray:
    """Return the row indices of `index` whose record_name is in `records`.

    If `dataset` is given, also require index['dataset'] == dataset (useful
    because record names can collide across datasets).
    """
    records = set(records)
    mask = np.isin(index["record_name"], list(records))
    if dataset is not None:
        mask &= (index["dataset"] == dataset)
    return np.nonzero(mask)[0]


def split_by_record(
    index: dict[str, np.ndarray],
    val_frac: float = 0.2,
    seed: int = 0,
    dataset: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random patient-level split: each (dataset, record_name) goes entirely
    to either train or val. Returns (train_rows, val_rows) into `index`.
    """
    if dataset is not None:
        ds_mask = (index["dataset"] == dataset)
    else:
        ds_mask = np.ones(len(index["record_name"]), dtype=bool)

    # Unique (dataset, record_name) pairs among the masked rows.
    key = np.char.add(
        np.char.add(index["dataset"].astype("<U32"), "/"),
        index["record_name"].astype("<U32"),
    )
    unique_keys = np.unique(key[ds_mask])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_keys))
    n_val = int(round(val_frac * len(unique_keys)))
    val_keys = set(unique_keys[perm[:n_val]].tolist())

    is_val = np.array([k in val_keys for k in key])
    train_rows = np.nonzero(ds_mask & ~is_val)[0]
    val_rows = np.nonzero(ds_mask & is_val)[0]
    return train_rows, val_rows
