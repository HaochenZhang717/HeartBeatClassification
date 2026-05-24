"""Build train / val / test HeartbeatDataset instances and a sampler.

Split policy
------------
- Test  : mitdb DS2 (22 records, held out — never touched by training/selection)
- Train pool : mitdb DS1 + full svdb + edb + stdb + nsrdb
- Val   : 10% of (dataset, record) keys from the train pool, patient-level

afdb is excluded — its beats default to 'N' (no AAMI annotation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from heartbeat_dataset import (
    AAMI_CLASSES,
    HeartbeatDataset,
    MITDB_DS1,
    MITDB_DS2,
)


TRAIN_POOL_EXTRA = ("svdb", "edb", "stdb", "nsrdb")
# mitdb enters the pool only via DS1 (DS2 is the test set).

# A cache that fits every processed record (~250 across all datasets) keeps
# random-access sampling cheap after epoch 1.
DEFAULT_CACHE_RECORDS = 300


def build_splits(
    index_path: str | Path,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) row indices into beats_index.npz.

    Splits are at the (dataset, record_name) level so no record is shared
    between any two splits.
    """
    raw = np.load(index_path, allow_pickle=False)
    ds = raw["dataset"]
    rn = raw["record_name"]

    test_mask = (ds == "mitdb") & np.isin(rn, list(MITDB_DS2))
    test_idx = np.nonzero(test_mask)[0]

    mitdb_ds1_mask = (ds == "mitdb") & np.isin(rn, list(MITDB_DS1))
    extra_mask = np.isin(ds, list(TRAIN_POOL_EXTRA))
    pool_mask = mitdb_ds1_mask | extra_mask

    keys = np.char.add(np.char.add(ds.astype("<U32"), "/"),
                       rn.astype("<U32"))
    pool_keys = np.unique(keys[pool_mask])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool_keys))
    n_val = int(round(val_frac * len(pool_keys)))
    val_key_set = set(pool_keys[perm[:n_val]].tolist())

    is_val = np.array([k in val_key_set for k in keys])
    train_idx = np.nonzero(pool_mask & ~is_val)[0]
    val_idx = np.nonzero(pool_mask & is_val)[0]
    return train_idx, val_idx, test_idx


def build_datasets(
    index_path: str | Path,
    classes: list[str] | tuple[str, ...] = ("N", "S", "V", "F", "Q"),
    val_frac: float = 0.1,
    seed: int = 0,
    pre_samples: int = 100,
    post_samples: int = 150,
    normalize: str = "zscore-window",
    return_torch: bool = True,
    cache_records: int = DEFAULT_CACHE_RECORDS,
) -> tuple[HeartbeatDataset, HeartbeatDataset, HeartbeatDataset]:
    train_idx, val_idx, test_idx = build_splits(
        index_path, val_frac=val_frac, seed=seed)
    common = dict(
        index_path=index_path,
        classes=list(classes),
        pre_samples=pre_samples,
        post_samples=post_samples,
        normalize=normalize,
        return_torch=return_torch,
        cache_records=cache_records,
    )
    train_ds = HeartbeatDataset(indices=train_idx, **common)
    val_ds = HeartbeatDataset(indices=val_idx, **common)
    test_ds = HeartbeatDataset(indices=test_idx, **common)
    return train_ds, val_ds, test_ds


def make_balanced_sampler(train_ds: HeartbeatDataset) -> WeightedRandomSampler:
    """Per-sample weight = 1 / count(class). Yields ~uniform class freq."""
    labels = np.asarray(train_ds._rows["label_int"], dtype=np.int64)
    n_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    class_weight = np.zeros_like(counts)
    nz = counts > 0
    class_weight[nz] = 1.0 / counts[nz]
    sample_weights = class_weight[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
