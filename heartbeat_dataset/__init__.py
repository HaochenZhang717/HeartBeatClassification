"""Heartbeat classification dataset built from processed PhysioNet records.

Each datum is a paired (waveform_window, label) sample, where the waveform is
centered on an R-peak and the label is the AAMI 5-class beat type.
"""

from .dataset import HeartbeatDataset, AAMI_CLASSES, AAMI_TO_INT, INT_TO_AAMI
from .splits import (
    MITDB_DS1,
    MITDB_DS2,
    split_by_record,
    filter_index_by_records,
)

__all__ = [
    "HeartbeatDataset",
    "AAMI_CLASSES",
    "AAMI_TO_INT",
    "INT_TO_AAMI",
    "MITDB_DS1",
    "MITDB_DS2",
    "split_by_record",
    "filter_index_by_records",
]
