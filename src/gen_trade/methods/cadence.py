"""EXTENSION (results 2026-11-05): decide weekly or monthly instead of daily.

The audited2 runs showed timing at chance and a significant loss from the
structure and cost of trading itself, so fewer decisions are tested directly.
At the first bar of each calendar week or month (by bar open time, UTC) the
strategy acts on the most recent nonzero signal raised since the previous
decision bar; every other bar emits no action. Decisions use only bars up to
the decision bar, and the shared simulators still execute them at the next
open. Profit targets, stops and holding deadlines keep running daily.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECISION_CADENCES = ("daily", "weekly", "monthly")


def decision_bars(timestamps, cadence: str) -> np.ndarray:
    """True on the first bar of each period (and on the first bar overall)."""
    if cadence not in DECISION_CADENCES:
        raise ValueError(f"decision_cadence must be one of {DECISION_CADENCES}")
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if cadence == "daily":
        return np.ones(len(times), dtype=bool)
    if cadence == "weekly":
        iso = times.isocalendar()
        period = np.asarray(iso["year"], dtype=np.int64) * 100 + np.asarray(iso["week"], dtype=np.int64)
    else:
        period = np.asarray(times.year * 12 + times.month, dtype=np.int64)
    first = np.ones(len(period), dtype=bool)
    first[1:] = period[1:] != period[:-1]
    return first


def apply_cadence(signals: np.ndarray, timestamps, cadence: str, axis: int = 0) -> np.ndarray:
    """Keep only decisions on decision bars, carrying each period's latest signal.

    ``signals`` holds -1/0/+1 with time along ``axis`` (0 or 1). Daily cadence
    returns the input unchanged.
    """
    values = np.asarray(signals)
    if cadence == "daily":
        return values
    moved = values if axis == 0 else np.moveaxis(values, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    if flat.shape[0] != len(timestamps):
        raise ValueError("Signals and timestamps must align along the time axis")
    decision = decision_bars(timestamps, cadence)
    index = np.arange(flat.shape[0])[:, None]
    latest = np.maximum.accumulate(np.where(flat != 0, index, -1), axis=0)
    previous = np.full(flat.shape[0], -1)
    points = np.flatnonzero(decision)
    previous[points[1:]] = points[:-1]  # last bar already decided before this one
    fresh = decision[:, None] & (latest > previous[:, None])
    picked = np.take_along_axis(flat, np.maximum(latest, 0), axis=0)
    result = np.where(fresh, picked, 0).astype(values.dtype).reshape(moved.shape)
    return result if axis == 0 else np.moveaxis(result, 0, axis)
