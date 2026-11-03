"""Causal directional-change states, with an optional Rust accelerator.

Rows describe what was knowable *at that observation*. A reversal is reported at
its confirmation, never backdated to the preceding extremum. The eventual end
of a trend and its overshoot are deliberately not labelled in this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    from gen_trade_native import dc_arrays as _native_dc_arrays
except ImportError:
    _native_dc_arrays = None


@dataclass(frozen=True)
class DCProfile:
    """Observation-aligned state for one relative directional-change threshold.

    ``direction`` is zero until a threshold first confirms an upturn (+1) or
    downturn (-1). Before that confirmation, anchor/confirmation prices are NaN
    and their indices are -1. ``extreme_price`` is the current price while the
    direction is unknown, then the running high/low of the confirmed trend.
    Equal extrema retain the first index. ``has_overshoot`` means that a strict
    favorable extension beyond this trend's actual confirmation price has been
    observed; it remains true until the next confirmation.

    Durations derived from these indices are in observations, not calendar time.
    A consumer needing physical durations must index its own timestamps.
    """

    threshold: float
    direction: np.ndarray
    confirmation: np.ndarray
    anchor_price: np.ndarray
    anchor_index: np.ndarray
    confirmation_index: np.ndarray
    confirmation_price: np.ndarray
    extreme_price: np.ndarray
    extreme_index: np.ndarray
    has_overshoot: np.ndarray

    def __len__(self) -> int:
        return self.direction.size


def _python_dc_arrays(prices: np.ndarray, threshold: float) -> tuple:
    n = prices.size
    direction = np.zeros(n, dtype=np.int8)
    confirmation = np.zeros(n, dtype=bool)
    anchor_prices = np.full(n, np.nan)
    anchor_indices = np.full(n, -1, dtype=np.int64)
    confirmation_indices = np.full(n, -1, dtype=np.int64)
    confirmation_prices = np.full(n, np.nan)
    extreme_prices = prices.copy()
    extreme_indices = np.arange(n, dtype=np.int64)
    overshoots = np.zeros(n, dtype=bool)
    if n == 0:
        return (direction, confirmation, anchor_prices, anchor_indices,
                confirmation_indices, confirmation_prices, extreme_prices,
                extreme_indices, overshoots)

    state = 0
    high = low = float(prices[0])
    high_index = low_index = 0
    anchor_price = confirmation_price = np.nan
    anchor_index = confirmation_index = -1
    extreme_price = float(prices[0])
    extreme_index = 0
    has_overshoot = False
    for i, price_value in enumerate(prices):
        price = float(price_value)
        confirmed = False
        if state == 0:
            if price > high:
                high, high_index = price, i
            if price < low:
                low, low_index = price, i
            if price > low and price >= low * (1.0 + threshold):
                state, anchor_price, anchor_index = 1, low, low_index
                confirmed = True
            elif price < high and price <= high * (1.0 - threshold):
                state, anchor_price, anchor_index = -1, high, high_index
                confirmed = True
        elif state == 1:
            if price < extreme_price and price <= extreme_price * (1.0 - threshold):
                state, anchor_price, anchor_index = -1, extreme_price, extreme_index
                confirmed = True
            elif price > extreme_price:
                extreme_price, extreme_index, has_overshoot = price, i, True
        else:
            if price > extreme_price and price >= extreme_price * (1.0 + threshold):
                state, anchor_price, anchor_index = 1, extreme_price, extreme_index
                confirmed = True
            elif price < extreme_price:
                extreme_price, extreme_index, has_overshoot = price, i, True

        if confirmed:
            confirmation_price, confirmation_index = price, i
            extreme_price, extreme_index, has_overshoot = price, i, False
        direction[i] = state
        confirmation[i] = confirmed
        if state:
            anchor_prices[i], anchor_indices[i] = anchor_price, anchor_index
            confirmation_prices[i], confirmation_indices[i] = confirmation_price, confirmation_index
            extreme_prices[i], extreme_indices[i] = extreme_price, extreme_index
            overshoots[i] = has_overshoot

    return (direction, confirmation, anchor_prices, anchor_indices,
            confirmation_indices, confirmation_prices, extreme_prices,
            extreme_indices, overshoots)


def directional_changes(
    prices: np.ndarray,
    threshold: float,
    *,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> DCProfile:
    """Profile a positive finite price sequence without future information.

    ``threshold`` is a fraction: 0.01 means one percent. Initial direction is
    unknown while running high/low are tracked, and whichever threshold crosses
    first establishes the trend. Equality confirms a threshold; extrema require
    a strict new high/low. No future observations revise an earlier output row.

    ``auto`` uses the native module when installed, otherwise the Python
    reference. ``rust`` fails explicitly if the extension is unavailable.
    """
    if backend not in ("auto", "python", "rust"):
        raise ValueError("backend must be 'auto', 'python', or 'rust'")
    try:
        values = np.asarray(prices)
        if np.iscomplexobj(values):
            raise ValueError("prices must be real numbers")
        values = np.asarray(values, dtype=np.float64)
        theta = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("prices and threshold must be real numeric values") from exc
    if values.ndim != 1:
        raise ValueError("prices must be one-dimensional")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("prices must be finite and strictly positive")
    if not np.isfinite(theta) or not 0.0 < theta < 1.0:
        raise ValueError("threshold must be finite and between zero and one")
    if backend == "rust" and _native_dc_arrays is None:
        raise RuntimeError("the gen_trade_native extension is not installed")

    if backend != "python" and _native_dc_arrays is not None:
        arrays = _native_dc_arrays(values.tolist(), theta)
        dtypes = (np.int8, bool, np.float64, np.int64, np.int64,
                  np.float64, np.float64, np.int64, bool)
        arrays = tuple(np.asarray(array, dtype=dtype) for array, dtype in zip(arrays, dtypes))
    else:
        arrays = _python_dc_arrays(values, theta)
    return DCProfile(theta, *arrays)
