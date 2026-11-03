"""Causal terminals for Long et al. (2026), DOI 10.1007/s10462-025-11390-9.

Tables 1 and 2 specify 28 DC and 28 physical-time terminals. The article does
not fully specify rolling/normalization conventions. Here completed DC values
become available only at the *reversing confirmation*, and their rolling means
use observation-weighted, forward-held values over elapsed calendar-day windows.
NDC/CDC count confirmation-time events; AT integrates the previously confirmed
direction over elapsed time. TA-Lib periods count observations (days only for
daily input). These are documented implementation choices, not author code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import talib

from ..data import validate_bars
from ..dc import directional_changes


def dc_features(bars: pd.DataFrame, threshold: float) -> dict[str, np.ndarray]:
    frame = validate_bars(bars)
    close = frame["close"].to_numpy()
    profile = directional_changes(close, threshold)
    n = len(frame)
    time_index = pd.DatetimeIndex(frame["available_at"])
    # pandas may retain microsecond resolution; asi8 is not unconditionally ns.
    nanos = time_index.as_unit("ns").asi8
    tmv = np.full(n, np.nan)
    tdc = np.full(n, np.nan)
    rdc = np.full(n, np.nan)
    completed_tmv = np.zeros(n)
    last_tmv = last_tdc = last_rdc = np.nan
    for i in range(n):
        if i and profile.confirmation[i] and profile.direction[i - 1] != 0:
            start = int(profile.anchor_index[i - 1])
            end = int(profile.anchor_index[i])
            duration = float(nanos[end] - nanos[start]) / 86_400_000_000_000.0
            if start >= 0 and end > start and duration > 0:
                last_tmv = (profile.anchor_price[i] / profile.anchor_price[i - 1] - 1) / threshold
                last_tdc = duration
                last_rdc = last_tmv * threshold / duration
                completed_tmv[i] = abs(last_tmv)
        tmv[i], tdc[i], rdc[i] = last_tmv, last_tdc, last_rdc
    with np.errstate(invalid="ignore", divide="ignore"):
        osv = (close / profile.confirmation_price - 1) / threshold
    result = {"dc_tmv": tmv, "dc_osv": osv, "dc_rdc": rdc, "dc_tdc": tdc}
    for name, values in (("osv", osv), ("rdc", rdc), ("tdc", tdc)):
        for window in (3, 5, 10):
            result[f"dc_avg_{name}_{window}d"] = pd.Series(values, index=time_index).rolling(
                f"{window}D", min_periods=1, closed="right"
            ).mean().to_numpy()
    events = pd.Series(profile.confirmation.astype(float), index=time_index)
    movements = pd.Series(completed_tmv, index=time_index)
    # The interval ending now was experienced under the previously known trend.
    elapsed = np.r_[0.0, np.diff(nanos).astype(float) / 86_400_000_000_000.0]
    signed_duration = elapsed * np.r_[0, profile.direction[:-1]]
    # Integrate piecewise-constant known direction. A sparse interval can begin
    # before a window's left edge, so summing intervals by their end timestamps
    # would overcount the portion outside that window. Interpolate the signed
    # cumulative integral at the exact left boundary instead.
    elapsed_days = (nanos - nanos[0]).astype(float) / 86_400_000_000_000.0
    cumulative_duration = np.cumsum(signed_duration)
    for window in (10, 20, 30, 40, 50):
        options = {"window": f"{window}D", "min_periods": 1, "closed": "right"}
        result[f"dc_ndc_{window}d"] = events.rolling(**options).sum().to_numpy()
        result[f"dc_cdc_{window}d"] = movements.rolling(**options).sum().to_numpy()
        left_integral = np.interp(
            elapsed_days - window, elapsed_days, cumulative_duration, left=0.0,
        )
        result[f"dc_at_{window}d"] = cumulative_duration - left_integral
    return result


def physical_features(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    frame = validate_bars(bars)
    high, low, close, volume = (frame[name].to_numpy() for name in ("high", "low", "close", "volume"))
    result = {}
    for period in (10, 20, 30, 40, 50):
        result[f"ta_ma_{period}"] = talib.SMA(close, timeperiod=period)
        result[f"ta_cci_{period}"] = talib.CCI(high, low, close, timeperiod=period)
        result[f"ta_rsi_{period}"] = talib.RSI(close, timeperiod=period)
        result[f"ta_willr_{period}"] = talib.WILLR(high, low, close, timeperiod=period)
    for period in (3, 5, 10):
        result[f"ta_atr_{period}"] = talib.ATR(high, low, close, timeperiod=period)
        result[f"ta_ema_{period}"] = talib.EMA(close, timeperiod=period)
    result["ta_obv"] = talib.OBV(close, volume)
    result["ta_psar"] = talib.SAR(high, low)
    return result


def raw_features(
    bars: pd.DataFrame, threshold: float, extra_features: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    features = {**dc_features(bars, threshold), **physical_features(bars)}
    for name, raw in (extra_features or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("Extra feature names must be nonempty strings")
        values = np.asarray(raw, dtype=float)
        if values.shape != (len(bars),) or np.isinf(values).any():
            raise ValueError(f"Extra feature {name!r} must have length n, with finite values or NaN")
        features[f"extra:{name}"] = values.copy()
    return features


@dataclass(frozen=True)
class FeatureScaler:
    """Training-fitted min/max bounds; missing inputs stay missing, no backfill.

    Constant/all-missing training columns carry no learned ordering: finite
    values map to zero. Future out-of-range values are clipped to [0,1].
    """

    bounds: dict[str, tuple[float, float]]

    @classmethod
    def fit(cls, features: dict[str, np.ndarray]) -> FeatureScaler:
        bounds = {}
        for name, values in features.items():
            finite = np.asarray(values)[np.isfinite(values)]
            bounds[name] = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
        return cls(bounds)

    def transform(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if features.keys() != self.bounds.keys():
            raise ValueError("Feature names differ from fitted training feature schema")
        transformed = {}
        for name, values in features.items():
            values = np.asarray(values, dtype=float)
            lo, hi = self.bounds[name]
            normalized = np.full(values.shape, np.nan)
            mask = np.isfinite(values)
            if hi > lo:
                normalized[mask] = np.clip((values[mask] - lo) / (hi - lo), 0.0, 1.0)
            else:
                normalized[mask] = 0.0
            transformed[name] = normalized
        return transformed
