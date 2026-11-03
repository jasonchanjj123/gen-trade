"""Validate candle provenance and align external features by knowledge time."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


def _utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values)
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} must contain explicit timezone-aware timestamps")
    if parsed.isna().any():
        raise ValueError(f"{name} cannot contain missing timestamps")
    return parsed.dt.tz_convert("UTC")


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return an independent UTC frame, rejecting unordered or impossible candles.

    timestamp is bar-open time; available_at is when its full OHLCV was known.
    Missing volume is allowed (e.g. spot FX); prices must be observed and finite.
    This function accepts exactly one instrument at a time.
    """
    required = {"timestamp", "available_at", *FIELDS}
    if missing := required.difference(bars.columns):
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    if bars.empty:
        raise ValueError("At least one candle is required")
    out = bars.copy(deep=True).reset_index(drop=True)
    out["timestamp"] = _utc(out["timestamp"], "timestamp")
    out["available_at"] = _utc(out["available_at"], "available_at")
    for name in ("timestamp", "available_at"):
        if not out[name].is_monotonic_increasing or out[name].duplicated().any():
            raise ValueError(f"{name} must be strictly increasing and unique")
    if (out["available_at"] < out["timestamp"]).any():
        raise ValueError("Candle cannot be available before it opens")
    # Each next bar may only use already completed OHLCV for this instrument.
    if (out["available_at"].iloc[:-1].to_numpy() > out["timestamp"].iloc[1:].to_numpy()).any():
        raise ValueError("Overlapping or delayed bars require an explicit ingestion adapter")
    if "symbol" in out and (out["symbol"].isna().any() or out["symbol"].nunique() != 1):
        raise ValueError("Pass exactly one nonmissing instrument to validate_bars")
    for name in FIELDS:
        out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
    price = out[["open", "high", "low", "close"]].to_numpy()
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("OHLC prices must be finite and positive")
    if (out["high"] < out[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("High must be at least open, low, and close")
    if (out["low"] > out[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Low must be at most open, high, and close")
    volume = out["volume"].to_numpy()
    if np.isinf(volume).any() or (volume < 0).any():
        raise ValueError("Volume must be nonnegative or missing")
    return out


@dataclass(frozen=True)
class FeatureGene:
    """A chromosome terminal referencing an instrument's observed OHLCV."""

    source: str
    field: str = "close"
    lag: int = 0
    window: int = 1
    max_age: str = "1D"

    def __post_init__(self):
        if not self.source or self.field not in FIELDS:
            raise ValueError("Feature gene needs a source and a valid OHLCV field")
        if not isinstance(self.lag, int) or self.lag < 0:
            raise ValueError("lag must be a nonnegative integer")
        if not isinstance(self.window, int) or self.window < 1:
            raise ValueError("window must be a positive integer")
        if pd.isna(pd.Timedelta(self.max_age)) or pd.Timedelta(self.max_age) <= pd.Timedelta(0):
            raise ValueError("max_age must be positive")

    def key(self) -> str:
        return f"{self.source}:{self.field}:lag{self.lag}:mean{self.window}:age{self.max_age}"


@dataclass(frozen=True)
class ChromosomeContext:
    """User extensions attached to a method-specific tree or weight chromosome.

    Pair/session contexts are explicit experiment choices, not hidden additions
    to the papers' original evolutionary operators.
    """

    pair: str
    entry_hours: tuple[int, ...] | None = None
    features: tuple[FeatureGene, ...] = ()

    def __post_init__(self):
        if not self.pair:
            raise ValueError("pair cannot be empty")
        if self.entry_hours is not None and (
            not self.entry_hours or any(type(x) is not int or not 0 <= x <= 23 for x in self.entry_hours)
        ):
            raise ValueError("entry_hours must be nonempty UTC hours in 0..23")

    def to_dict(self) -> dict:
        return asdict(self)


def align_features(
    target: pd.DataFrame, sources: dict[str, pd.DataFrame], genes: tuple[FeatureGene, ...]
) -> dict[str, np.ndarray]:
    """Backward as-of join, with lags and finite staleness limits; never backfill.

    max_age bounds the age of the published feature row. A lag is intentional
    historical information; it does not pretend the underlying observation is
    current. Missing calendar observations remain missing after the tolerance.
    """
    decisions = validate_bars(target)[["available_at"]]
    result = {}
    for gene in genes:
        if gene.source not in sources:
            raise ValueError(f"Missing source {gene.source}")
        source = validate_bars(sources[gene.source])
        values = source[gene.field].shift(gene.lag).rolling(gene.window, min_periods=gene.window).mean()
        right = pd.DataFrame({"known_at": source["available_at"], "value": values})
        joined = pd.merge_asof(
            decisions, right, left_on="available_at", right_on="known_at", direction="backward",
            tolerance=pd.Timedelta(gene.max_age), allow_exact_matches=True,
        )
        if ((joined["known_at"].notna()) & (joined["known_at"] > joined["available_at"])).any():
            raise AssertionError("Noncausal feature join")
        if gene.key() in result:
            raise ValueError("Duplicate feature genes")
        result[gene.key()] = joined["value"].to_numpy(dtype=float)
    return result


def read_market_data(path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "symbol" not in frame:
        raise ValueError("Market-data file must have a symbol column")
    if frame.empty or frame["symbol"].isna().any():
        raise ValueError("Market data must contain nonmissing instrument rows")
    return {str(symbol): validate_bars(part) for symbol, part in frame.groupby("symbol", sort=False)}
