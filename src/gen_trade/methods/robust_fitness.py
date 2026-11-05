"""EXTENSION (review 2026-11-05): fitness that measures what the tests measure.

Each candidate is scored by the annualized information ratio of its training
returns in excess of a static cash/buy-and-hold blend with the candidate's own
mean exposure, ``d_t = r_t - e * b_t``. Under no timing skill the expected
excess is zero whatever the exposure, so the score cannot be raised by holding
the market longer or shorter, only by being invested on better days.

Scoring uses a different random half of the training months every generation
(stochastic sub-window fitness; cf. Goncalves and Silva 2012 random sampling
and down-sampled selection, Geiger et al. 2025), so a rule must do well on
many resamples, not only on the full sample. Elites are re-scored on each new
sample. Final choices use the full training window.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..backtest import ExecutionConfig, run_backtest


def plain_execution(execution: ExecutionConfig) -> ExecutionConfig:
    return replace(execution, take_profit=None, stop_loss=None, max_holding_days=None)


def period_returns(equity: np.ndarray) -> np.ndarray:
    """Per-bar returns from (bars + 1) equity marks; columns are strategies."""
    values = np.asarray(equity, dtype=float)
    return np.divide(np.diff(values, axis=0), values[:-1], out=np.zeros_like(values[1:]), where=values[:-1] > 0)


def benchmark_returns(bars: pd.DataFrame, execution: ExecutionConfig, market=None) -> np.ndarray:
    """Next-open buy-and-hold under the same costs, per bar."""
    ones = np.ones(len(bars), dtype=np.int8)
    plain = plain_execution(execution)
    if plain.instrument is not None:
        from ..market_execution import MarketArrays, execute_batch
        market = market if market is not None else MarketArrays.from_bars(bars, plain.instrument)
        equity = execute_batch(market, ones, plain, record_equity=True).equity[:, 0]
    else:
        equity = run_backtest(bars, ones, plain).equity
    return period_returns(equity)


def bar_months(bars: pd.DataFrame) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(bars["timestamp"], utc=True))
    return np.asarray(times.year * 12 + times.month - 1)


class SubWindows:
    """Deterministic random subsets of training months, one per generation."""

    def __init__(self, months: np.ndarray, seed: int, fraction: float = 0.5, min_months: int = 6):
        if not 0 < fraction <= 1 or type(min_months) is not int or min_months < 1:
            raise ValueError("Sub-window fraction must be in (0,1] and min_months a positive integer")
        self.months = np.asarray(months)
        self.unique = np.unique(self.months)
        self.seed, self.fraction = int(seed), float(fraction)
        self.size = int(min(len(self.unique), max(min_months, round(fraction * len(self.unique)))))

    def mask(self, generation: int | None) -> np.ndarray:
        if generation is None or self.size >= len(self.unique):
            return np.ones(len(self.months), dtype=bool)
        rng = np.random.default_rng([self.seed, int(generation), 104729])
        chosen = rng.choice(self.unique, size=self.size, replace=False)
        return np.isin(self.months, chosen)

    def describe(self) -> dict:
        return {"training_months": int(len(self.unique)), "months_per_sample": self.size,
                "fraction": self.fraction, "rule": "fresh random month subset each generation; full window for final choice"}


def excess_paths(returns: np.ndarray, exposure: np.ndarray, benchmark: np.ndarray,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """Excess over the exposure-matched blend on the masked periods.

    ``returns`` and ``exposure`` are periods x candidates; the blend exposure is
    each candidate's mean exposure over the same masked periods.
    """
    r, x, b = np.asarray(returns, dtype=float), np.asarray(exposure, dtype=float), np.asarray(benchmark, dtype=float)
    if r.ndim == 1:
        r, x = r[:, None], x[:, None]
    if r.shape != x.shape or r.shape[0] != b.shape[0]:
        raise ValueError("Excess paths need aligned returns, exposure and benchmark")
    if mask is not None:
        r, x, b = r[mask], x[mask], b[mask]
    return r - x.mean(axis=0)[None, :] * b[:, None]


def excess_information_ratio(returns, exposure, benchmark, mask=None, *, periods_per_year: float) -> np.ndarray:
    """Annualized mean/std of the excess; zero for constant (e.g. never-invested) paths."""
    excess = excess_paths(returns, exposure, benchmark, mask)
    if excess.shape[0] < 3:
        return np.zeros(excess.shape[1])
    mean, std = excess.mean(axis=0), excess.std(axis=0, ddof=1)
    return np.divide(mean, std, out=np.zeros_like(mean), where=std > 1e-15) * np.sqrt(periods_per_year)


def periods_per_year(bars: pd.DataFrame) -> float:
    times = pd.to_datetime(bars["available_at"], utc=True)
    years = (times.iloc[-1] - pd.to_datetime(bars["timestamp"], utc=True).iloc[0]).total_seconds() / (365.25 * 86400)
    return float(len(bars) / max(years, 1e-9))


def selection_haircut(excess: np.ndarray, selected: int, *, periods_per_year: float, reps: int = 500, seed: int = 0) -> dict:
    """Rademacher Anti-Serum over every evaluated candidate's full-window excess.

    Reports how much of the selected candidate's training excess Sharpe
    survives the data-snooping haircut implied by the whole search.
    """
    from ..statistics import rademacher_antiserum
    values = np.asarray(excess, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] < 8:
        return {"note": "Too few candidates or periods for a haircut."}
    ras = rademacher_antiserum(values, reps=reps, seed=seed, periods_per_year=periods_per_year)
    annual = np.sqrt(periods_per_year)
    return {"candidates": int(values.shape[1]), "periods": int(values.shape[0]),
            "selected_excess_sharpe_annualized": float(ras["sharpe_per_period"][selected] * annual),
            "data_snooping_haircut_annualized": float(ras["data_snooping_haircut"] * annual),
            "selected_after_haircut_annualized": float(ras["snooping_adjusted_sharpe_per_period"][selected] * annual),
            "selected_rademacher_positive": bool(ras["rademacher_positive"][selected]),
            "rademacher_positive_candidates": int(np.sum(ras["rademacher_positive"])),
            "method": "Rademacher Anti-Serum (Paleologo 2024) over all evaluated candidates' training excess returns"}
