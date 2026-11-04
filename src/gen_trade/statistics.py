"""Dependent-data statistics for held-out, net-of-cost equity paths.

The ordinary equity-return Sharpe reported here is distinct from MOO3's
population-normalized modified Sharpe selection score.  No inference function
corrects for the adaptive search that produced a strategy.  Evaluate untouched
data and retain all trials before interpreting an interval or p-value.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap

if TYPE_CHECKING:
    from gen_trade.backtest import BacktestResult


def _settings(
    periods_per_year: float, bootstrap_reps: int, block_size: float, seed: int
) -> tuple[float, int, float, int]:
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    if isinstance(bootstrap_reps, (bool, np.bool_)) or not isinstance(
        bootstrap_reps, (int, np.integer)
    ) or bootstrap_reps < 100:
        raise ValueError("bootstrap_reps must be an integer of at least 100")
    if not np.isfinite(block_size) or block_size < 1:
        raise ValueError("block_size must be finite and at least one")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return float(periods_per_year), int(bootstrap_reps), float(block_size), int(seed)


def _path(result: BacktestResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    equity = np.asarray(result.equity, dtype=float)
    if equity.ndim != 1 or not equity.size:
        raise ValueError("equity must be a nonempty one-dimensional array")
    if not np.isfinite(equity).all() or np.any(equity <= 0):
        raise ValueError("equity must contain finite positive values")
    trades = np.asarray(result.trade_returns, dtype=float)
    if trades.ndim != 1 or not np.isfinite(trades).all():
        raise ValueError("trade_returns must be a finite one-dimensional array")
    if len(result.trades) != trades.size:
        raise ValueError("trade_returns and trades must have the same length")
    returns = np.diff(equity) / equity[:-1]
    if not np.isfinite(returns).all():
        raise ValueError("equity implies nonfinite period returns")
    return equity, returns, trades


def _finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _constant(values: np.ndarray) -> bool:
    # Equity division introduces roundoff even for a mathematically constant
    # geometric path. Do not report a huge Sharpe from that numerical noise.
    tolerance = 8 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
    return float(np.ptp(values)) <= tolerance


def _inference(
    values: np.ndarray, *, bootstrap_reps: int, block_size: float, seed: int
) -> dict[str, Any]:
    """A two-sided percentile interval and asymptotic intercept-only HAC test.

    A fixed minimum of eight observations is an implementation guard, not a
    statement that eight observations suffice for reliable financial inference.
    Mean bootstrap block length is capped at the observed series length.
    """
    n = values.size
    output: dict[str, Any] = {
        "mean": float(np.mean(values)) if n else None,
        "mean_ci95_stationary_bootstrap": None,
        "hac_standard_error": None,
        "hac_pvalue_mean_zero": None,
        "hac_maxlags": None,
        "bootstrap_reps": bootstrap_reps,
        "requested_mean_block_length": block_size,
        "effective_mean_block_length": min(block_size, float(n)) if n else None,
        "inference_note": None,
    }
    if n < 8:
        output["inference_note"] = "Fewer than eight periods; dependent-data inference omitted."
        return output
    if _constant(values):
        output["inference_note"] = "Constant series; uncertainty and mean-zero tests are undefined."
        return output
    effective_block = min(block_size, float(n))
    bootstrap = StationaryBootstrap(effective_block, values, seed=seed)
    interval = bootstrap.conf_int(
        np.mean, reps=bootstrap_reps, method="percentile", size=0.95
    ).reshape(-1)
    if interval.size == 2 and np.isfinite(interval).all():
        output["mean_ci95_stationary_bootstrap"] = interval.tolist()
    maxlags = min(n - 1, max(1, int(np.floor(4 * (n / 100.0) ** (2 / 9)))))
    fit = sm.OLS(values, np.ones((n, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}, use_t=False
    )
    standard_error = _finite(fit.bse[0])
    output["hac_maxlags"] = maxlags
    if standard_error is not None and standard_error > 0:
        output["hac_standard_error"] = standard_error
        output["hac_pvalue_mean_zero"] = _finite(fit.pvalues[0])
    else:
        output["inference_note"] = "HAC variance is degenerate; mean-zero test omitted."
    return output


def _drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    drawdowns = 1.0 - equity / np.maximum.accumulate(equity)
    current_duration = longest_duration = 0
    for underwater in drawdowns > 0:
        current_duration = current_duration + 1 if underwater else 0
        longest_duration = max(longest_duration, current_duration)
    return float(np.max(drawdowns)), int(longest_duration), int(current_duration)


def _loss_tail(returns: np.ndarray) -> tuple[float | None, float | None]:
    """95% empirical loss VaR and integrated empirical upper-tail mean loss.

    Loss is minus the simple period return.  A negative risk value means the
    corresponding tail still made money.  Expected shortfall fractionally
    weights the boundary observation so the tail probability is exactly 5%.
    """
    if not returns.size:
        return None, None
    losses = np.sort(-returns)[::-1]
    value_at_risk = float(np.quantile(-returns, 0.95, method="inverted_cdf"))
    mass = 0.05 * losses.size
    whole = int(np.floor(mass))
    fraction = mass - whole
    tail_sum = float(losses[:whole].sum())
    if fraction > 0 and whole < losses.size:
        tail_sum += fraction * float(losses[whole])
    return value_at_risk, tail_sum / mass


def detailed_statistics(
    result: BacktestResult,
    *,
    periods_per_year: float,
    bootstrap_reps: int = 500,
    block_size: float = 12,
    seed: int = 0,
) -> dict[str, Any]:
    """Describe an equity path without choosing an annualization convention.

    All returns, losses, volatilities, and drawdowns are decimal fractions.
    Profit factor below uses *trade-return fractions*, not dollar P&L, because
    variable position capital changes their dollar weighting.  Turnover is not
    guessed from the number of trades.  JSON output contains no NaN or infinity.
    """
    periods_per_year, bootstrap_reps, block_size, seed = _settings(
        periods_per_year, bootstrap_reps, block_size, seed
    )
    equity, returns, trades = _path(result)
    n = returns.size
    sample_std = (
        0.0 if _constant(returns) else float(np.std(returns, ddof=1))
    ) if n >= 2 else None
    mean = float(np.mean(returns)) if n else None
    annual_volatility = (
        _finite(sample_std * np.sqrt(periods_per_year))
        if sample_std is not None else None
    )
    sharpe = (
        _finite(mean / sample_std * np.sqrt(periods_per_year))
        if mean is not None and sample_std is not None and sample_std > 0 else None
    )
    annualized_return = None
    if n:
        exponent = (np.log(equity[-1]) - np.log(equity[0])) * periods_per_year / n
        if exponent <= np.log(np.finfo(float).max):
            annualized_return = _finite(np.expm1(exponent))
    positive = trades[trades > 0]
    negative = trades[trades < 0]
    profit_factor = (
        _finite(positive.sum() / -negative.sum()) if negative.size else None
    )
    drawdown, longest_duration, current_duration = _drawdown(equity)
    var95, es95 = _loss_tail(returns)
    inference = _inference(
        returns, bootstrap_reps=bootstrap_reps, block_size=block_size, seed=seed
    )
    return {
        "period_count": int(n),
        "execution_ledger":getattr(result,"ledger",None),
        "trade_count": int(trades.size),
        "net_total_return": float(equity[-1] / equity[0] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annual_volatility,
        "sharpe": sharpe,
        "periods_per_year": periods_per_year,
        "risk_free_return_per_period": 0.0,
        "max_drawdown": drawdown,
        "max_drawdown_duration_bars": longest_duration,
        "current_drawdown_duration_bars": current_duration,
        "win_rate": float(np.mean(trades > 0)) if trades.size else None,
        "mean_trade_return": float(np.mean(trades)) if trades.size else None,
        "return_based_profit_factor": profit_factor,
        "loss_var95": var95,
        "loss_cvar95": es95,
        "period_return_inference": inference,
        "assumptions": [
            "Period returns include the equity path's execution costs; annualization is supplied by the caller.",
            "Sharpe uses sample standard deviation and zero risk-free return; it is not paper mSR.",
            "Trade-return profit factor weights fractional trade returns rather than dollar profits.",
            "Confidence intervals use the stationary block bootstrap; HAC p-values use an asymptotic normal test.",
            "These intervals and p-values do not correct for strategy selection, overlapping experiments, or multiple testing.",
        ],
    }


def paired_comparison(
    first: BacktestResult,
    second: BacktestResult,
    *,
    periods_per_year: float,
    bootstrap_reps: int = 500,
    block_size: float = 12,
    seed: int = 0,
) -> dict[str, Any]:
    """Infer the mean aligned period-return difference, first minus second.

    Pairing preserves contemporaneous market shocks.  Callers must supply the
    same chronological observation periods.  Nonempty results must include one
    timestamp per period, and timestamp equality is checked before pairing.
    """
    periods_per_year, bootstrap_reps, block_size, seed = _settings(
        periods_per_year, bootstrap_reps, block_size, seed
    )
    _, first_returns, _ = _path(first)
    _, second_returns, _ = _path(second)
    if first_returns.shape != second_returns.shape:
        raise ValueError("paired comparisons require equal-length aligned periods")
    first_times = np.asarray(getattr(first, "timestamps", None))
    second_times = np.asarray(getattr(second, "timestamps", None))
    expected_shape = (first_returns.size,)
    if first_times.shape != expected_shape or second_times.shape != expected_shape:
        raise ValueError("paired comparisons require one timestamp for every period")
    if not np.array_equal(first_times, second_times):
        raise ValueError("paired comparisons require identical timestamps")
    try:
        chronological = np.all(first_times[1:] > first_times[:-1])
    except TypeError as error:
        raise ValueError("paired timestamps must be comparable and chronological") from error
    if not chronological:
        raise ValueError("paired timestamps must be strictly increasing")
    inference = _inference(
        first_returns - second_returns,
        bootstrap_reps=bootstrap_reps,
        block_size=block_size,
        seed=seed,
    )
    return {
        "difference": "first minus second simple period returns",
        "period_count": int(first_returns.size),
        "periods_per_year": periods_per_year,
        "paired_return_difference_inference": inference,
        "alignment": "Identical result timestamps",
        "multiplicity_adjusted": False,
        "interpretation": "Held-out, same-period comparison; this test does not account for adaptive strategy search or multiple comparisons.",
    }


def holm_adjust(pvalues: Iterable[float | None]) -> list[float | None]:
    """Holm family-wise adjustment over the supplied finite p-values.

    Missing tests remain missing and are excluded from this *declared* family.
    This helper does not discover unreported trials or correct adaptive search.
    """
    values = list(pvalues)
    observed: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("p-values must be finite values in [0, 1] or None")
        observed.append((index, float(value)))
    ordered = sorted(observed, key=lambda pair: pair[1])
    adjusted: list[float | None] = [None] * len(values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * value)
        adjusted[index] = min(1.0, running)
    return adjusted
