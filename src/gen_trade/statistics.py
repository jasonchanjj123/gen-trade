"""Dependent-data statistics for held-out, net-of-cost equity paths.

The ordinary equity-return Sharpe reported here is distinct from MOO3's
population-normalized modified Sharpe selection score.  No inference function
corrects for the adaptive search that produced a strategy.  Evaluate untouched
data and retain all trials before interpreting an interval or p-value.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

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


# ---------------------------------------------------------------------------
# EXTENSIONS (audit item 3 and selection-bias reporting; docs/EXTENSIONS.md).
# None of these corrects the adaptive genetic search itself; they make the
# held-out comparison less confounded and quantify selection effects.

EULER_GAMMA = 0.5772156649015329


def exposure_path(result: BacktestResult) -> np.ndarray:
    """Capital-weighted fraction invested in each equity period.

    Period t is the equity step from mark t to mark t+1 (bar t). A position
    entered at the open of bar e and exited at the open of bar x is counted as
    invested in periods e..x-1; a sample-end liquidation at the final close is
    invested through its exit bar. Portfolio sleeves add their allocation.
    """
    periods = len(np.asarray(result.equity)) - 1
    exposure = np.zeros(max(periods, 0))
    for trade in result.trades:
        stop = trade.exit_index + 1 if trade.forced_at_end else trade.exit_index
        exposure[max(trade.entry_index, 0):min(stop, periods)] += float(trade.allocation_fraction)
    return np.clip(exposure, 0.0, 1.0)


def _period_returns(result: BacktestResult) -> np.ndarray:
    equity = np.asarray(result.equity, dtype=float)
    return np.diff(equity) / equity[:-1]


def alpha_beta(result: BacktestResult, benchmark: BacktestResult, *, periods_per_year: float) -> dict[str, Any]:
    """OLS of strategy period returns on benchmark returns, HAC standard errors."""
    y, x = _period_returns(result), _period_returns(benchmark)
    if y.shape != x.shape:
        raise ValueError("alpha/beta needs aligned periods")
    output = {"alpha_per_period": None, "alpha_annualized": None, "beta": None,
              "alpha_hac_pvalue": None, "r_squared": None, "note": None}
    if y.size < 8 or _constant(x):
        output["note"] = "Too few periods or constant benchmark; regression omitted."
        return output
    if _constant(y):
        output.update(alpha_per_period=float(y[0]), beta=0.0, note="Constant strategy returns (cash).")
        output["alpha_annualized"] = _finite(float(y[0]) * periods_per_year)
        return output
    maxlags = min(y.size - 1, max(1, int(np.floor(4 * (y.size / 100.0) ** (2 / 9)))))
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}, use_t=False)
    alpha, beta = float(fit.params[0]), float(fit.params[1])
    output.update(alpha_per_period=_finite(alpha), alpha_annualized=_finite(alpha * periods_per_year),
                  beta=_finite(beta), alpha_hac_pvalue=_finite(fit.pvalues[0]), r_squared=_finite(fit.rsquared))
    return output


def exposure_matched_blend(benchmark: BacktestResult, exposure: float) -> BacktestResult:
    """Static mix of cash (zero return) and the benchmark at a given exposure."""
    from gen_trade.backtest import BacktestResult as Result
    returns = exposure * _period_returns(benchmark)
    equity = float(benchmark.equity[0]) * np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    return Result(equity, np.asarray([]), (), benchmark.config, tuple(benchmark.timestamps))


def random_interval_signals(trades, periods: int, reps: int, rng: np.random.Generator) -> np.ndarray | None:
    """Signals that replay the same holding durations at random times.

    Holding lengths (in bars) are shuffled and placed with random gaps, keeping
    at least one flat bar between positions. Entry/exit signals are emitted one
    bar early because decisions execute at the next open. Returns n-by-reps.
    """
    if not trades:
        return None
    durations = np.asarray([max(1, t.exit_index - t.entry_index) for t in trades], dtype=int)
    count = len(durations)
    # Entries start at bar >= 1; the last exit must be a signalled open <= n-1.
    free = (periods - 2) - int(durations.sum()) - (count - 1)
    if free < 0:
        return None
    signals = np.zeros((periods, reps), dtype=np.int8)
    for rep in range(reps):
        order = rng.permutation(durations)
        cuts = np.sort(rng.integers(0, free + 1, size=count))
        gaps = np.diff(np.concatenate([[0], cuts]))
        position = 1
        for gap, duration in zip(gaps, order):
            entry = position + int(gap)
            exit_ = entry + int(duration)
            signals[entry - 1, rep] = 1
            if exit_ - 1 < periods - 1:
                signals[exit_ - 1, rep] = -1
            position = exit_ + 1
    return signals


def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark_sharpe: float = 0.0) -> float | None:
    """PSR of Bailey and Lopez de Prado: P(true per-period SR > benchmark)."""
    from scipy.stats import norm, skew, kurtosis
    values = np.asarray(returns, dtype=float)
    if values.size < 8 or _constant(values):
        return None
    sharpe = float(np.mean(values) / np.std(values, ddof=1))
    gamma3, gamma4 = float(skew(values)), float(kurtosis(values, fisher=False))
    variance = 1 - gamma3 * sharpe + (gamma4 - 1) / 4 * sharpe ** 2
    if variance <= 0:
        return None
    return _finite(norm.cdf((sharpe - benchmark_sharpe) * np.sqrt(values.size - 1) / np.sqrt(variance)))


def deflated_sharpe_ratio(returns: np.ndarray, trials: int, sharpe_variance: float) -> dict[str, Any]:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado 2014).

    The benchmark SR0 is the expected maximum per-period Sharpe among ``trials``
    independent unskilled trials whose Sharpe estimates have variance
    ``sharpe_variance``; DSR is the PSR against that SR0.
    """
    from scipy.stats import norm
    if type(trials) is not int or trials < 1 or not np.isfinite(sharpe_variance) or sharpe_variance < 0:
        raise ValueError("DSR needs a positive integer trial count and a finite nonnegative Sharpe variance")
    if trials == 1:
        threshold = 0.0
    else:
        threshold = float(np.sqrt(sharpe_variance) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / trials)
                                                       + EULER_GAMMA * norm.ppf(1 - 1 / (trials * np.e))))
    return {"deflated_sharpe_ratio": probabilistic_sharpe_ratio(returns, threshold),
            "expected_max_null_sharpe_per_period": threshold, "trials": trials,
            "trial_sharpe_variance": float(sharpe_variance)}


def pbo_cscv(matrix: np.ndarray, blocks: int = 8) -> dict[str, Any]:
    """Probability of backtest overfitting by CSCV (Bailey et al. 2017).

    Columns are strategies, rows are aligned periods. For every half/half split
    of ``blocks`` contiguous blocks, the in-sample best column (by Sharpe) is
    ranked out of sample; PBO is the share of splits where that rank falls at or
    below the out-of-sample median (logit <= 0).
    """
    from itertools import combinations
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or blocks < 2 or blocks % 2 or values.shape[0] < 2 * blocks:
        raise ValueError("PBO needs >=2 strategies, an even block count and enough rows")
    parts = np.array_split(np.arange(values.shape[0]), blocks)
    def sharpe(rows):
        sample = values[rows]
        std = sample.std(axis=0, ddof=1)
        return np.divide(sample.mean(axis=0), std, out=np.zeros(values.shape[1]), where=std > 0)
    logits = []
    for chosen in combinations(range(blocks), blocks // 2):
        inside = np.concatenate([parts[i] for i in chosen])
        outside = np.concatenate([parts[i] for i in range(blocks) if i not in chosen])
        best = int(np.argmax(sharpe(inside)))
        oos = sharpe(outside)
        rank = (np.sum(oos < oos[best]) + 0.5 * (np.sum(oos == oos[best]) - 1) + 1) / (values.shape[1] + 1)
        logits.append(float(np.log(rank / (1 - rank))))
    logits = np.asarray(logits)
    return {"pbo": float(np.mean(logits <= 0)), "splits": int(logits.size), "blocks": blocks,
            "strategies": int(values.shape[1]), "median_logit": float(np.median(logits))}


def spa_test(benchmark_returns: np.ndarray, model_returns: np.ndarray, *, block_size: float = 12,
             reps: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Hansen (2005) SPA: H0 = no model beats the benchmark (losses = -returns)."""
    from arch.bootstrap import SPA
    benchmark = np.asarray(benchmark_returns, dtype=float)
    models = np.asarray(model_returns, dtype=float)
    if models.ndim == 1:
        models = models[:, None]
    if benchmark.shape[0] != models.shape[0] or benchmark.shape[0] < 8:
        raise ValueError("SPA needs aligned benchmark/model periods")
    if _constant(benchmark) and all(_constant(models[:, i]) for i in range(models.shape[1])):
        return {"pvalue_consistent": None, "note": "All series constant."}
    test = SPA(-benchmark, -models, block_size=int(max(1, round(block_size))), reps=reps, seed=seed)
    test.compute()
    values = test.pvalues
    return {"pvalue_lower": _finite(values["lower"]), "pvalue_consistent": _finite(values["consistent"]),
            "pvalue_upper": _finite(values["upper"]), "models": int(models.shape[1]), "reps": reps,
            "null": "no model has lower expected loss (higher expected return) than the benchmark"}


# --- EXTENSION (review 2026-11-05): tests on the quantity that measures timing ---

def excess_over_blend(returns: np.ndarray, benchmark_returns: np.ndarray, exposure: float) -> np.ndarray:
    """Per-period excess over a static cash/benchmark mix, ``r_t - e * b_t``.

    With ``e`` the strategy's mean exposure this removes the return a static
    position of the same average size would have earned; with ``e = beta`` it
    is the beta-hedged return whose mean is the regression alpha.
    """
    r, b = np.asarray(returns, dtype=float), np.asarray(benchmark_returns, dtype=float)
    if r.shape != b.shape or not np.isfinite(exposure):
        raise ValueError("Excess returns need aligned periods and a finite exposure")
    return r - float(exposure) * b


def spa_excess(excess: np.ndarray, *, block_size: float = 12, reps: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Hansen SPA on excess returns: H0 = no column has a positive mean excess.

    The benchmark loss is zero, so each model is judged against its own
    exposure-matched blend (or beta hedge) instead of against buy-and-hold,
    which a low-exposure strategy cannot beat in a rising market regardless
    of timing.
    """
    values = np.asarray(excess, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    result = spa_test(np.zeros(values.shape[0]), values, block_size=block_size, reps=reps, seed=seed)
    result["null"] = "no model has a positive expected excess return"
    return result


def stepm_excess(excess: np.ndarray, names: Sequence[str], *, size: float = 0.05, block_size: float = 12,
                 reps: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Romano-Wolf StepM on excess returns: which models have positive mean excess.

    Controls the familywise error rate at ``size`` across the listed models.
    """
    import pandas as pd
    from arch.bootstrap import StepM
    values = np.asarray(excess, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names) or values.shape[0] < 8:
        raise ValueError("StepM needs an aligned periods-by-models excess matrix")
    if all(_constant(values[:, i]) for i in range(values.shape[1])):
        return {"superior_models": [], "size": size, "note": "All series constant."}
    test = StepM(np.zeros(values.shape[0]), pd.DataFrame(-values, columns=list(names)), size=size,
                 block_size=int(max(1, round(block_size))), reps=reps, seed=seed)
    test.compute()
    return {"superior_models": [str(x) for x in test.superior_models], "size": size, "reps": reps,
            "null": "each model has a nonpositive expected excess return (familywise error controlled)"}


def rademacher_antiserum(returns: np.ndarray, *, delta: float = 0.05, reps: int = 2000, seed: int = 0,
                         periods_per_year: float | None = None) -> dict[str, Any]:
    """Rademacher Anti-Serum lower bound on every tested strategy's Sharpe.

    Paleologo (2024), as summarized in secondary sources: with returns
    standardized by each column's sample volatility, the empirical Rademacher
    complexity is R = E_eps[max_n (1/T) sum_t eps_t x_tn], and with
    probability >= 1 - delta, simultaneously for all N columns,

        SR_n >= SR_hat_n - 2R - 3 sqrt(2 ln(2/delta)/T) - sqrt(2 ln(2N/delta)/T).

    Per-period Sharpe ratios. ``2R`` is the data-snooping haircut; a column
    with SR_hat - 2R > 0 is "Rademacher positive". The estimation terms are
    conservative at daily frequency, so the full bound is rarely positive.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 8 or not np.isfinite(values).all():
        raise ValueError("RAS needs a finite periods-by-strategies return matrix")
    if not 0 < delta < 1 or type(reps) is not int or reps < 1:
        raise ValueError("RAS needs delta in (0,1) and a positive integer rep count")
    periods, count = values.shape
    scale = values.std(axis=0, ddof=1)
    standardized = np.divide(values, scale, out=np.zeros_like(values), where=scale > 0)
    sharpe = standardized.mean(axis=0)
    rng = np.random.default_rng(seed)
    total = 0.0
    for start in range(0, reps, 256):
        signs = rng.choice((-1.0, 1.0), size=(min(256, reps - start), periods))
        total += float(np.sum(np.max(signs @ standardized / periods, axis=1)))
    complexity = total / reps
    estimation = 3 * np.sqrt(2 * np.log(2 / delta) / periods) + np.sqrt(2 * np.log(2 * count / delta) / periods)
    snooping_adjusted = sharpe - 2 * complexity
    bound = snooping_adjusted - estimation
    annual = np.sqrt(periods_per_year) if periods_per_year else None
    return {"delta": delta, "periods": periods, "strategies": count, "reps": reps,
            "rademacher_complexity": complexity, "data_snooping_haircut": 2 * complexity,
            "estimation_haircut": float(estimation),
            "sharpe_per_period": sharpe.tolist(), "snooping_adjusted_sharpe_per_period": snooping_adjusted.tolist(),
            "lower_bound_per_period": bound.tolist(),
            "rademacher_positive": (snooping_adjusted > 0).tolist(), "bound_positive": (bound > 0).tolist(),
            "annualized_snooping_adjusted_sharpe": (snooping_adjusted * annual).tolist() if annual else None,
            "source": "Paleologo (2024), The Elements of Quantitative Investing; constants via secondary summaries"}
