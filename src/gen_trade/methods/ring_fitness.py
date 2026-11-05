"""IGTSP monthly-trade equations 6--16 and batched causal trade profiles.

This is distinct from marked-to-market portfolio equity used for common test
statistics. The detailed paper uses absolute monthly wealth drawdowns (Eq.15),
weighted component risks (Eq.13), and unnormalized group entropy (Eq.16).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest import ExecutionConfig
from ..data import validate_bars

INVALID_FITNESS = -1e12


@dataclass(frozen=True)
class PreparedMarket:
    opens: np.ndarray
    closes: np.ndarray
    time_ns: np.ndarray
    known_ns: np.ndarray
    entry_hours: np.ndarray
    open_month: np.ndarray
    final_month: int
    months: tuple[str, ...]
    advanced: object | None = None

    @classmethod
    def from_bars(cls, bars: pd.DataFrame, instrument=None) -> PreparedMarket:
        frame = validate_bars(bars)
        opens = pd.DatetimeIndex(frame["timestamp"])
        known = pd.DatetimeIndex(frame["available_at"])
        first, last = opens[0].year * 12 + opens[0].month - 1, known[-1].year * 12 + known[-1].month - 1
        months = tuple(f"{i//12:04d}-{i%12+1:02d}" for i in range(first, last + 1))
        from ..market_execution import MarketArrays
        advanced = MarketArrays.from_bars(frame,instrument) if instrument else None
        return cls(frame["open"].to_numpy(), frame["close"].to_numpy(), opens.as_unit("ns").asi8,
                   known.as_unit("ns").asi8, opens.hour.to_numpy(),
                   np.asarray(opens.year * 12 + opens.month - 1 - first), last-first, months,advanced)


@dataclass(frozen=True)
class MonthlyProfiles:
    wealth: np.ndarray
    trade_counts: np.ndarray
    mean_returns: np.ndarray
    monthly_trade_counts: np.ndarray
    months: tuple[str, ...]
    # EXTENSION (review 2026-11-05): marked-to-market daily wealth per sleeve
    # (strategies x bars+1, starting at 1) and per-bar invested flags.
    daily_wealth: np.ndarray | None = None
    held: np.ndarray | None = None


def monthly_trade_profiles(market: PreparedMarket, signals: np.ndarray, config: ExecutionConfig,
                           *, initial_cash=None, take_profit=None, stop_loss=None, daily: bool = False) -> MonthlyProfiles:
    """Vectorize strategies while matching run_backtest fills, costs and exits.

    Attribute each closed trade to its UTC exit month, average trade fractions
    within months, set empty months to zero, and compound monthly means (Eq.9).
    ``daily=True`` (extension) also returns marked-to-market daily wealth and
    per-bar invested flags for every strategy.
    """
    signals = np.asarray(signals)
    if config.instrument is not None:
        if market.advanced is None:
            raise ValueError("Instrument execution requires an instrument-prepared market")
        from ..market_execution import execute_batch
        outcome = execute_batch(market.advanced,signals,config,initial_cash=initial_cash,
                                take_profit=take_profit,stop_loss=stop_loss,record_equity=daily,record_exposure=daily)
        wealth = np.column_stack((np.ones(signals.shape[1]),np.cumprod(1+outcome.monthly_returns,axis=1)))
        daily_wealth = held_flags = None
        if daily:
            start = outcome.equity[0]
            daily_wealth = np.divide(outcome.equity, start, out=np.ones_like(outcome.equity), where=start > 0).T
            held_flags = outcome.held.T.astype(float)
        return MonthlyProfiles(wealth,outcome.trade_counts,outcome.monthly_returns,outcome.monthly_counts,market.months,
                               daily_wealth,held_flags)
    if (signals.ndim != 2 or signals.shape[0] != len(market.opens) or not signals.shape[1]
            or not np.isin(signals, [-1, 0, 1]).all()):
        raise ValueError("Trade profiles need an aligned n-by-strategy signal matrix")
    n = signals.shape[1]
    sums = np.zeros((len(market.months), n))
    counts = np.zeros_like(sums, dtype=np.int64)
    held, pending = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    entry_fill, entry_cost = np.ones(n), np.ones(n)
    entry_time = np.zeros(n, dtype=np.int64)
    slip = config.slippage_bps / 10_000
    holding_ns = pd.Timedelta(days=config.max_holding_days).value if config.max_holding_days is not None else None
    capital = np.ones(n)
    daily_wealth = np.ones((n, len(market.opens) + 1)) if daily else None
    held_flags = np.zeros((n, len(market.opens))) if daily else None
    for i in range(len(market.opens)):
        signal = signals[i-1] if i and market.known_ns[i-1] <= market.time_ns[i] else np.zeros(n, dtype=np.int8)
        exits = held & (pending | (signal == -1))
        if holding_ns is not None:
            exits |= held & (market.time_ns[i] >= entry_time + holding_ns)
        if exits.any():
            sums[market.open_month[i], exits] += market.opens[i] * (1-slip) * (1-config.fee) / entry_cost[exits] - 1
            counts[market.open_month[i], exits] += 1
            capital[exits] *= market.opens[i] * (1-slip) * (1-config.fee) / entry_cost[exits]
            held[exits], pending[exits] = False, False
        allowed = config.entry_hours is None or market.entry_hours[i] in config.entry_hours
        if allowed:
            entries = ~held & ~exits & (signal == 1)
            entry_fill[entries] = market.opens[i] * (1+slip)
            entry_cost[entries] = entry_fill[entries] * (1+config.fee)
            entry_time[entries] = market.time_ns[i]
            held[entries] = True
        pending[:] = False
        if config.take_profit is not None:
            pending |= held & (market.closes[i] >= entry_fill * (1+config.take_profit))
        if config.stop_loss is not None:
            pending |= held & (market.closes[i] <= entry_fill * (1-config.stop_loss))
        if daily:
            held_flags[:, i] = held
            daily_wealth[:, i+1] = np.where(held, capital * market.closes[i] * (1-slip) * (1-config.fee) / entry_cost,
                                            capital)
    if held.any():
        sums[market.final_month, held] += market.closes[-1] * (1-slip) * (1-config.fee) / entry_cost[held] - 1
        counts[market.final_month, held] += 1
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    wealth = np.column_stack((np.ones(n), np.cumprod(1+means, axis=0).T))
    return MonthlyProfiles(wealth, counts.sum(axis=0), means.T, counts.T, market.months, daily_wealth, held_flags)


FITNESS_VARIANTS = ("paper", "mixture_drawdown", "mixture_mtm", "blend_excess")


def portfolio_fitness(chromosome, monthly_wealth: np.ndarray, trade_counts: np.ndarray,
                      *, risk_floor: float = 1e-12, ratio_cap: float = 1e9,
                      variant: str = "paper", sign_adjusted: bool = False) -> dict:
    """Paper Eq.6 + Eqs.7--16, exactly aggregated over all group combinations.

    Because Eqs.8 and 13 are linear, group means equal explicit enumeration.
    No mean-portfolio drawdown, log-growth objective or weight entropy is used.
    Zero-risk denominators use an explicit finite floor/cap; eligibility needs
    two completed active-sleeve trades. These degeneracy guards are adaptations.

    EXTENSION ``variant="mixture_drawdown"`` (audit item 4): the paper ratio is
    linear-fractional in the group weights, so its optimum puts all capital on
    one group. The extension measures return and absolute drawdown on the
    capital-weighted MIXTURE of member wealth paths, which is nonlinear and
    rewards low correlation, and replaces membership entropy with entropy of the
    group capital weights. ``sign_adjusted`` applies Israelsen's refinement.
    """
    if variant not in FITNESS_VARIANTS[:3]:
        raise ValueError(f"portfolio_fitness variant must be one of {FITNESS_VARIANTS[:3]}; blend_excess uses portfolio_excess_fitness")
    paths, counts = np.asarray(monthly_wealth, dtype=float), np.asarray(trade_counts)
    if (paths.ndim != 2 or paths.shape[0] != len(chromosome.assignments) or paths.shape[1] < 2
            or counts.shape != (len(chromosome.assignments),)
            or not np.isfinite(paths).all() or (paths <= 0).any()
            or not np.allclose(paths[:, 0], 1., rtol=0, atol=1e-12)
            or not np.isfinite(counts).all() or (counts < 0).any()
            or not np.isfinite(risk_floor) or risk_floor <= 0
            or not np.isfinite(ratio_cap) or ratio_cap <= 0):
        raise ValueError("Fitness requires positive normalized monthly wealth, aligned counts and finite positive guards")
    group_weights = chromosome.group_weights()
    weights = chromosome.allocations()
    if variant == "paper":
        returns = paths[:, -1] - 1
        drawdowns = np.max(np.maximum.accumulate(paths, axis=1) - paths, axis=1)
        group_returns = np.array([np.mean(returns[list(members)]) for members in chromosome.groups])
        group_risks = np.array([np.mean(drawdowns[list(members)]) for members in chromosome.groups])
        portfolio_return, portfolio_risk = float(group_weights @ group_returns), float(group_weights @ group_risks)
        proportions = np.array([len(members) for members in chromosome.groups]) / len(chromosome.assignments)
        balance = float(-np.sum(proportions * np.log(proportions)))
    else:
        mixture = weights @ paths
        portfolio_return = float(mixture[-1] - 1)
        portfolio_risk = float(np.max(np.maximum.accumulate(mixture) - mixture))
        positive = group_weights[group_weights > 0]
        balance = float(-np.sum(positive * np.log(positive)))
    raw = portfolio_return / max(portfolio_risk, risk_floor)
    if sign_adjusted and portfolio_return < 0:
        raw = portfolio_return * max(portfolio_risk, risk_floor)
    ratio = float(np.clip(raw, -ratio_cap, ratio_cap))
    count = int(np.sum(counts[weights > 0]))
    eligible = (count >= 2 and bool(np.isfinite(ratio))
                and (portfolio_risk >= risk_floor or abs(portfolio_return) >= risk_floor))
    return {"fitness":ratio + balance if eligible else INVALID_FITNESS, "eligible":eligible,
            "paper_portfolio_return":portfolio_return, "paper_portfolio_risk":portfolio_risk,
            "group_balance":balance, "return_risk_ratio":ratio,
            "risk_denominator_floored":portfolio_risk < risk_floor,
            "ratio_capped":abs(raw) > ratio_cap,
            "completed_sleeve_trades":count,
            "fitness_equations":{"paper":"6-16","mixture_drawdown":"extension_mixture_drawdown_capital_entropy",
                                 "mixture_mtm":"extension_marked_to_market_daily_mixture_drawdown_capital_entropy"}[variant],
            "fitness_variant":variant, "sign_adjusted":sign_adjusted}


def portfolio_excess_fitness(chromosome, daily_wealth: np.ndarray, held: np.ndarray, trade_counts: np.ndarray,
                             benchmark: np.ndarray, mask: np.ndarray | None, *, periods_per_year: float,
                             weights: np.ndarray | None = None) -> dict:
    """EXTENSION (review 2026-11-05): information ratio of the capital mixture's
    daily return in excess of a cash/buy-and-hold blend with the mixture's own
    mean exposure, on the masked training periods (see robust_fitness)."""
    from .robust_fitness import excess_information_ratio
    allocations = chromosome.allocations() if weights is None else np.asarray(weights, dtype=float)
    mixture = allocations @ np.asarray(daily_wealth, dtype=float)
    returns = np.diff(mixture) / mixture[:-1]
    exposure = allocations @ np.asarray(held, dtype=float)
    count = int(np.sum(np.asarray(trade_counts)[allocations > 0]))
    score = float(excess_information_ratio(returns, exposure, benchmark, mask, periods_per_year=periods_per_year)[0])
    eligible = count >= 2 and bool(np.isfinite(score))
    return {"fitness":score if eligible else INVALID_FITNESS, "eligible":eligible,
            "excess_information_ratio":score, "mean_exposure":float(exposure.mean()),
            "mixture_return":float(mixture[-1] - 1), "completed_sleeve_trades":count,
            "fitness_equations":"extension_excess_over_exposure_matched_blend_information_ratio",
            "fitness_variant":"blend_excess"}


def drawdown_ratio_weights(group_paths: np.ndarray, *, risk_floor: float = 1e-6, max_iterations: int = 50,
                           tolerance: float = 1e-10) -> tuple[np.ndarray | None, dict]:
    """EXTENSION: simplex weights maximizing (M_T - 1) / max drawdown of M = w @ P.

    Absolute drawdown of a linear mixture is a maximum of linear functions, so
    max R(w) - lam * DD(w) is a linear program (running-peak formulation of
    Chekhlov, Uryasev and Zabarankin). Dinkelbach's iteration on lam reaches
    the global maximum of the ratio because R is linear and DD convex.
    ``group_paths`` is groups x (bars + 1). Returns (weights, diagnostics);
    weights are None when no mixture has a positive return.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack
    paths = np.asarray(group_paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] < 2 or not np.isfinite(paths).all():
        raise ValueError("LP weights need a finite groups-by-periods wealth matrix")
    k, n = paths.shape
    gains = paths[:, -1] - 1
    if gains.max() <= 0:
        return None, {"status":"no_positive_return", "iterations":0}

    def drawdown(w):
        mixture = w @ paths
        return max(float(np.max(np.maximum.accumulate(mixture) - mixture)), risk_floor)

    def ratio(w):
        return float(gains @ w) / drawdown(w)

    rows_p, cols_p, values_p = np.repeat(np.arange(n), k), np.tile(np.arange(k), n), paths.T.ravel()
    eye, width = np.arange(n), k + n + 1
    # Variables: w (k), u (n, running peak), z (max drawdown).
    peak = coo_matrix((np.concatenate([values_p, -np.ones(n)]),
                       (np.concatenate([rows_p, eye]), np.concatenate([cols_p, k + eye]))), shape=(n, width))
    monotone = coo_matrix((np.concatenate([np.ones(n - 1), -np.ones(n - 1)]),
                           (np.concatenate([eye[:-1], eye[:-1]]), np.concatenate([k + eye[:-1], k + eye[1:]]))),
                          shape=(n - 1, width))
    depth = coo_matrix((np.concatenate([-values_p, np.ones(n), -np.ones(n)]),
                        (np.concatenate([rows_p, eye, eye]), np.concatenate([cols_p, k + eye, np.full(n, k + n)]))),
                       shape=(n, width))
    a_ub = vstack([peak, monotone, depth]).tocsr()
    b_ub = np.zeros(a_ub.shape[0])
    a_eq = np.zeros((1, width)); a_eq[0, :k] = 1
    bounds = [(0, None)] * k + [(None, None)] * n + [(risk_floor, None)]
    weights = np.full(k, 1 / k)
    lam = max(ratio(weights), 0.0)
    status, iterations = "max_iterations", 0
    for iterations in range(1, max_iterations + 1):
        solution = linprog(np.concatenate([-gains, np.zeros(n), [lam]]), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq,
                           b_eq=[1.0], bounds=bounds, method="highs")
        if solution.status != 0:
            return None, {"status":f"lp_failed: {solution.message}", "iterations":iterations}
        candidate = np.clip(solution.x[:k], 0, None)
        candidate /= candidate.sum()
        if ratio(candidate) >= ratio(weights):
            weights = candidate
        if float(gains @ candidate) - lam * drawdown(candidate) <= tolerance * max(1.0, lam):
            status = "optimal"
            break
        lam = ratio(candidate)
    return weights, {"status":status, "iterations":iterations, "ratio":ratio(weights),
                     "equal_weight_ratio":ratio(np.full(k, 1 / k)),
                     "effective_groups":float(1 / np.sum(weights ** 2))}
