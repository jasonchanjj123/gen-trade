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


def monthly_trade_profiles(market: PreparedMarket, signals: np.ndarray, config: ExecutionConfig,
                           *, initial_cash=None, take_profit=None, stop_loss=None) -> MonthlyProfiles:
    """Vectorize strategies while matching run_backtest fills, costs and exits.

    Attribute each closed trade to its UTC exit month, average trade fractions
    within months, set empty months to zero, and compound monthly means (Eq.9).
    """
    signals = np.asarray(signals)
    if config.instrument is not None:
        if market.advanced is None:
            raise ValueError("Instrument execution requires an instrument-prepared market")
        from ..market_execution import execute_batch
        outcome = execute_batch(market.advanced,signals,config,initial_cash=initial_cash,
                                take_profit=take_profit,stop_loss=stop_loss)
        wealth = np.column_stack((np.ones(signals.shape[1]),np.cumprod(1+outcome.monthly_returns,axis=1)))
        return MonthlyProfiles(wealth,outcome.trade_counts,outcome.monthly_returns,outcome.monthly_counts,market.months)
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
    for i in range(len(market.opens)):
        signal = signals[i-1] if i and market.known_ns[i-1] <= market.time_ns[i] else np.zeros(n, dtype=np.int8)
        exits = held & (pending | (signal == -1))
        if holding_ns is not None:
            exits |= held & (market.time_ns[i] >= entry_time + holding_ns)
        if exits.any():
            sums[market.open_month[i], exits] += market.opens[i] * (1-slip) * (1-config.fee) / entry_cost[exits] - 1
            counts[market.open_month[i], exits] += 1
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
    if held.any():
        sums[market.final_month, held] += market.closes[-1] * (1-slip) * (1-config.fee) / entry_cost[held] - 1
        counts[market.final_month, held] += 1
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    wealth = np.column_stack((np.ones(n), np.cumprod(1+means, axis=0).T))
    return MonthlyProfiles(wealth, counts.sum(axis=0), means.T, counts.T, market.months)


FITNESS_VARIANTS = ("paper", "mixture_drawdown")


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
    if variant not in FITNESS_VARIANTS:
        raise ValueError(f"Ring fitness variant must be one of {FITNESS_VARIANTS}")
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
            "fitness_equations":"6-16" if variant == "paper" else "extension_mixture_drawdown_capital_entropy",
            "fitness_variant":variant, "sign_adjusted":sign_adjusted}
