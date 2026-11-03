"""Long-only, next-open, cost-aware simulator shared by the research methods."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import validate_bars


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 10_000.0
    fee: float = 0.0025
    slippage_bps: float = 0.0
    max_holding_days: float | None = None
    take_profit: float | None = None
    entry_hours: tuple[int, ...] | None = None

    def __post_init__(self):
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if not np.isfinite(self.fee) or not 0 <= self.fee < 1:
            raise ValueError("fee must be in [0,1)")
        if not np.isfinite(self.slippage_bps) or not 0 <= self.slippage_bps < 10_000:
            raise ValueError("slippage_bps must be in [0,10000)")
        for name in ("max_holding_days", "take_profit"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive or None")
        if self.entry_hours is not None and (
            not self.entry_hours or any(type(x) is not int or not 0 <= x <= 23 for x in self.entry_hours)
        ):
            raise ValueError("entry_hours must contain UTC integers in 0..23")


@dataclass(frozen=True)
class Trade:
    entry_index: int
    exit_index: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    return_fraction: float
    forced_at_end: bool
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray
    trade_returns: np.ndarray
    trades: tuple[Trade, ...]
    config: ExecutionConfig
    timestamps: tuple[pd.Timestamp, ...] = ()

    @property
    def total_return(self) -> float:
        return float(self.equity[-1] / self.config.initial_cash - 1.0)

    @property
    def paper_total_return(self) -> float:
        """MOO3 Eq.3 unit-price aggregate, percent; NOT compounded equity return."""
        if not self.trades:
            return 0.0
        c = self.config.fee
        profit = sum(t.exit_price * (1 - c) - t.entry_price * (1 + c) for t in self.trades)
        return 100.0 * profit / (self.trades[0].entry_price * (1 + c))


def run_backtest(
    bars: pd.DataFrame, signals: np.ndarray, config: ExecutionConfig | None = None
) -> BacktestResult:
    """Signals at candle completion execute at the NEXT available bar open.

    BUY=1, HOLD=0, SELL=-1. No shorting/pyramiding. Full cash allocation with
    fractional units. Holding deadlines use elapsed calendar time, not bar count.
    Profit targets are observed at closes and exit next open, never filled at a
    conveniently assumed intrabar price. Terminal positions are explicitly
    liquidated at the final observed close, with exit costs. Final liquidation
    is a sample-boundary convention, not a claim of an executable earlier signal.
    """
    config = config or ExecutionConfig()
    frame = validate_bars(bars)
    signals = np.asarray(signals)
    if signals.shape != (len(frame),) or not np.isin(signals, [-1, 0, 1]).all():
        raise ValueError("signals must be a length-n vector containing only -1,0,1")
    opens = frame["open"].to_numpy()
    closes = frame["close"].to_numpy()
    times = frame["timestamp"].tolist()
    known = frame["available_at"].tolist()
    slip = config.slippage_bps / 10_000.0
    cash = config.initial_cash
    units = 0.0
    entry_index = -1
    entry_price = 0.0
    entry_cost = 0.0
    pending_exit = False
    equity = np.empty(len(frame) + 1, dtype=float)
    equity[0] = cash
    trades: list[Trade] = []

    def sell(index: int, price: float, time: pd.Timestamp, reason: str, forced=False):
        nonlocal cash, units
        fill = float(price * (1 - slip))
        cash = units * fill * (1 - config.fee)
        trade_return = fill * (1 - config.fee) / entry_cost - 1.0
        trades.append(Trade(entry_index, index, times[entry_index], time, entry_price, fill,
                            float(trade_return), forced, reason))
        units = 0.0

    for i in range(len(frame)):
        signal = int(signals[i - 1]) if i > 0 and known[i - 1] <= times[i] else 0
        deadline = bool(units and config.max_holding_days is not None and
                        times[i] >= times[entry_index] + pd.Timedelta(days=config.max_holding_days))
        exited = False
        if units and (deadline or pending_exit or signal == -1):
            reason = "holding_deadline" if deadline else "profit_target" if pending_exit else "signal"
            sell(i, opens[i], times[i], reason)
            pending_exit = False
            exited = True
        if not units and signal == 1 and not exited:
            allowed = config.entry_hours is None or times[i].hour in config.entry_hours
            if allowed:
                entry_index = i
                entry_price = float(opens[i] * (1 + slip))
                entry_cost = entry_price * (1 + config.fee)
                units = cash / entry_cost
                cash = 0.0
        if units and config.take_profit is not None:
            pending_exit = closes[i] >= entry_price * (1 + config.take_profit)
        equity[i + 1] = cash + units * closes[i]
    if units:
        sell(len(frame) - 1, closes[-1], known[-1], "sample_end", forced=True)
        equity[-1] = cash
    return BacktestResult(equity, np.asarray([t.return_fraction for t in trades]), tuple(trades), config,
                          tuple(known))
