"""Long-only, next-open, cost-aware simulator shared by the research methods."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .data import validate_bars
from .instruments import InstrumentSpec


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 10_000.0
    fee: float = 0.0025
    slippage_bps: float = 0.0
    max_holding_days: float | None = None
    take_profit: float | None = None
    entry_hours: tuple[int, ...] | None = None
    stop_loss: float | None = None
    instrument: InstrumentSpec | None = None

    def __post_init__(self):
        if self.instrument is not None and not isinstance(self.instrument,InstrumentSpec):
            raise ValueError("instrument must be an InstrumentSpec")
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if not np.isfinite(self.fee) or not 0 <= self.fee < 1:
            raise ValueError("fee must be in [0,1)")
        if not np.isfinite(self.slippage_bps) or not 0 <= self.slippage_bps < 10_000:
            raise ValueError("slippage_bps must be in [0,10000)")
        for name in ("max_holding_days",):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive or None")
        if self.take_profit is not None and (not np.isfinite(self.take_profit) or self.take_profit < 0):
            raise ValueError("take_profit must be finite and nonnegative or None")
        if self.stop_loss is not None and (not np.isfinite(self.stop_loss) or not 0 <= self.stop_loss < 1):
            raise ValueError("stop_loss must be in [0,1) or None")
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
    strategy_id: str | None = None
    allocation_fraction: float = 1.0
    quantity: float | None = None
    exit_quantity: float | None = None
    cash_pnl: float | None = None
    fees: float | None = None
    financing: float | None = None
    dividend_income: float | None = None
    account_currency: str | None = None
    unit_net_profit: float | None = None
    unit_entry_cost: float | None = None


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray
    trade_returns: np.ndarray
    trades: tuple[Trade, ...]
    config: ExecutionConfig
    timestamps: tuple[pd.Timestamp, ...] = ()
    ledger: dict | None = None

    @property
    def total_return(self) -> float:
        return float(self.equity[-1] / self.config.initial_cash - 1.0)

    @property
    def paper_total_return(self) -> float:
        """MOO3 Eq.3 unit-price aggregate, percent; NOT compounded equity return."""
        if not self.trades:
            return 0.0
        if all(t.unit_net_profit is not None for t in self.trades):
            return 100.0 * sum(t.unit_net_profit for t in self.trades) / self.trades[0].unit_entry_cost
        c = self.config.fee
        profit = sum(t.exit_price * (1 - c) - t.entry_price * (1 + c) for t in self.trades)
        return 100.0 * profit / (self.trades[0].entry_price * (1 + c))


@dataclass(frozen=True)
class PortfolioSignals:
    """Independent strategy sleeves with fixed initial capital and a cash reserve.

    Capital is allocated once, with no rebalancing or borrowing between sleeves.
    Signals remain candle-close decisions with next-open execution in each sleeve.
    """

    names: tuple[str, ...]
    signals: np.ndarray
    weights: tuple[float, ...]
    cash_weight: float = 0.0
    take_profit: float | None = None
    stop_loss: float | None = None

    def __post_init__(self):
        values = np.asarray(self.signals)
        weights = np.asarray((*self.weights, self.cash_weight), dtype=float)
        if (not self.names or len(set(self.names)) != len(self.names)
                or len(self.weights) != len(self.names)
                or values.ndim != 2 or values.shape[1] != len(self.names)
                or not np.isin(values, [-1, 0, 1]).all()):
            raise ValueError("Portfolio needs unique names and aligned n-by-strategy signals/weights")
        if (not np.isfinite(weights).all() or (weights < 0).any()
                or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0)):
            raise ValueError("Portfolio allocations including cash must be nonnegative and sum to one")
        ExecutionConfig(take_profit=self.take_profit, stop_loss=self.stop_loss)

    def slice(self, start: int, stop: int | None = None) -> PortfolioSignals:
        return replace(self, signals=self.signals[start:stop])


def run_portfolio_backtest(
    bars: pd.DataFrame, plan: PortfolioSignals, config: ExecutionConfig | None = None,
) -> BacktestResult:
    """Add costed sleeve equity paths; never compound a weighted return average.

    Summing fixed-capital paths also gives the expected equity path across all
    one-member-per-group portfolios when group capital is split equally among
    its members. Trade statistics pool active sleeves without capital weighting;
    each trade retains its sleeve ID and initial allocation for audit.
    """
    config = config or ExecutionConfig()
    overrides = {name:getattr(plan, name) for name in ("take_profit", "stop_loss")
                 if getattr(plan, name) is not None}
    if overrides:
        config = replace(config, **overrides)
    frame = validate_bars(bars)
    if plan.signals.shape[0] != len(frame):
        raise ValueError("Portfolio signals must align with bars")
    equity = np.full(len(frame) + 1, config.initial_cash * plan.cash_weight)
    trades = []
    ledger_totals = {"commissions":0.,"financing":0.,"dividends":0.}
    for i, (name, weight) in enumerate(zip(plan.names, plan.weights)):
        if weight == 0:
            continue
        sleeve_config = replace(config,initial_cash=config.initial_cash*weight) if config.instrument is not None else config
        sleeve = run_backtest(frame, plan.signals[:, i], sleeve_config)
        equity += sleeve.equity if config.instrument is not None else weight * sleeve.equity
        if sleeve.ledger:
            for key in ledger_totals:
                ledger_totals[key] += sleeve.ledger[key]
        trades.extend(replace(trade, strategy_id=name, allocation_fraction=weight)
                      for trade in sleeve.trades)
    trades.sort(key=lambda trade: (trade.exit_time, trade.strategy_id or "", trade.entry_time))
    return BacktestResult(equity, np.asarray([t.return_fraction for t in trades]), tuple(trades),
                          config, tuple(frame["available_at"]),
                          {**ledger_totals,"account_currency":config.instrument.account_currency} if config.instrument else None)


def run_backtest(
    bars: pd.DataFrame, signals: np.ndarray, config: ExecutionConfig | None = None
) -> BacktestResult:
    """Signals at candle completion execute at the NEXT available bar open.

    BUY=1, HOLD=0, SELL=-1. No shorting/pyramiding. Full cash allocation with
    fractional units. Holding deadlines use elapsed calendar time, not bar count.
    Profit/stop thresholds are observed at closes and exit next open, never filled at a
    conveniently assumed intrabar price. Terminal positions are explicitly
    liquidated at the final observed close, with exit costs. Final liquidation
    is a sample-boundary convention, not a claim of an executable earlier signal.
    """
    config = config or ExecutionConfig()
    if config.instrument is not None:
        if np.asarray(signals).shape != (len(bars),):
            raise ValueError("signals must be a length-n decision vector")
        from .market_execution import run_market_backtest
        return run_market_backtest(bars,signals,config)
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
    pending_exit: str | None = None
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
            reason = "holding_deadline" if deadline else pending_exit if pending_exit else "signal"
            sell(i, opens[i], times[i], reason)
            pending_exit = None
            exited = True
        if not units and signal == 1 and not exited:
            allowed = config.entry_hours is None or times[i].hour in config.entry_hours
            if allowed:
                entry_index = i
                entry_price = float(opens[i] * (1 + slip))
                entry_cost = entry_price * (1 + config.fee)
                units = cash / entry_cost
                cash = 0.0
        if units:
            if config.stop_loss is not None and closes[i] <= entry_price * (1 - config.stop_loss):
                pending_exit = "stop_loss"
            elif config.take_profit is not None and closes[i] >= entry_price * (1 + config.take_profit):
                pending_exit = "profit_target"
        equity[i + 1] = cash + units * closes[i]
    if units:
        sell(len(frame) - 1, closes[-1], known[-1], "sample_end", forced=True)
        equity[-1] = cash
    return BacktestResult(equity, np.asarray([t.return_fraction for t in trades]), tuple(trades), config,
                          tuple(known))
