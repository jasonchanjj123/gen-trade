"""The Rust execution engine must reproduce the Python reference exactly."""
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

import gen_trade.market_execution as execution
from gen_trade.backtest import ExecutionConfig
from gen_trade.instruments import InstrumentSpec, benchmark_spec
from gen_trade.market_execution import MarketArrays, MarketBatch, execute_batch, execution_backend

native = pytest.mark.skipif(execution._native_execute_batch_arrays is None,
                            reason="gen_trade_native with execute_batch_arrays is not installed")


def _clock(rng, n):
    days = np.cumsum(rng.integers(1, 4, n))
    hours = rng.integers(0, 23, n)
    opens = pd.Timestamp("2020-01-02", tz="UTC") + pd.to_timedelta(days, "D") + pd.to_timedelta(hours, "h")
    return opens, opens + pd.Timedelta(hours=1)


def _path(rng, n, start, volatility):
    close = start * np.exp(np.cumsum(rng.normal(0, volatility, n)))
    open_ = np.concatenate(([start], close[:-1])) * np.exp(rng.normal(0, volatility / 4, n))
    return open_, close


def _frame(opens, available, open_, close):
    return pd.DataFrame({"timestamp": opens, "available_at": available, "open": open_,
                         "high": np.maximum(open_, close) * 1.01, "low": np.minimum(open_, close) * 0.99,
                         "close": close, "volume": 1000.0})


def _conversion(rng, frame, n, level):
    bid_open, bid_close = _path(rng, n, level, 0.004)
    for name, bid in (("open", bid_open), ("close", bid_close)):
        frame[f"quote_to_account_bid_{name}"] = bid
        frame[f"quote_to_account_ask_{name}"] = bid * (1 + rng.uniform(0, 0.002, n))


def stock_scenario(rng, n):
    opens, available = _clock(rng, n)
    open_, close = _path(rng, n, rng.uniform(5, 300), rng.uniform(0.005, 0.04))
    frame = _frame(opens, available, open_, close)
    frame["split_ratio"] = np.where(rng.random(n) < 0.03, rng.choice([0.5, 2.0, 3.0], n), 1.0)
    frame["dividend"] = np.where(rng.random(n) < 0.05, rng.uniform(0.01, 2.0, n), 0.0)
    quote = "EUR" if rng.random() < 0.3 else "USD"
    if quote != "USD":
        _conversion(rng, frame, n, 1.1)
    spec = InstrumentSpec(
        "TEST", "stock", quote_currency=quote,
        quantity_step=float(rng.choice([0.0, 1.0, 0.01])), leverage=float(rng.choice([1.0, 1.0, 2.0])),
        allow_short=bool(rng.random() < 0.5), spread_bps=float(rng.uniform(0, 10)),
        commission_bps=float(rng.uniform(0, 5)), commission_per_unit=float(rng.choice([0.0, 0.005])),
        minimum_commission=float(rng.choice([0.0, 1.0])), financing_long=float(rng.uniform(0, 0.05)),
        financing_short=float(rng.uniform(0, 0.05)), debit_interest_rate=float(rng.uniform(0, 0.08)),
        maintenance_margin_fraction=float(rng.uniform(0.1, 0.9)), minimum_spread=float(rng.choice([0.0, 0.01])))
    return frame, spec


def fx_scenario(rng, n):
    opens, available = _clock(rng, n)
    pair = str(rng.choice(["EUR/USD", "USD/JPY", "EUR/GBP"]))
    base, quote = pair.split("/")
    level = {"EUR/USD": 1.1, "USD/JPY": 110.0, "EUR/GBP": 0.85}[pair]
    open_, close = _path(rng, n, level, rng.uniform(0.002, 0.02))
    frame = _frame(opens, available, open_, close)
    for name, mid in (("open", open_), ("high", frame["high"]), ("low", frame["low"]), ("close", close)):
        half = rng.uniform(0, 0.0005, n)
        frame[f"bid_{name}"] = mid * (1 - half)
        frame[f"ask_{name}"] = mid * (1 + half)
    if quote not in (base, "USD") and base != "USD":
        _conversion(rng, frame, n, 1.25)
    if rng.random() < 0.4:
        frame["rollover_long_per_base_unit_quote"] = rng.normal(0, 0.0002, n)
        frame["rollover_short_per_base_unit_quote"] = rng.normal(0, 0.0002, n)
    spec = InstrumentSpec(
        pair, "fx", base_currency=base, quote_currency=quote,
        quantity_step=float(rng.choice([0.0, 1.0, 1000.0])), leverage=float(rng.choice([1.0, 2.0, 20.0])),
        allow_short=bool(rng.random() < 0.6), commission_bps=float(rng.uniform(0, 1)),
        minimum_commission=float(rng.choice([0.0, 2.0])), financing_long=float(rng.uniform(0, 0.04)),
        financing_short=float(rng.uniform(0, 0.04)), debit_interest_rate=float(rng.uniform(0, 0.05)),
        maintenance_margin_fraction=float(rng.uniform(0.2, 1.0)))
    return frame, spec


def config_for(rng, spec):
    return ExecutionConfig(
        initial_cash=float(rng.choice([1_000.0, 100_000.0])), fee=float(rng.uniform(0, 0.003)),
        slippage_bps=float(rng.uniform(0, 5)),
        take_profit=None if rng.random() < 0.5 else float(rng.uniform(0, 0.1)),
        stop_loss=None if rng.random() < 0.5 else float(rng.uniform(0, 0.08)),
        max_holding_days=None if rng.random() < 0.6 else float(rng.uniform(1, 20)),
        entry_hours=None if rng.random() < 0.7 else tuple(sorted(int(h) for h in rng.choice(24, 12, replace=False))),
        instrument=spec)


def assert_batches_equal(native_batch: MarketBatch, python_batch: MarketBatch):
    for field in fields(MarketBatch):
        ours, reference = getattr(native_batch, field.name), getattr(python_batch, field.name)
        if field.name == "trades":
            assert ours == reference
        elif field.name == "months":
            assert ours == reference
        elif reference is None:
            assert ours is None, field.name
        else:
            np.testing.assert_array_equal(ours, reference, err_msg=field.name, strict=True)


def run_both(market, signals, config, **options):
    return (execute_batch(market, signals, config, backend="rust", **options),
            execute_batch(market, signals, config, backend="python", **options))


@native
@pytest.mark.parametrize("seed", range(40))
def test_random_stock_and_fx_scenarios_match_exactly(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 160))
    frame, spec = (stock_scenario if seed % 2 else fx_scenario)(rng, n)
    config = config_for(rng, spec)
    market = MarketArrays.from_bars(frame, spec)
    width = int(rng.choice([1, 5, 40]))
    activity = rng.uniform(0.02, 0.5)
    signals = rng.choice([-1, 0, 1], size=(n, width), p=[activity / 2, 1 - activity, activity / 2])
    options = {}
    if rng.random() < 0.5:
        options["initial_cash"] = rng.uniform(0, 50_000, width)
    if rng.random() < 0.5:
        options["take_profit"] = np.where(rng.random(width) < 0.3, np.nan, rng.uniform(0, 0.1, width))
        options["stop_loss"] = np.where(rng.random(width) < 0.3, np.nan, rng.uniform(0, 0.1, width))
    for flags in ({"record": True, "record_exposure": True}, {"record_equity": True}, {}):
        assert_batches_equal(*run_both(market, signals, config, **options, **flags))


@native
@pytest.mark.parametrize("symbol", ["AAPL", "EUR/USD", "USD/JPY"])
def test_benchmark_contracts_match_exactly(symbol):
    rng = np.random.default_rng(len(symbol))
    n = 300
    frame, _ = (fx_scenario if "/" in symbol else stock_scenario)(rng, n)
    frame = frame.drop(columns=[c for c in frame if c.startswith(("quote_to_account", "rollover"))])
    for stress in (False, True):
        spec = benchmark_spec(symbol, stress=stress, stock_minimum_spread=0.01)
        config = ExecutionConfig(initial_cash=100_000, fee=0.0, slippage_bps=1.0, instrument=spec)
        signals = rng.choice([-1, 0, 1], size=(n, 64), p=[0.1, 0.8, 0.1])
        assert_batches_equal(*run_both(MarketArrays.from_bars(frame, spec), signals, config, record=True))


@native
def test_delayed_availability_and_zero_length_accrual_match_exactly():
    rng = np.random.default_rng(3)
    frame, spec = fx_scenario(rng, 60)
    spec = replace(spec, financing_long=0.03, financing_short=0.02, debit_interest_rate=0.05, allow_short=True)
    market = MarketArrays.from_bars(frame, spec)
    known = market.known_ns.copy()
    known[::4] = market.time_ns[::4]              # available at the open: no intrabar financing
    known[1::7] = market.time_ns[np.minimum(np.arange(1, 60, 7) + 1, 59)] + 1  # later than the next open
    market = replace(market, known_ns=known)
    signals = rng.choice([-1, 0, 1], size=(60, 9))
    config = ExecutionConfig(initial_cash=10_000, fee=0.001, take_profit=0.01, stop_loss=0.01, instrument=spec)
    assert_batches_equal(*run_both(market, signals, config, record=True, record_exposure=True))


@native
def test_empty_strategy_batch_matches():
    rng = np.random.default_rng(5)
    frame, spec = stock_scenario(rng, 20)
    config = ExecutionConfig(instrument=spec)
    assert_batches_equal(*run_both(MarketArrays.from_bars(frame, spec), np.zeros((20, 0), dtype=int), config,
                                   record=True, record_exposure=True))


def test_backend_selection(monkeypatch):
    monkeypatch.setenv(execution.BACKEND_VARIABLE, "python")
    assert execution_backend() == "python"
    assert execution_backend("auto") == ("python" if execution._native_execute_batch_arrays is None else "rust")
    monkeypatch.setenv(execution.BACKEND_VARIABLE, "fast")
    with pytest.raises(ValueError, match="backend"):
        execution_backend()
    monkeypatch.delenv(execution.BACKEND_VARIABLE)
    monkeypatch.setattr(execution, "_native_execute_batch_arrays", None)
    assert execution_backend() == "python"
    with pytest.raises(RuntimeError, match="not installed"):
        execution_backend("rust")
