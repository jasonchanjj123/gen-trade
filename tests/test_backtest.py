"""Execution and accounting invariants independent of optimizer fitness."""
import numpy as np
import pandas as pd
import pytest

from gen_trade.backtest import ExecutionConfig, run_backtest


def candles(opens, closes=None, timestamps=None):
    opens = np.asarray(opens, dtype=float)
    closes = opens.copy() if closes is None else np.asarray(closes, dtype=float)
    timestamps = pd.date_range("2024-01-01 09:00", periods=len(opens), freq="D", tz="UTC") if timestamps is None else pd.DatetimeIndex(timestamps)
    return pd.DataFrame({
        "timestamp": timestamps, "available_at": timestamps + pd.Timedelta(hours=1),
        "open": opens, "high": np.maximum(opens, closes) * 1.001,
        "low": np.minimum(opens, closes) * 0.999, "close": closes,
        "volume": 100., "symbol": "TEST",
    })


def test_signals_execute_next_open_and_last_signal_has_no_fill():
    bars = candles([100, 200, 300, 400])
    result = run_backtest(bars, np.array([1, -1, 0, 1]), ExecutionConfig(fee=0))
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert (trade.entry_index, trade.exit_index) == (1, 2)
    assert (trade.entry_price, trade.exit_price) == (200, 300)
    assert trade.return_fraction == pytest.approx(0.5)
    assert result.total_return == pytest.approx(0.5)
    assert not trade.forced_at_end
    assert trade.reason == "signal"


def test_commission_and_adverse_slippage_apply_on_both_sides():
    config = ExecutionConfig(initial_cash=12_345, fee=0.01, slippage_bps=100)
    result = run_backtest(candles([100, 110, 120, 130]), [1, 0, -1, 0], config)
    trade = result.trades[0]
    expected_entry = 110 * 1.01
    expected_exit = 130 * 0.99
    expected_growth = expected_exit * 0.99 / (expected_entry * 1.01)
    assert trade.entry_price == pytest.approx(expected_entry)
    assert trade.exit_price == pytest.approx(expected_exit)
    assert trade.return_fraction == pytest.approx(expected_growth - 1)
    assert result.equity[-1] == pytest.approx(12_345 * expected_growth)
    assert result.paper_total_return == pytest.approx(100 * (expected_growth - 1))


def test_long_only_no_pyramiding_and_no_exit_reentry_on_same_open():
    result = run_backtest(candles([100, 90, 100, 110, 120]), [-1, 1, 1, -1, 1], ExecutionConfig(fee=0))
    assert len(result.trades) == 1
    assert result.trades[0].entry_index == 2
    assert result.trades[0].exit_index == 4
    assert result.total_return == pytest.approx(0.2)
    assert result.equity[1] == result.equity[2] == 10_000
    flat = run_backtest(candles([100, 90, 80]), [-1, -1, -1], ExecutionConfig(fee=0))
    assert not flat.trades
    assert (flat.equity == 10_000).all()


def test_holding_deadline_is_elapsed_calendar_time():
    times = pd.to_datetime(["2024-01-01 09:00Z", "2024-01-02 09:00Z", "2024-01-02 14:00Z", "2024-01-02 16:00Z", "2024-01-03 09:00Z"])
    result = run_backtest(candles([100, 100, 110, 120, 130], timestamps=times), [1, 0, 0, 0, 0], ExecutionConfig(fee=0, max_holding_days=0.25))
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert (trade.entry_index, trade.exit_index) == (1, 3)
    assert trade.reason == "holding_deadline"
    assert trade.exit_time - trade.entry_time == pd.Timedelta(hours=7)


def test_forced_deadline_exit_does_not_reenter_on_same_open_buy_signal():
    result = run_backtest(candles([100, 100, 110]), [1, 1, 0], ExecutionConfig(fee=0, max_holding_days=1))
    assert len(result.trades) == 1
    assert (result.trades[0].entry_index, result.trades[0].exit_index) == (1, 2)
    assert result.trades[0].reason == "holding_deadline"
    assert not result.trades[0].forced_at_end


def test_profit_target_uses_observed_close_then_next_open_not_target_price():
    result = run_backtest(candles([100, 100, 90, 95], closes=[100, 110, 90, 95]), [1, 0, 0, 0], ExecutionConfig(fee=0, take_profit=0.05))
    trade = result.trades[0]
    assert (trade.entry_index, trade.exit_index) == (1, 2)
    assert trade.exit_price == 90
    assert trade.return_fraction == pytest.approx(-0.1)
    assert trade.reason == "profit_target"
    bars = candles([100, 100, 101], closes=[100, 101, 101])
    bars.loc[1, "high"] = 150  # Crossing only an intrabar high is not an assumed fill.
    held = run_backtest(bars, [1, 0, 0], ExecutionConfig(fee=0, take_profit=0.05))
    assert held.trades[0].reason == "sample_end"


def test_terminal_liquidation_uses_final_close_and_records_boundary_convention():
    bars = candles([100, 110, 120], closes=[100, 110, 125])
    result = run_backtest(bars, [0, 1, -1], ExecutionConfig(fee=0.01))
    trade = result.trades[0]
    assert (trade.entry_index, trade.exit_index) == (2, 2)
    assert trade.exit_time == bars.available_at.iloc[-1]
    assert trade.exit_price == 125
    assert trade.forced_at_end
    assert trade.reason == "sample_end"
    assert result.total_return == pytest.approx(125 * 0.99 / (120 * 1.01) - 1)


def test_paper_total_return_is_distinct_from_all_capital_compounding():
    result = run_backtest(candles([100, 100, 110, 200, 240, 260]), [1, -1, 1, -1, 0, 0], ExecutionConfig(fee=0))
    np.testing.assert_allclose(result.trade_returns, [0.1, 0.2])
    assert result.total_return == pytest.approx(0.32)
    assert result.paper_total_return == pytest.approx(50.0)


def test_entry_hours_apply_to_utc_execution_time():
    times = pd.date_range("2024-01-01 09:00", periods=4, freq="h", tz="UTC")
    bars = candles([100, 110, 120, 130], timestamps=times)
    # Candles close precisely when the next candle opens.
    allowed = run_backtest(bars, [1, 1, -1, 0], ExecutionConfig(fee=0, entry_hours=(10,)))
    assert allowed.trades[0].entry_time.hour == 10
    denied = run_backtest(bars, [1, 1, -1, 0], ExecutionConfig(fee=0, entry_hours=(9,)))
    assert not denied.trades


@pytest.mark.parametrize("signals", [[1], [1, 0, 0, 0], [1, np.nan, 0], [1, 2, 0], [[1, 0, 0]]])
def test_signal_shape_and_alphabet_validation(signals):
    with pytest.raises(ValueError):
        run_backtest(candles([100, 110, 120]), signals)


@pytest.mark.parametrize("kwargs", [
    {"initial_cash":0}, {"initial_cash":np.inf}, {"fee":-0.01}, {"fee":1},
    {"slippage_bps":-1}, {"slippage_bps":10_000}, {"take_profit":-.01},
    {"stop_loss":-.01}, {"stop_loss":1}, {"stop_loss":np.nan},
    {"max_holding_days":0}, {"entry_hours":()}, {"entry_hours":(24,)}, {"entry_hours":(True,)},
])
def test_invalid_execution_parameters(kwargs):
    with pytest.raises(ValueError):
        ExecutionConfig(**kwargs)


def test_close_observed_stop_fills_next_open_and_records_reason():
    bars = candles([100,100,92,103], closes=[100,94,98,103])
    result = run_backtest(bars,[1,0,0,0],ExecutionConfig(fee=0,stop_loss=.05))
    assert result.trades[0].entry_index == 1
    assert result.trades[0].exit_index == 2
    assert result.trades[0].exit_price == 92
    assert result.trades[0].reason == 'stop_loss'
    assert result.trades[0].return_fraction == pytest.approx(-.08)


def test_intrabar_stop_is_not_invented_and_zero_encoded_thresholds_are_valid():
    bars = candles([100,100,100])
    bars['low'] = 50.
    stopped = run_backtest(bars,[1,0,0],ExecutionConfig(fee=0,stop_loss=.05))
    assert stopped.trades[0].reason == 'sample_end'
    zero = run_backtest(bars,[1,0,0],ExecutionConfig(fee=0,take_profit=0,stop_loss=0))
    assert zero.trades[0].exit_index == 2
    assert zero.trades[0].reason == 'stop_loss'
