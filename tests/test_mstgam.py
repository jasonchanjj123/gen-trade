import json

import numpy as np
import pandas as pd
import pytest

from gen_trade.backtest import ExecutionConfig, run_backtest
from gen_trade.methods.mstgam import (
    GENES, INVALID_FITNESS, THRESHOLDS, MSTGAM, _fixed_uniform,
    _profile_features, aggregate_signals, normalise_weights, paper_sharpe,
)


def candles(n=100):
    rng = np.random.default_rng(31)
    # Multiple unequal-amplitude cycles give more than one nonidentical trade.
    x = np.arange(n)
    closes = 100 * np.exp(0.045 * np.sin(x * 0.45) + 0.018 * np.sin(x * 0.12) + rng.normal(0, 0.001, n))
    opens = np.r_[closes[0], closes[:-1]]
    times = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"timestamp": times, "available_at": times + pd.Timedelta(days=1),
                         "open": opens, "high": np.maximum(opens, closes) * 1.001,
                         "low": np.minimum(opens, closes) * 0.999, "close": closes,
                         "volume": 100.0})


def test_paper_gene_grid_and_simplex():
    assert len(GENES) == 70
    assert all(sum(strategy == s for strategy, _ in GENES) == (10 if s <= 6 else 5)
               for s in range(1, 9))
    assert THRESHOLDS == (0.00098, 0.0022, 0.0048, 0.0072, 0.0098,
                          0.0122, 0.0155, 0.0170, 0.0200, 0.0255)
    weights = normalise_weights(np.arange(70.0))
    assert np.isclose(weights.sum(), 1) and (weights >= 0).all()
    np.testing.assert_allclose(normalise_weights(np.zeros(70)), np.full(70, 1 / 70))
    for invalid in [[-1, 1], [np.nan, 1], []]:
        with pytest.raises(ValueError):
            normalise_weights(invalid)


def test_vote_override_boundary_and_ties():
    # Hold dominates unless three genes are active, independent of their weights.
    opinion = np.array([[1, 1, 0, 0], [1, 1, -1, 0], [1, -1, 0, 0]])
    weights = np.array([0.1, 0.1, 0.1, 0.7])
    np.testing.assert_array_equal(aggregate_signals(opinion, weights), [0, 1, 0])
    np.testing.assert_array_equal(aggregate_signals(opinion, weights, min_votes=2), [1, 1, 0])
    assert aggregate_signals(np.array([[1, -1]]), np.array([0.5, 0.5]), 1)[0] == 0


def test_paper_trade_sharpe_has_explicit_units_and_invalid_policy():
    returns = np.array([0.10, -0.03, 0.05])
    expected = (returns.sum() - 0.025) / returns.std(ddof=0)
    assert paper_sharpe(returns) == pytest.approx(expected)
    assert paper_sharpe([]) == INVALID_FITNESS
    assert paper_sharpe([0.1]) == INVALID_FITNESS
    assert paper_sharpe([0.1, 0.1]) == INVALID_FITNESS
    assert paper_sharpe([0.1, np.nan]) == INVALID_FITNESS


def test_seeded_random_draws_are_timestamp_and_prefix_stable():
    times = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC").asi8
    full = _fixed_uniform(times, 7, 2)
    np.testing.assert_array_equal(full[:10], _fixed_uniform(times[:10], 7, 2))
    np.testing.assert_array_equal(full[5:], _fixed_uniform(times[5:], 7, 2))
    assert np.all((full >= 0) & (full < 1))
    assert not np.array_equal(full, _fixed_uniform(times, 8, 2))


def test_dc_indicators_use_theoretical_confirmation_and_calendar_time():
    prices = np.array([100, 105, 110, 109, 104, 100, 99, 98.0])
    times = np.array([0, 1, 2, 3, 4, 10, 20, 30], dtype=float)
    f = _profile_features(prices, times, 0.05)
    # Downturn confirmation at104 from110, so theoretical confirmation=104.5.
    assert f.dc.confirmation[4]
    assert f.osv[5] == pytest.approx(abs((100 - 104.5) / (0.05 * 104.5)))
    assert f.tmv[5] == pytest.approx(abs((100 - 110) / (0.05 * 110)))
    assert f.duration_ratio[5] == pytest.approx((10 - 4) / (4 - 2))
    assert f.st2_ready[5]


def test_three_os_pattern_waits_until_current_extension_is_observed():
    up = np.array([100, 105, 110, 104, 110, 115, 109, 115, 120, 114.0])
    down = np.array([100, 95, 90, 96, 90, 85, 90, 85, 80, 85.0])
    times = np.arange(len(up), dtype=float)
    f_up = _profile_features(up, times, 0.05)
    f_down = _profile_features(down, times, 0.05)
    assert np.flatnonzero(f_up.pattern_up).tolist() == [8]
    assert np.flatnonzero(f_down.pattern_down).tolist() == [8]
    assert not f_up.pattern_up[7]  # Third upturn confirmation is not yet an OS.
    assert not f_down.pattern_down[7]
    assert f_up.dc.confirmation[9] and f_up.dc.direction[9] == -1
    assert f_down.dc.confirmation[9] and f_down.dc.direction[9] == 1


def test_small_ga_is_reproducible_and_prediction_never_refits():
    bars = candles(110)
    kwargs = dict(population_size=8, generations=2, seed=11)
    first, second = MSTGAM(**kwargs).fit(bars), MSTGAM(**kwargs).fit(bars)
    np.testing.assert_array_equal(first.weights_, second.weights_)
    assert first.parameters() == second.parameters()
    assert len(first.weights_) == 70 and np.isclose(first.weights_.sum(), 1)
    assert first.diagnostics["best_closed_trades"] >= 2
    assert first.diagnostics["best_fitness"] > INVALID_FITNESS
    assert first.diagnostics["fitness_cache_hits"] > 0
    assert np.all(np.diff(first.diagnostics["generation_best"]) >= 0)
    before = json.dumps(first.parameters(), sort_keys=True, allow_nan=False)
    future = candles(140)
    predicted = first.predict(future)
    for length in [1, 10, 40, 100, 139]:
        np.testing.assert_array_equal(first.predict(future.iloc[:length]), predicted[:length])
    assert json.dumps(first.parameters(), sort_keys=True, allow_nan=False) == before


def test_pandas_microsecond_and_nanosecond_frames_have_identical_models():
    nano = candles(100)
    micro = nano.copy()
    for field in ("timestamp", "available_at"):
        nano[field] = nano[field].dt.as_unit("ns")
        micro[field] = micro[field].dt.as_unit("us")
    options = dict(population_size=4, generations=0, seed=2)
    left, right = MSTGAM(**options).fit(nano), MSTGAM(**options).fit(micro)
    assert left.parameters() == right.parameters()
    np.testing.assert_array_equal(left.predict(nano), left.predict(micro))


def test_fee_override_is_used_in_shared_fitness_and_execution():
    bars = candles(100)
    model = MSTGAM(population_size=6, generations=1, seed=4,
                   execution=ExecutionConfig(fee=0.0075)).fit(bars)
    result = run_backtest(bars, model.predict(bars), model.execution)
    assert model.diagnostics["best_fitness"] == pytest.approx(paper_sharpe(result.trade_returns))
    assert model.parameters()["execution"]["fee"] == 0.0075
    assert model.parameters()["uses_external_features"] is False


def test_all_cash_training_is_an_explicit_failure():
    bars = candles(30)
    bars[["open", "high", "low", "close"]] = 100.0
    with pytest.raises(ValueError, match="no valid candidate"):
        MSTGAM(population_size=3, generations=1).fit(bars)
    with pytest.raises(RuntimeError, match="Fit"):
        MSTGAM().predict(bars)
