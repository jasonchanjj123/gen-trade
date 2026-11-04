import json

import numpy as np
import pandas as pd
import pytest

from gen_trade.backtest import ExecutionConfig
from gen_trade.methods.indicators import FeatureScaler, dc_features, physical_features, raw_features
from gen_trade.methods.moo3 import MOO3, Node, _Individual, evaluate_tree, modified_sharpe, rank_fronts


def candles(n=100):
    t = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 * np.exp(0.0005 * np.arange(n) + 0.04 * np.sin(np.arange(n) / 2))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp": t, "available_at": t + pd.Timedelta(days=1),
                         "open": opening, "high": np.maximum(opening, close) * 1.005,
                         "low": np.minimum(opening, close) * 0.995, "close": close, "volume": np.arange(n) + 10.0})


def test_all_56_indicators_are_causal_and_missing_stays_missing():
    frame = candles()
    assert len(dc_features(frame, .01)) == 28
    assert len(physical_features(frame)) == 28
    full = raw_features(frame, .01)
    for length in (1, 15, 53):
        prefix = raw_features(frame.iloc[:length], .01)
        for name in full:
            np.testing.assert_allclose(prefix[name], full[name][:length], equal_nan=True, err_msg=name)
    assert np.isnan(full["dc_tmv"][0])
    assert np.isnan(full["ta_ma_50"][:49]).all()


def test_training_only_scaler_does_not_refit_or_backfill():
    scaler = FeatureScaler.fit({"x": np.array([np.nan, 10., 20.]), "constant": np.ones(3)})
    output = scaler.transform({"x": np.array([np.nan, 30., 0.]), "constant": np.array([2., 0., np.nan])})
    np.testing.assert_allclose(output["x"], [np.nan, 1, 0], equal_nan=True)
    np.testing.assert_allclose(output["constant"], [0, 0, np.nan], equal_nan=True)
    assert scaler.bounds["x"] == (10, 20)
    with pytest.raises(ValueError, match="names"):
        scaler.transform({"other": np.ones(3)})


def test_completed_dc_metrics_are_released_only_after_reversing_confirmation():
    frame = candles(4)
    frame["close"] = [100., 102., 105., 103.]
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] + 1
    frame["low"] = frame["close"] - 1
    features = dc_features(frame, .01)
    assert np.isnan(features["dc_tmv"][:3]).all()
    assert features["dc_tmv"][3] == pytest.approx(5.)
    assert features["dc_tdc"][3] == pytest.approx(2.)
    assert features["dc_rdc"][3] == pytest.approx(.025)
    assert features["dc_ndc_10d"][3] == 2
    assert features["dc_cdc_10d"][3] == pytest.approx(5.)


def test_physical_durations_do_not_depend_on_pandas_datetime_unit():
    frame = candles(70)
    microseconds, nanoseconds = frame.copy(), frame.copy()
    for name in ("timestamp", "available_at"):
        microseconds[name] = microseconds[name].dt.as_unit("us")
        nanoseconds[name] = nanoseconds[name].dt.as_unit("ns")
    us_features, ns_features = dc_features(microseconds, .01), dc_features(nanoseconds, .01)
    for name in us_features:
        np.testing.assert_allclose(us_features[name], ns_features[name], equal_nan=True, err_msg=name)


def test_at_clips_sparse_intervals_at_exact_window_boundary():
    frame = candles(5)
    frame["timestamp"] = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta([0, 1, 11, 31, 32], unit="D")
    frame["available_at"] = frame["timestamp"] + pd.Timedelta(days=1)
    frame["close"] = [100., 102., 103., 100., 99.]
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] + 1
    frame["low"] = frame["close"] - 1
    values = dc_features(frame, .01)
    # At offset 31 the last 10 days were all known-uptrend, even though the
    # interval ending there spans 20 days; count only its 10-day intersection.
    assert values["dc_at_10d"][3] == pytest.approx(10.)
    # The final ten-day window contains nine up days and one down day.
    assert values["dc_at_10d"][4] == pytest.approx(8.)
    assert abs(values["dc_at_10d"]).max() <= 10


def test_pareto_selection_respects_risk_direction_and_invalid_eligibility():
    rule = Node("gt", (Node("constant", value=1), Node("constant", value=0)))
    population = [_Individual(rule, (10, .1, .2), True, 5),
                  _Individual(rule, (20, .2, .1), True, 5),
                  _Individual(rule, (30, .3, .4), True, 5),
                  _Individual(rule, (999, 999, 0), False, 1)]
    fronts = rank_fronts(population)
    assert set(fronts[0]) == {1, 2}
    assert fronts[1] == [0]
    assert fronts[2] == [3]


def test_modified_sharpe_exact_front_formula_and_preferences():
    values = np.array([[10., .1, .2], [20., .2, .4]])
    score, bounds = modified_sharpe(values, (.5, .5, 0))
    np.testing.assert_allclose(score, [1., 2.])
    score, _ = modified_sharpe(values, (0, 0, 1))
    np.testing.assert_allclose(score, [1, .5])
    score, _ = modified_sharpe(np.ones((2, 3)))
    np.testing.assert_allclose(score, [1, 1])
    assert bounds["minimum"] == [10, .1, .2]
    with pytest.raises(ValueError, match="summing"):
        modified_sharpe(values, (.33, .33, .33))


def test_boolean_rules_never_buy_on_unknown_comparisons():
    tree = Node("gt", (Node("feature", value="x"), Node("constant", value=.3)))
    np.testing.assert_array_equal(evaluate_tree(tree, {"x": np.array([np.nan, .2, .4])}, 3), [False, False, True])


def test_small_run_is_reproducible_and_prediction_cannot_reselect_from_future():
    frame = candles()
    kwargs = dict(population_size=10, generations=2, seed=7, threshold=.01,
                  execution=ExecutionConfig(fee=.00025, max_holding_days=5, take_profit=.05))
    first, second = MOO3(**kwargs).fit(frame.iloc[:65]), MOO3(**kwargs).fit(frame.iloc[:65])
    assert first.parameters() == second.parameters()
    assert first.diagnostics == second.diagnostics
    before = first.diagnostics
    full = first.predict(frame)
    np.testing.assert_array_equal(full[:45], first.predict(frame.iloc[:45]))
    changed = frame.copy()
    changed.loc[70:, ["open", "high", "low", "close"]] *= 100
    np.testing.assert_array_equal(first.predict(changed)[:70], full[:70])
    assert first.diagnostics == before
    assert set(full).issubset({0, 1})
    json.dumps(first.parameters(), allow_nan=False)
    json.dumps(first.diagnostics, allow_nan=False)
    assert first.diagnostics["feature_count"] == 56


def test_external_terminals_schema_and_unfitted_guard():
    frame = candles(60)
    model = MOO3(population_size=4, generations=0, seed=2)
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(frame)
    model.fit(frame, {"gold_close_lagged": np.arange(60.)})
    assert model.diagnostics["feature_count"] == 57
    with pytest.raises(ValueError, match="names"):
        model.predict(frame)
    with pytest.raises(ValueError, match="length"):
        model.predict(frame, {"gold_close_lagged": np.ones(2)})


def test_no_trade_and_one_trade_samples_fail_clearly():
    with pytest.raises(ValueError, match="two completed trades"):
        MOO3(population_size=4, generations=1).fit(candles(2))
    midday_only = ExecutionConfig(fee=.00025, max_holding_days=5, entry_hours=(12,))
    with pytest.raises(ValueError, match="two completed trades"):
        MOO3(population_size=4, generations=1, execution=midday_only).fit(candles(30))


def test_tree_chromosome_rejects_invalid_types_constants_and_arity():
    number = Node("constant", value=.5)
    boolean = Node("gt", (number, number))
    for constructor in (
        lambda: Node("unknown"), lambda: Node("constant", value=np.nan),
        lambda: Node("constant", value=-.1), lambda: Node("feature", value=""),
        lambda: Node("gt", (boolean, number)), lambda: Node("and", (boolean,)),
    ):
        with pytest.raises(ValueError):
            constructor()


def test_typed_variation_preserves_depth_and_valid_grammar_across_many_parents():
    model = MOO3(population_size=4, generations=0, max_depth=4)
    rng = np.random.default_rng(31)
    names = ("x", "y", "z")
    first = model._tree(rng, names, model.max_depth)
    for _ in range(200):
        second = model._tree(rng, names, model.max_depth)
        first = model._offspring(first, second, rng, names)
        assert first.kind == "boolean"
        assert first.depth <= 4
        assert evaluate_tree(first, {name:np.array([0, .5, 1.]) for name in names}, 3).dtype == bool
