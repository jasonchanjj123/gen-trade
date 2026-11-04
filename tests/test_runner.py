"""Chronological experiment boundaries and frozen held-out evaluation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gen_trade.data import ChromosomeContext, FeatureGene
from gen_trade.runner import compare_methods, synthetic_markets, walk_forward_folds


def test_walk_forward_bounds_are_chronological_with_nonoverlapping_tests():
    folds = walk_forward_folds(140, 60, 20, embargo_bars=5)
    assert len(folds) == 3
    for fold in folds:
        assert fold.train_stop - fold.train_start == 60
        assert fold.test_start - fold.train_stop == 5
        assert fold.test_stop - fold.test_start == 20
        assert 0 <= fold.train_start < fold.train_stop <= fold.test_start < fold.test_stop <= 140
    assert all(left.test_stop == right.test_start for left, right in zip(folds, folds[1:]))
    assert len(walk_forward_folds(140, 60, 20, embargo_bars=5, max_folds=1)) == 1


@pytest.mark.parametrize("args,kwargs", [
    ((80, 60, 20), {"embargo_bars":1}),
    ((140, 0, 20), {}),
    ((140, 60, 20), {"embargo_bars":-1}),
    ((140, 60, 20), {"max_folds":0}),
    ((True, 60, 20), {}),
])
def test_invalid_or_insufficient_walk_forward_protocol_is_rejected(args, kwargs):
    with pytest.raises(ValueError):
        walk_forward_folds(*args, **kwargs)


def _fake_methods(monkeypatch):
    from gen_trade.methods import moo3, mstgam, igtsp_ring

    fits = []

    def factory(method_name):
        class Method:
            def __init__(self, *, population_size, generations, seed, execution,
                         threshold=.001, min_votes=3, groups=17, **options):
                self.execution = execution
                self.seed = seed
                self.configuration = (threshold, min_votes, groups)

            def fit(self, bars, extra_features=None):
                self.training_end = bars["available_at"].iloc[-1].isoformat()
                self.training_close = float(bars["close"].iloc[-1])
                fits.append({
                    "method":method_name,
                    "bars":bars.copy(deep=True),
                    "features":None if extra_features is None else {
                        key:value.copy() for key,value in extra_features.items()
                    },
                })
                return self

            def predict(self, bars, extra_features=None):
                # A fixed causal schedule exercises fills and closes without
                # letting optimizer stochasticity obscure the split assertions.
                index = np.arange(len(bars))
                alternate = self.configuration != (.001, 3, 17)
                period = 6 if alternate else 4
                return np.where(index % period == 0, 1, np.where(index % period == period // 2, -1, 0))

            def parameters(self):
                return {"training_close":self.training_close, "fee":self.execution.fee,
                        "slippage_bps":self.execution.slippage_bps, "seed":self.seed,
                        "configuration":self.configuration}

            @property
            def diagnostics(self):
                return {"training_end":self.training_end}

        return Method

    monkeypatch.setattr(moo3, "MOO3", factory("moo3"))
    monkeypatch.setattr(mstgam, "MSTGAM", factory("mstgam"))
    monkeypatch.setattr(igtsp_ring, "IGTSPRing", factory("igtsp_ring"))
    return fits


def test_future_price_changes_cannot_change_training_inputs_or_frozen_parameters(monkeypatch):
    fits = _fake_methods(monkeypatch)
    original = synthetic_markets(120, seed=8)
    modified = {symbol:frame.copy(deep=True) for symbol,frame in original.items()}
    for frame in modified.values():
        frame.loc[60:, ["open", "high", "low", "close"]] *= 3.0
    context = ChromosomeContext(
        pair="BTC/USDT", features=(FeatureGene("XAU/USD", lag=1, window=2, max_age="4D"),)
    )
    options = dict(train_bars=60,test_bars=20,embargo_bars=5,max_folds=1,
                   population_size=4,generations=0,bootstrap_reps=100,synthetic=True,
                   validation_bars=0,workers=1)
    before = compare_methods(original, context, **options)
    first_fits = fits[:]
    fits.clear()
    after = compare_methods(modified, context, **options)
    assert before["dataset_sha256"] != after["dataset_sha256"]
    assert len(first_fits) == len(fits) == 3
    for first, second in zip(first_fits, fits):
        assert len(first["bars"]) == len(second["bars"]) == 60
        pd.testing.assert_frame_equal(first["bars"], second["bars"])
        assert first["bars"]["available_at"].iloc[-1] < original["BTC/USDT"]["timestamp"].iloc[65]
        if first["method"] == "moo3":
            assert first["features"]
            for key in first["features"]:
                np.testing.assert_allclose(first["features"][key], second["features"][key], equal_nan=True)
        else:
            assert first["features"] is None
        name = first["method"]
        assert before["folds"][0]["methods"][name]["parameters"] == after["folds"][0]["methods"][name]["parameters"]


def test_comparison_records_identical_costs_and_inference_limits(monkeypatch):
    _fake_methods(monkeypatch)
    report = compare_methods(
        synthetic_markets(120), ChromosomeContext("BTC/USDT", entry_hours=(9,)),
        train_bars=60,test_bars=20,embargo_bars=5,max_folds=1,
        validation_bars=0,workers=1,
        population_size=4,generations=0,fee=0.001,slippage_bps=3,
        periods_per_year=252,bootstrap_reps=100,synthetic=True,
    )
    assert report["data_kind"] == "synthetic test fixture"
    assert report["protocol"]["folds_start_flat"] is True
    assert report["protocol"]["common_fee_per_side"] == 0.001
    for method in report["folds"][0]["methods"].values():
        assert method["parameters"]["fee"] == 0.001
        assert method["parameters"]["slippage_bps"] == 3
        assert method["statistics"]["period_count"] == 20
    paired = report["folds"][0]["paired_moo3_minus_mstgam"]
    assert paired["alignment"] == "Identical result timestamps"
    assert paired["multiplicity_adjusted"] is False
    assert report["limitations"]
    json.dumps(report, allow_nan=False)


def test_validation_bounds_have_two_embargoes_and_nonoverlapping_tests():
    folds = walk_forward_folds(200, 60, 20, validation_bars=30, embargo_bars=3)
    assert folds[0].validation_start == 63
    assert folds[0].validation_stop == 93
    assert folds[0].test_start == 96
    assert folds[0].test_stop == 116
    assert all(a.test_stop == b.test_start for a, b in zip(folds, folds[1:]))
    for fold in folds:
        assert fold.validation_start - fold.train_stop == 3
        assert fold.test_start - fold.validation_stop == 3
        assert fold.validation_stop - fold.validation_start == 30


def test_validation_selection_and_refit_never_see_held_out_prices(monkeypatch):
    fits = _fake_methods(monkeypatch)
    markets = synthetic_markets(160, seed=5)
    options = dict(train_bars=60, validation_bars=30, test_bars=20, embargo_bars=3,
                   population_size=4, generations=0, max_folds=1, workers=1, bootstrap_reps=100)
    before = compare_methods(markets, ChromosomeContext("BTC/USDT"), **options)
    first_fits = fits[:]
    fits.clear()
    changed = {symbol:frame.copy(deep=True) for symbol, frame in markets.items()}
    for frame in changed.values():
        frame.loc[96:, ["open", "high", "low", "close"]] *= 17
    after = compare_methods(changed, ChromosomeContext("BTC/USDT"), **options)
    assert len(first_fits) == len(fits) == 9  # two train-only trials and one fresh refit per method
    for name in ("moo3", "mstgam", "igtsp_ring"):
        method_fits = [fit for fit in fits if fit["method"] == name]
        assert [len(fit["bars"]) for fit in method_fits] == [60, 60, 93]
        first = before["folds"][0]["methods"][name]
        second = after["folds"][0]["methods"][name]
        assert first["parameters"] == second["parameters"]
        assert first["selection"] == second["selection"]
        assert first["selection"]["test_used_for_selection"] is False
        scores = [trial["validation_score"] for trial in first["selection"]["trials"]]
        assert len(set(scores)) == 2
        assert first["selection"]["selected_candidate"] == int(np.argmax(scores))
    assert len(before["folds"][0]["paired_comparisons"]) == 3
    for first, second in zip(first_fits, fits):
        pd.testing.assert_frame_equal(first["bars"], second["bars"])


def test_every_requested_seed_is_reported_without_test_selection(monkeypatch):
    _fake_methods(monkeypatch)
    report = compare_methods(synthetic_markets(120), ChromosomeContext("BTC/USDT"),
                             train_bars=60, validation_bars=0, test_bars=20, max_folds=1,
                             workers=1, population_size=4, generations=0, bootstrap_reps=100,
                             seeds=(3, 7))
    assert [fold["base_seed"] for fold in report["folds"]] == [3, 7]
    assert report["seeds"] == [3, 7]
    assert all(len(fold["methods"]) == 3 for fold in report["folds"])


@pytest.mark.parametrize("settings", [{"workers":0}, {"workers":2,"island_workers":2},
    {"seeds":(1,1)}, {"method_configs":{"other":[{}]}},
    {"validation_bars":0,"method_configs":{"moo3":[{},{}]}},
    {"method_configs":{"moo3":[{"seed":2}]}}])
def test_invalid_parallelism_and_candidate_protocols_fail_before_fitting(settings):
    with pytest.raises(ValueError):
        compare_methods(synthetic_markets(120), ChromosomeContext("BTC/USDT"),
                        train_bars=60, test_bars=20, **settings)


def test_real_three_method_process_experiment_matches_serial():
    markets = synthetic_markets(120, seed=22)
    options = dict(train_bars=65, validation_bars=0, test_bars=25, max_folds=1,
                   population_size=8, generations=1, seed=12, bootstrap_reps=100)
    serial = compare_methods(markets, ChromosomeContext("BTC/USDT"), workers=1, **options)
    parallel = compare_methods(markets, ChromosomeContext("BTC/USDT"), workers=3, **options)
    first, second = serial["folds"][0], parallel["folds"][0]
    assert set(first["methods"]) == {"moo3", "mstgam", "igtsp_ring"}
    assert len({method["worker_pid"] for method in second["methods"].values()}) >= 2
    for name in first["methods"]:
        for field in ("parameters", "training", "selection", "statistics"):
            assert first["methods"][name][field] == second["methods"][name][field]
    assert first["paired_comparisons"] == second["paired_comparisons"]
