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
    from gen_trade.methods import moo3, mstgam

    fits = []

    def factory(method_name):
        class Method:
            def __init__(self, *, population_size, generations, seed, execution):
                self.execution = execution
                self.seed = seed

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
                return np.where(index % 4 == 0, 1, np.where(index % 4 == 2, -1, 0))

            def parameters(self):
                return {"training_close":self.training_close, "fee":self.execution.fee,
                        "slippage_bps":self.execution.slippage_bps, "seed":self.seed}

            @property
            def diagnostics(self):
                return {"training_end":self.training_end}

        return Method

    monkeypatch.setattr(moo3, "MOO3", factory("moo3"))
    monkeypatch.setattr(mstgam, "MSTGAM", factory("mstgam"))
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
                   population_size=4,generations=0,bootstrap_reps=100,synthetic=True)
    before = compare_methods(original, context, **options)
    first_fits = fits[:]
    fits.clear()
    after = compare_methods(modified, context, **options)
    assert before["dataset_sha256"] != after["dataset_sha256"]
    assert len(first_fits) == len(fits) == 2
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
