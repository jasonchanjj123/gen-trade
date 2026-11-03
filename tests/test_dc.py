"""Checks that DC confirmation is causal and numerical backends agree."""
from dataclasses import fields

import numpy as np
import pytest

from gen_trade.dc import DCProfile, directional_changes


def assert_profiles_equal(left: DCProfile, right: DCProfile) -> None:
    assert left.threshold == right.threshold
    for field in fields(DCProfile):
        if field.name != "threshold":
            np.testing.assert_array_equal(getattr(left, field.name), getattr(right, field.name))


def test_hand_calculated_confirmation_and_reversal():
    profile = directional_changes(np.array([100, 101, 102, 100, 99, 100, 102.]), 0.01, backend="python")
    np.testing.assert_array_equal(profile.direction, [0, 1, 1, -1, -1, 1, 1])
    np.testing.assert_array_equal(profile.confirmation, [False, True, False, True, False, True, False])
    np.testing.assert_array_equal(profile.anchor_index, [-1, 0, 0, 2, 2, 4, 4])
    np.testing.assert_array_equal(profile.confirmation_index, [-1, 1, 1, 3, 3, 5, 5])
    np.testing.assert_array_equal(profile.extreme_index, [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(profile.has_overshoot, [False, False, True, False, True, False, True])
    np.testing.assert_allclose(profile.anchor_price, [np.nan, 100, 100, 102, 102, 99, 99])
    np.testing.assert_allclose(profile.confirmation_price, [np.nan, 101, 101, 100, 100, 100, 100])


def test_prefixes_do_not_acquire_future_labels():
    prices = np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.02, 120)))
    full = directional_changes(prices, 0.025, backend="python")
    for length in range(len(prices) + 1):
        prefix = directional_changes(prices[:length], 0.025, backend="python")
        for field in fields(DCProfile):
            if field.name != "threshold":
                np.testing.assert_array_equal(getattr(prefix, field.name), getattr(full, field.name)[:length])


def test_initial_extrema_and_plateaus_are_observation_causal():
    p = directional_changes(np.array([100., 100., 99.5, 99.5, 99., 99., 100., 100.]), 0.01, backend="python")
    np.testing.assert_array_equal(p.direction, [0, 0, 0, 0, -1, -1, 1, 1])
    assert p.anchor_index[4] == 0  # Equal highs retain their earliest index.
    assert p.anchor_index[6] == 4  # Equal lows likewise.
    assert not p.has_overshoot.any()


def test_empty_and_constant_inputs():
    assert len(directional_changes(np.array([]), 0.01)) == 0
    p = directional_changes(np.ones(8), 0.01)
    assert len(p) == 8
    assert (p.direction == 0).all()
    assert (p.anchor_index == -1).all()
    assert np.isnan(p.confirmation_price).all()


def test_subprecision_threshold_does_not_invent_changes_on_equal_prices():
    profile = directional_changes(np.ones(5), 1e-20, backend="python")
    assert (profile.direction == 0).all()
    prices = np.array([100., 101., 101., 101., 100., 100.])
    profile = directional_changes(prices, 1e-20, backend="python")
    np.testing.assert_array_equal(profile.direction, [0, 1, 1, 1, -1, -1])
    np.testing.assert_array_equal(profile.confirmation, [False, True, False, False, True, False])


@pytest.mark.parametrize("prices", [[0, 1], [-1, 1], [1, np.nan], [1, np.inf], [[1, 2]], np.array([1 + 1j])])
def test_invalid_prices(prices):
    with pytest.raises(ValueError):
        directional_changes(prices, 0.01)


@pytest.mark.parametrize("threshold", [0, -0.01, 1, 2, np.nan, np.inf])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        directional_changes(np.array([1., 2.]), threshold)


def test_invalid_backend():
    with pytest.raises(ValueError):
        directional_changes(np.ones(2), 0.01, backend="unavailable")


def test_explicit_missing_native_backend(monkeypatch):
    import gen_trade.dc as module
    monkeypatch.setattr(module, "_native_dc_arrays", None)
    with pytest.raises(RuntimeError, match="extension"):
        directional_changes(np.array([1., 2.]), 0.01, backend="rust")
    assert len(directional_changes(np.array([1., 2.]), 0.01, backend="auto")) == 2


def test_native_matches_reference():
    pytest.importorskip("gen_trade_native")
    for prices in [np.array([]), np.ones(10), np.array([100., 101., 102., 100., 99., 100., 102.]),
                   np.exp(np.cumsum(np.random.default_rng(9).normal(0, 0.01, 1000)))]:
        for theta in [1e-20, 0.00098, 0.01, 0.0255]:
            assert_profiles_equal(directional_changes(prices, theta, backend="python"),
                                  directional_changes(prices, theta, backend="rust"))
