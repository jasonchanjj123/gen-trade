"""Statistical behavior, dependent samples, and defined edge cases."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gen_trade.statistics import detailed_statistics, holm_adjust, paired_comparison


def result(equity, trades=(), timestamps=None):
    equity = np.asarray(equity, dtype=float)
    return SimpleNamespace(
        equity=equity,
        trade_returns=np.asarray(trades, dtype=float),
        trades=tuple(object() for _ in trades),
        timestamps=np.arange(max(0, equity.size - 1)) if timestamps is None else timestamps,
    )


def test_drawdown_duration_and_trade_metrics_are_defined():
    report = detailed_statistics(
        result([100, 120, 90, 96, 120, 108], [0.2, -0.1]),
        periods_per_year=252, bootstrap_reps=100,
    )
    assert report["max_drawdown"] == pytest.approx(0.25)
    assert report["max_drawdown_duration_bars"] == 2
    assert report["current_drawdown_duration_bars"] == 1
    assert report["net_total_return"] == pytest.approx(0.08)
    assert report["win_rate"] == 0.5
    assert report["return_based_profit_factor"] == pytest.approx(2.0)
    assert report["period_return_inference"]["mean_ci95_stationary_bootstrap"] is None
    json.dumps(report, allow_nan=False)


def test_flat_equity_no_trades_has_no_fabricated_sharpe_or_significance():
    report = detailed_statistics(
        result(np.full(30, 100.0)), periods_per_year=365, bootstrap_reps=100,
    )
    assert report["sharpe"] is None
    assert report["annualized_volatility"] == 0.0
    assert report["win_rate"] is None
    assert report["return_based_profit_factor"] is None
    inference = report["period_return_inference"]
    assert inference["hac_pvalue_mean_zero"] is None
    assert inference["mean_ci95_stationary_bootstrap"] is None
    assert "Constant" in inference["inference_note"]
    json.dumps(report, allow_nan=False)


def test_loss_var_and_cvar_weight_empirical_tail():
    returns = np.linspace(-0.1, 0.09, 20)
    equity = 100 * np.r_[1.0, np.cumprod(1 + returns)]
    report = detailed_statistics(
        result(equity), periods_per_year=252, bootstrap_reps=100,
    )
    assert report["loss_var95"] == pytest.approx(0.09)
    assert report["loss_cvar95"] == pytest.approx(0.10)


def test_dependent_series_bootstrap_is_seeded_and_hac_exceeds_iid_error():
    rng = np.random.default_rng(12)
    innovations = rng.normal(0, 0.001, 400)
    returns = np.zeros(400)
    for i in range(1, returns.size):
        returns[i] = 0.85 * returns[i - 1] + innovations[i]
    equity = 100 * np.r_[1.0, np.cumprod(1 + returns)]
    sample = result(equity)
    kwargs = dict(periods_per_year=252, bootstrap_reps=150, block_size=12, seed=18)
    first = detailed_statistics(sample, **kwargs)
    second = detailed_statistics(sample, **kwargs)
    assert first == second
    inference = first["period_return_inference"]
    low, high = inference["mean_ci95_stationary_bootstrap"]
    assert low < inference["mean"] < high
    iid_error = returns.std(ddof=1) / np.sqrt(returns.size)
    assert inference["hac_standard_error"] > 1.5 * iid_error
    json.dumps(first, allow_nan=False)


def test_annualization_is_explicit_and_changes_sharpe_by_square_root():
    equity = [100, 101, 100, 102, 101]
    low = detailed_statistics(result(equity), periods_per_year=4, bootstrap_reps=100)
    high = detailed_statistics(result(equity), periods_per_year=16, bootstrap_reps=100)
    assert high["sharpe"] == pytest.approx(2 * low["sharpe"])
    assert high["annualized_volatility"] == pytest.approx(2 * low["annualized_volatility"])


def test_paired_comparison_uses_aligned_differences_and_checks_timestamps():
    rng = np.random.default_rng(9)
    common = rng.normal(0, 0.01, 100)
    lift = np.resize([0.001, 0.0015, 0.0005, 0.002], 100)
    first = result(100 * np.r_[1, np.cumprod(1 + common + lift)], timestamps=np.arange(100))
    second = result(100 * np.r_[1, np.cumprod(1 + common)], timestamps=np.arange(100))
    report = paired_comparison(first, second, periods_per_year=252, bootstrap_reps=150, seed=4)
    inference = report["paired_return_difference_inference"]
    assert inference["mean"] == pytest.approx(float(lift.mean()))
    assert inference["mean_ci95_stationary_bootstrap"][0] > 0
    with pytest.raises(ValueError, match="identical timestamps"):
        paired_comparison(first, result(second.equity, timestamps=np.arange(100) + 1), periods_per_year=252)
    with pytest.raises(ValueError, match="equal-length"):
        paired_comparison(first, result(second.equity[:-1]), periods_per_year=252)


def test_identical_paired_returns_do_not_report_a_significant_difference():
    first = result([100, 101, 100, 103, 104, 103, 105, 106, 105, 107])
    report = paired_comparison(first, first, periods_per_year=252, bootstrap_reps=100)
    inference = report["paired_return_difference_inference"]
    assert inference["mean"] == 0
    assert inference["hac_pvalue_mean_zero"] is None


def test_constant_growth_does_not_turn_roundoff_into_an_enormous_sharpe():
    sample = result(100 * np.power(1.01, np.arange(30)))
    report = detailed_statistics(sample, periods_per_year=252, bootstrap_reps=100)
    assert report["sharpe"] is None
    assert report["annualized_volatility"] == 0
    assert report["period_return_inference"]["hac_pvalue_mean_zero"] is None


def test_paired_timestamps_cannot_be_missing_or_duplicate():
    sample = result([100, 101, 100, 103])
    missing = result(sample.equity, timestamps=())
    duplicate = result(sample.equity, timestamps=[1, 1, 2])
    with pytest.raises(ValueError, match="every period"):
        paired_comparison(missing, missing, periods_per_year=252)
    with pytest.raises(ValueError, match="strictly increasing"):
        paired_comparison(duplicate, duplicate, periods_per_year=252)


@pytest.mark.parametrize("equity", [[], [100, np.nan], [100, 0], [[100], [101]]])
def test_invalid_equity_is_rejected(equity):
    with pytest.raises(ValueError, match="equity"):
        detailed_statistics(result(equity), periods_per_year=252)


@pytest.mark.parametrize("settings", [
    {"periods_per_year":0}, {"periods_per_year":np.nan},
    {"bootstrap_reps":99}, {"bootstrap_reps":True},
    {"block_size":0.5}, {"block_size":np.inf}, {"seed":-1},
])
def test_invalid_inference_settings_are_rejected(settings):
    kwargs = {"periods_per_year":252, **settings}
    with pytest.raises(ValueError):
        detailed_statistics(result([100, 101]), **kwargs)


def test_holm_adjustment_is_order_preserving_and_handles_missing_tests():
    adjusted = holm_adjust([0.04, 0.001, None, 0.02])
    assert adjusted[2] is None
    assert [adjusted[i] for i in [0, 1, 3]] == pytest.approx([0.04, 0.003, 0.04])
    assert holm_adjust([None]) == [None]
    with pytest.raises(ValueError):
        holm_adjust([1.1])
