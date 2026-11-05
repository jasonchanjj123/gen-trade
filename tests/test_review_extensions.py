"""Tests for the review 2026-11-05 extensions (docs/EXTENSIONS.md, last section).

Each test switches one extension on and checks a defining property; the
paper and improved profiles are covered by the existing suites.
"""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from gen_trade.backtest import ExecutionConfig, PortfolioSignals, run_backtest, run_portfolio_backtest
from gen_trade.benchmark import (PROTOCOLS, check_lockbox, freeze_profiles, pooled_controls, profile_hash,
                                 stitch_excess)
from gen_trade.experiments import (CANDIDATE_PROFILES, IMPROVED_CANDIDATES, ROBUST_CANDIDATES, ExperimentJob,
                                   blend_excess_score, run_trials)
from gen_trade.instruments import InstrumentSpec
from gen_trade.market_execution import MarketArrays, execute_batch, run_market_backtest
from gen_trade.methods.igtsp_ring import IGTSPRing
from gen_trade.methods.moo3 import MOO3, node_count
from gen_trade.methods.mstgam import MSTGAM
from gen_trade.methods.ring_fitness import PreparedMarket, drawdown_ratio_weights, monthly_trade_profiles
from gen_trade.methods.robust_fitness import (SubWindows, bar_months, benchmark_returns, excess_information_ratio,
                                              period_returns)
from gen_trade.runner import protocol_header, random_timing
from gen_trade.statistics import exposure_path, rademacher_antiserum, spa_excess, stepm_excess


SPEC = InstrumentSpec("X", "stock", spread_bps=2)


def candles(n=400, seed=3, start="2019-01-02"):
    rng = np.random.default_rng(seed)
    times = pd.bdate_range(start, periods=n, tz="UTC")
    close = 100 * np.exp(.03 * np.sin(np.arange(n) / 4) + np.cumsum(rng.normal(.0003, .01, n)))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp":times, "available_at":times + pd.Timedelta(hours=8), "open":opening,
                         "high":np.maximum(opening, close) * 1.01, "low":np.minimum(opening, close) * .99,
                         "close":close, "volume":rng.uniform(100, 200, n), "symbol":"X"})


def execution(**kw):
    return ExecutionConfig(initial_cash=100000, fee=0, slippage_bps=1, instrument=SPEC, **kw)


def random_signals(n, columns, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n, columns), p=[.1, .8, .1])


# --- execution recording -----------------------------------------------------

def test_equity_and_exposure_recording_match_full_record():
    bars = candles(200)
    signals = random_signals(len(bars), 5)
    market = MarketArrays.from_bars(bars, SPEC)
    light = execute_batch(market, signals, execution(), record_equity=True, record_exposure=True)
    full = execute_batch(market, signals, execution(), record=True)
    assert np.array_equal(light.equity, full.equity) and np.array_equal(light.terminal_equity, full.terminal_equity)
    for column in range(signals.shape[1]):
        result = run_market_backtest(bars, signals[:, column], execution())
        assert np.array_equal(light.held[:, column].astype(float), exposure_path(result))


# --- test statistics -----------------------------------------------------------

def test_excess_spa_and_stepm_find_only_the_positive_model():
    rng = np.random.default_rng(1)
    excess = rng.normal(0, .01, (1500, 3)) + np.array([.0015, 0, -.001])
    assert spa_excess(excess, reps=300)["pvalue_consistent"] < .05
    assert spa_excess(excess[:, 1:], reps=300)["pvalue_consistent"] > .2
    assert stepm_excess(excess, ["good", "null", "bad"], reps=300)["superior_models"] == ["good"]


def test_rademacher_antiserum_haircut_grows_with_the_number_of_strategies():
    rng = np.random.default_rng(2)
    few = rademacher_antiserum(rng.normal(0, .01, (500, 2)), reps=500)
    many = rademacher_antiserum(rng.normal(0, .01, (500, 200)), reps=500)
    assert many["data_snooping_haircut"] > 3 * few["data_snooping_haircut"]
    strong = rng.normal(0, .01, (500, 20)); strong[:, 0] += .004
    ras = rademacher_antiserum(strong, reps=500)
    assert ras["rademacher_positive"][0] and not any(ras["rademacher_positive"][1:])
    expected = np.array(ras["sharpe_per_period"]) - ras["data_snooping_haircut"] - ras["estimation_haircut"]
    assert np.allclose(ras["lower_bound_per_period"], expected)


def test_portfolio_random_timing_reduces_to_single_position_for_one_full_sleeve():
    bars = candles(260)
    signal = np.zeros(len(bars), dtype=np.int8); signal[[10, 60, 120]] = 1; signal[[30, 90, 150]] = -1
    single = run_market_backtest(bars, signal, execution())
    plan = PortfolioSignals(("only",), signal[:, None], (1.0,))
    portfolio = run_portfolio_backtest(bars, plan, execution())
    a, _ = random_timing(single, bars, execution(), reps=50, seed=5)
    b, _ = random_timing(portfolio, bars, execution(), reps=50, seed=5, portfolio=True)
    assert a["percentile"] == b["percentile"] and np.isclose(a["random_median_net_return"], b["random_median_net_return"])


def test_portfolio_random_timing_keeps_idle_capital_in_cash():
    bars = candles(260)
    signal = np.zeros(len(bars), dtype=np.int8); signal[20] = 1; signal[40] = -1
    plan = PortfolioSignals(("a", "b"), np.column_stack([signal, np.zeros_like(signal)]), (.3, .7))
    result = run_portfolio_backtest(bars, plan, execution())
    timing, note = random_timing(result, bars, execution(), reps=40, seed=1, portfolio=True)
    assert note is None and timing["reps"] == 40
    assert abs(timing["random_median_net_return"]) < .3 * .5  # only the 30% sleeve can move


# --- robust fitness ------------------------------------------------------------

def test_subwindows_are_deterministic_and_half_the_months():
    bars = candles(750)
    windows = SubWindows(bar_months(bars), seed=4)
    assert np.array_equal(windows.mask(3), SubWindows(bar_months(bars), seed=4).mask(3))
    assert not np.array_equal(windows.mask(3), windows.mask(4))
    chosen = np.unique(windows.months[windows.mask(3)])
    assert len(chosen) == windows.size == round(len(windows.unique) / 2)
    assert windows.mask(None).all()


def test_excess_information_ratio_ignores_static_exposure():
    rng = np.random.default_rng(0)
    benchmark = rng.normal(.001, .01, 2000)
    static = np.column_stack([.3 * benchmark, .8 * benchmark])
    exposure = np.column_stack([np.full(2000, .3), np.full(2000, .8)])
    assert np.allclose(excess_information_ratio(static, exposure, benchmark, periods_per_year=252), 0)
    timed = np.where(benchmark > 0, benchmark, 0)
    assert excess_information_ratio(timed, (benchmark > 0).astype(float), benchmark, periods_per_year=252)[0] > 5


def test_mstgam_blend_excess_is_deterministic_and_respects_the_trade_floor():
    bars = candles(500)
    kw = dict(population_size=12, generations=3, seed=2, execution=execution(), fitness="blend_excess",
              mutation="lognormal", min_trades_per_year=3.)
    first, second = MSTGAM(**kw).fit(bars), MSTGAM(**kw).fit(bars)
    assert np.array_equal(first.weights_, second.weights_)
    variant = first.diagnostics["variant"]
    assert first.diagnostics["best_closed_trades"] >= variant["minimum_completed_trades"] >= 2
    assert variant["best_fitness_scope"].startswith("final population")
    assert "selected_excess_sharpe_annualized" in variant["selection_haircut"]


def test_moo3_scalar_objectives_and_complexity_penalty():
    bars = candles(500)
    base = dict(population_size=16, generations=3, seed=1, execution=execution(max_holding_days=5, take_profit=.05),
                feature_transform="scale_free", threshold_volatility_multiple=1.)
    single = MOO3(objectives="single", **base).fit(bars)
    front = single.diagnostics["pareto_front"]
    assert all(row["lcb_annual_log_growth"] == row["lcb_annual_log_growth_copy"] for row in front)
    assert single.diagnostics["variant"]["effective_survival"] == "nsga2"
    lean = MOO3(objectives="blend_excess", complexity_penalty=10., **base).fit(bars)
    assert node_count(lean._selected) == 3  # one comparison of two terminals
    assert "selection_score" in lean.diagnostics["pareto_front"][0]


# --- Ring ----------------------------------------------------------------------

def test_daily_profiles_match_compounded_trades_without_instrument():
    bars = candles(300)
    signals = random_signals(len(bars), 4, seed=3)
    config = ExecutionConfig(initial_cash=1, fee=.001, slippage_bps=1)
    profile = monthly_trade_profiles(PreparedMarket.from_bars(bars), signals, config, daily=True)
    for column in range(signals.shape[1]):
        result = run_backtest(bars, signals[:, column], config)
        assert np.isclose(profile.daily_wealth[column, -1], np.prod(1 + result.trade_returns))


def test_lp_weights_reach_the_global_ratio_maximum():
    rng = np.random.default_rng(3)
    steps = rng.normal(.0004, .01, (4, 250)) + np.array([.0006, .0002, 0, -.0002])[:, None]
    paths = np.column_stack([np.ones(4), np.cumprod(1 + steps, axis=1)])
    weights, info = drawdown_ratio_weights(paths)
    def ratio(w):
        mixture = w @ paths
        return (mixture[-1] - 1) / max(np.max(np.maximum.accumulate(mixture) - mixture), 1e-6)
    sampled = max(ratio(w) for w in rng.dirichlet(np.full(4, .3), size=20000))
    assert info["status"] == "optimal" and np.isclose(weights.sum(), 1) and ratio(weights) >= sampled - 1e-9
    assert drawdown_ratio_weights(np.vstack([np.linspace(1, .9, 50)] * 2))[0] is None


@pytest.mark.parametrize("options", [dict(fitness="mixture_mtm", weight_mode="equal"),
                                     dict(fitness="mixture_mtm", weight_mode="lp"),
                                     dict(fitness="blend_excess", weight_mode="equal")])
def test_ring_review_variants_fit_and_deploy_their_weights(options):
    bars = candles(600, seed=5)
    model = IGTSPRing(population_size=8, generations=2, seed=3, island_count=2, groups=3, strategy_count=6,
                      execution=execution(), **options).fit(bars)
    params = model.parameters()
    weights = np.asarray(params["group_weights"])
    assert np.isclose(weights.sum(), 1)
    if options["weight_mode"] == "equal":
        assert np.allclose(weights, 1 / 3)
    else:
        assert model.diagnostics["variant"]["lp_weights"]["iterations"] >= 0
    plan = model.predict(bars)
    assert np.isclose(sum(plan.weights), 1) and np.allclose(plan.weights, model.allocations())
    assert len(model.generation_snapshots()) >= 2


# --- protocol, validation and lockbox ---------------------------------------------

def test_blend_excess_validation_score_and_protocol_header():
    bars = candles(700)
    job = ExperimentJob("mstgam", bars, {}, 450, 455, 600, 605, 8, 1, 1, execution(),
                        [{"fitness":"blend_excess", "min_votes":2}], min_validation_trades=1,
                        validation_metric="blend_excess")
    trial = run_trials(job)[0]
    assert trial["status"] in ("eligible", "no_validation_trades")
    if trial["status"] == "eligible":
        assert np.isfinite(trial["validation_score"])
    header = dict(train_bars=1, test_bars=1, embargo_bars=0, validation_bars=1, on_no_eligible="cash", fold_plan=None,
                  instrument=None, initial_cash=1, stress_instrument=None, population_size=4, generations=1,
                  periods_per_year=252, fee=0, slippage_bps=0, workers=1, island_workers=1, configs={},
                  selection="pooled", deploy="validated", min_validation_trades=2, validation_curves=False,
                  exposure_controls_enabled=True)
    assert "validation_metric" not in protocol_header(**header)
    assert protocol_header(**header, validation_metric="blend_excess")["validation_metric"] == "blend_excess"
    assert PROTOCOLS["audited2"]["compare"]["validation_metric"] == "blend_excess"


def test_profiles_keep_paper_and_improved_unchanged():
    assert {"paper", "improved", "robust"} <= set(CANDIDATE_PROFILES)
    assert IMPROVED_CANDIDATES["igtsp_ring"][0]["fitness"] == "mixture_drawdown"
    assert all(c["fitness"] == "blend_excess" for c in ROBUST_CANDIDATES["mstgam"] + ROBUST_CANDIDATES["igtsp_ring"])


def test_lockbox_refuses_until_a_frozen_profile_unlocks(tmp_path):
    record = freeze_profiles(tmp_path / "lockbox.json", first_test_year=2026, symbols=["MSFT"], profiles=["robust"])
    lock = json.loads((tmp_path / "lockbox.json").read_text())
    common = dict(method_profile="robust", candidates=ROBUST_CANDIDATES)
    assert check_lockbox(lock, symbols=["AAPL"], last_test_year=2025, unlock=False, **common) == {"lockbox_touched":False}
    with pytest.raises(ValueError, match="held out"):
        check_lockbox(lock, symbols=["MSFT"], last_test_year=2025, unlock=False, **common)
    with pytest.raises(ValueError, match="held out"):
        check_lockbox(lock, symbols=["AAPL"], last_test_year=2026, unlock=False, **common)
    opened = check_lockbox(lock, symbols=["MSFT"], last_test_year=2026, unlock=True, **common)
    assert opened["profile_hash"] == record["frozen_profiles"]["robust"] == profile_hash(ROBUST_CANDIDATES)
    changed = {**ROBUST_CANDIDATES, "mstgam":({"fitness":"tstat"},)}
    with pytest.raises(ValueError, match="changed"):
        check_lockbox(lock, symbols=["MSFT"], last_test_year=2026, unlock=True, method_profile="robust",
                      candidates=changed)


# --- summaries -------------------------------------------------------------------

def _fold(index, seed, year, method_equity, baseline_equity, exposure, beta, percentile=None):
    controls = {"mean_exposure":exposure, "alpha_beta_vs_buy_and_hold":{"beta":beta, "alpha_annualized":.01},
                "minus_exposure_matched_blend":{"paired_return_difference_inference":{"mean":.0001}},
                "random_timing":None if percentile is None else {"percentile":percentile}}
    path = {"equity":list(method_equity), "initial_cash":method_equity[0], "trades":[], "timestamps":[]}
    return {"index":index, "base_seed":seed, "bounds":{"test_start":index * 10}, "test_first_open":f"{year}-01-02",
            "methods":{name:{"backtest":path} for name in ("moo3", "mstgam", "igtsp_ring")},
            "baseline_backtest":{"equity":list(baseline_equity), "initial_cash":baseline_equity[0]},
            "exposure_controls":{name:controls for name in ("moo3", "mstgam", "igtsp_ring")}}


def test_stitched_excess_subtracts_the_folds_blend_and_ensembles_average():
    method = np.array([100., 101, 103, 102]); base = np.array([100., 102, 103, 101])
    report = {"folds":[_fold(0, 1, 2020, method, base, .4, .5), _fold(0, 2, 2020, method * 1.0, base, .6, .7)]}
    r, b = np.diff(method) / method[:-1], np.diff(base) / base[:-1]
    one, years = stitch_excess(report, "moo3", 1)
    assert np.allclose(one, r - .4 * b) and years == ["2020"] * 3
    both, _ = stitch_excess(report, "moo3", [1, 2])
    assert np.allclose(both, r - .5 * b)
    hedged, _ = stitch_excess(report, "moo3", 2, kind="beta")
    assert np.allclose(hedged, r - .7 * b)


def test_pooled_controls_cluster_by_year():
    equity = np.array([100., 101, 102]); folds = []
    for i, year in enumerate((2019, 2020, 2021)):
        folds += [_fold(i, 1, year, equity, equity, .3, .3, .6), _fold(i, 2, year, equity, equity, .3, .3, .8)]
    reports = {"A":{"folds":folds, "periods_per_year_used":252.}, "B":{"folds":folds, "periods_per_year_used":252.}}
    rows = pooled_controls(reports, {"instruments":{}})
    timing = rows[0]["random_timing_percentile"]
    assert rows[0]["instrument_years"] == 6 and timing["cells"] == 6 and timing["years"] == 3
    assert np.isclose(timing["mean"], .7) and timing["above_reference"] == 6


# --- results 2026-11-05: cadence, vote margin, Ring haircut scope -------------------

from gen_trade.methods.cadence import apply_cadence, decision_bars
from gen_trade.methods.mstgam import aggregate_signals
from gen_trade.experiments import ROBUST_MONTHLY_CANDIDATES, ROBUST_WEEKLY_CANDIDATES


def test_decision_bars_open_each_week_and_month():
    times = pd.bdate_range("2024-01-01", periods=70, tz="UTC")
    weekly, monthly = decision_bars(times, "weekly"), decision_bars(times, "monthly")
    assert weekly[0] and all(times[i].weekday() == 0 for i in np.flatnonzero(weekly)[1:])
    assert all(times[i].month != times[i - 1].month for i in np.flatnonzero(monthly)[1:])
    assert decision_bars(times, "daily").all()


def test_cadence_carries_the_latest_signal_and_is_causal():
    times = pd.bdate_range("2024-01-01", periods=30, tz="UTC")
    rng = np.random.default_rng(7)
    signals = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(30, 6), p=[.15, .7, .15])
    out = apply_cadence(signals, times, "weekly")
    decision = decision_bars(times, "weekly")
    assert not out[~decision].any()
    points = np.flatnonzero(decision)
    for previous, current in zip(points[:-1], points[1:]):
        for column in range(signals.shape[1]):
            window = signals[previous + 1:current + 1, column]
            raised = window[window != 0]
            assert out[current, column] == (raised[-1] if raised.size else 0)
    changed = signals.copy(); changed[20:] = -changed[20:]
    assert np.array_equal(apply_cadence(changed, times, "weekly")[:20], out[:20])
    assert np.array_equal(apply_cadence(signals.T, times, "weekly", axis=1), out.T)
    assert apply_cadence(signals, times, "daily") is signals


def test_vote_margin_is_a_hurdle_and_zero_keeps_the_paper_rule():
    rng = np.random.default_rng(0)
    opinions = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(500, 70))
    weights = rng.uniform(size=70)
    paper = aggregate_signals(opinions, weights)
    assert np.array_equal(aggregate_signals(opinions, weights, vote_margin=0.0), paper)
    counts = [np.count_nonzero(aggregate_signals(opinions, weights, vote_margin=m)) for m in (0, .05, .1, .2)]
    assert counts == sorted(counts, reverse=True) and counts[-1] < counts[0]
    hurdle = aggregate_signals(opinions, weights, vote_margin=.1)
    assert np.all((hurdle == 0) | (hurdle == paper))


@pytest.mark.parametrize("cadence", ["weekly", "monthly"])
def test_methods_only_act_on_decision_bars(cadence):
    bars = candles(600, seed=4)
    decision = decision_bars(bars["timestamp"], cadence)
    floor = 3. if cadence == "weekly" else 1.
    mstgam = MSTGAM(population_size=8, generations=2, seed=1, execution=execution(), fitness="blend_excess",
                    min_trades_per_year=floor, vote_margin=.1, decision_cadence=cadence).fit(bars)
    moo3 = MOO3(population_size=16, generations=2, seed=1, execution=execution(max_holding_days=5, take_profit=.05),
                objectives="blend_excess", min_trades_per_year=floor, feature_transform="scale_free",
                decision_cadence=cadence).fit(bars)
    ring = IGTSPRing(population_size=8, generations=2, seed=3, island_count=2, groups=3, strategy_count=6,
                     execution=execution(), fitness="blend_excess", weight_mode="equal",
                     decision_cadence=cadence).fit(bars)
    for signals in (mstgam.predict(bars), moo3.predict(bars), ring.predict(bars).signals):
        values = np.asarray(signals).reshape(len(bars), -1)
        assert not values[~decision].any()
    for model in (mstgam, moo3, ring):
        assert model.parameters()["decision_cadence"] == cadence


def test_ring_haircut_covers_every_final_stage_evaluation():
    model = IGTSPRing(population_size=8, generations=4, seed=3, island_count=2, groups=3, strategy_count=6,
                      execution=execution(), fitness="blend_excess", weight_mode="equal").fit(candles(600, seed=5))
    haircut = model.diagnostics["variant"]["selection_haircut"]
    assert haircut["scope"] == "every chromosome evaluated in the final stage"
    assert haircut["candidates"] > 8  # more than the final population
    assert haircut["candidates"] <= model.diagnostics["unique_evaluations"]


def test_cadence_profiles_replace_the_inert_mstgam_knob():
    for profile, cadence, floor in ((ROBUST_WEEKLY_CANDIDATES, "weekly", 3.), (ROBUST_MONTHLY_CANDIDATES, "monthly", 1.)):
        assert all(c["decision_cadence"] == cadence for name in profile for c in profile[name])
        assert [c["vote_margin"] for c in profile["mstgam"]] == [.1, .25]
        assert all("min_votes" not in c and c["min_trades_per_year"] == floor for c in profile["mstgam"])
        assert all(c["min_trades_per_year"] == floor for c in profile["moo3"])
    assert set(CANDIDATE_PROFILES) >= {"robust_weekly", "robust_monthly"}


from gen_trade.statistics import random_aligned_signals


def test_decision_bars_respect_the_bar_before_the_window():
    times = pd.bdate_range("2024-01-03", periods=10, tz="UTC")  # starts on a Wednesday
    assert decision_bars(times, "weekly")[0]
    assert not decision_bars(times, "weekly", previous=pd.Timestamp("2024-01-02", tz="UTC"))[0]
    assert decision_bars(times, "weekly", previous=pd.Timestamp("2023-12-29", tz="UTC"))[0]


def test_aligned_placements_enter_only_after_decision_bars_and_keep_lengths():
    bars = candles(260, start="2024-01-01")
    decision = decision_bars(bars["timestamp"], "monthly")
    signal = apply_cadence(np.where(np.arange(len(bars)) % 37 < 18, 1, -1).astype(np.int8), bars["timestamp"], "monthly")
    actual = run_market_backtest(bars, signal, execution())
    placed = random_aligned_signals(actual.trades, len(bars), 60, np.random.default_rng(3), decision)
    assert placed is not None
    entries = np.argwhere(placed == 1)[:, 0]
    assert decision[entries].all()
    kept = sorted(t.exit_index - t.entry_index for t in actual.trades if not t.forced_at_end)
    for rep in range(placed.shape[1]):
        replay = run_market_backtest(bars, placed[:, rep], execution())
        assert len(replay.trades) == len(actual.trades)
        exits = [t for t in replay.trades if not t.forced_at_end]
        assert all(decision[t.exit_index - 1] for t in exits)  # monthly exits stay on decision bars
        periods = lambda ts: sorted(int(np.sum(decision[t.entry_index - 1:t.exit_index - 1])) for t in ts)
        assert periods(exits) == periods([t for t in actual.trades if not t.forced_at_end])
    assert kept  # the fixture has completed trades


def test_random_timing_uses_aligned_placements_when_given_a_cadence():
    bars = candles(260, start="2024-01-01")
    decision = decision_bars(bars["timestamp"], "weekly")
    signal = apply_cadence(np.where(np.arange(len(bars)) % 11 < 5, 1, -1).astype(np.int8), bars["timestamp"], "weekly")
    result = run_market_backtest(bars, signal, execution())
    timing, note = random_timing(result, bars, execution(), reps=30, seed=2, decision=decision)
    assert note is None and timing["construction"].endswith("decision bars")


# --- lockbox universe and pre-registered hypotheses ------------------------------------

from gen_trade.benchmark import evaluate_hypotheses, hypotheses_hash
from gen_trade.market_data import (DEFAULT_FX, DEFAULT_STOCKS, DJIA_2015, LOCKBOX_EXCLUDED, LOCKBOX_FX,
                                   LOCKBOX_STOCKS, UNIVERSES)


def test_lockbox_universe_follows_its_rule_and_avoids_examined_instruments():
    assert len(DJIA_2015) == len(set(DJIA_2015)) == 30 and "AAPL" not in DJIA_2015
    assert set(LOCKBOX_STOCKS) == set(DJIA_2015) - set(LOCKBOX_EXCLUDED) and len(LOCKBOX_STOCKS) == 22
    assert all("/" in pair and "USD" in pair.split("/") for pair in LOCKBOX_FX)
    assert not set(LOCKBOX_STOCKS + LOCKBOX_FX) & set(DEFAULT_STOCKS + DEFAULT_FX)
    assert UNIVERSES["lockbox"][0] == LOCKBOX_STOCKS and "2015-01-02" in UNIVERSES["lockbox"][2]


def test_hypotheses_are_hashed_at_freeze_and_edits_block_the_unlock(tmp_path):
    hypotheses = [{"id":"H1", "profile":"robust", "method":"mstgam", "kind":"pooled_excess_vs_blend", "alpha":.05}]
    record = freeze_profiles(tmp_path / "lock.json", first_test_year=2026, symbols=["MSFT"], profiles=["robust"],
                             hypotheses=hypotheses, universe_rule="rule")
    assert record["hypotheses_sha256"] == hypotheses_hash(hypotheses) and record["universe_rule"] == "rule"
    lock = json.loads((tmp_path / "lock.json").read_text())
    common = dict(symbols=["MSFT"], last_test_year=2025, unlock=True, method_profile="robust",
                  candidates=ROBUST_CANDIDATES)
    assert check_lockbox(lock, **common)["hypotheses_sha256"] == record["hypotheses_sha256"]
    lock["hypotheses"][0]["alpha"] = .1
    with pytest.raises(ValueError, match="edited"):
        check_lockbox(lock, **common)


def _lock_fold(index, seed, year, excess_mean, train_ir):
    controls = {"minus_exposure_matched_blend":{"paired_return_difference_inference":{"mean":excess_mean}}}
    training = {"variant":{"selection_haircut":{"selected_excess_sharpe_annualized":train_ir}}}
    return {"index":index, "base_seed":seed, "test_first_open":f"{year}-01-02",
            "exposure_controls":{"mstgam":controls, "igtsp_ring":controls},
            "methods":{"mstgam":{"training":training}, "igtsp_ring":{"training":training}}}


def test_preregistered_hypotheses_are_evaluated_and_settings_are_enforced():
    rng = np.random.default_rng(4)
    reports = {}
    for symbol in ("A", "B", "C", "D"):
        folds = []
        for i, year in enumerate(range(2019, 2026)):
            skill = rng.normal(.0003, .0001)
            for seed in (11, 22):
                folds.append(_lock_fold(i, seed, year, skill, skill * 1e4 + rng.normal(0, .1)))
        reports[symbol] = {"folds":folds, "periods_per_year_used":252.}
    specs = [{"id":"H1", "profile":"p", "protocol":"audited2", "method":"mstgam", "kind":"pooled_excess_vs_blend",
              "alpha":.05, "settings":{"population_size":64}},
             {"id":"H2", "profile":"p", "protocol":"audited2", "method":"igtsp_ring",
              "kind":"training_test_rank_correlation", "alpha":.05},
             {"id":"H3", "profile":"other", "method":"mstgam", "kind":"pooled_excess_vs_blend", "alpha":.05}]
    rows = evaluate_hypotheses(specs, method_profile="p", protocol="audited2", reports=reports, manifest={},
                               settings={"population_size":64})
    assert [r["id"] for r in rows] == ["H1", "H2"] and all(r["decision_level"] == .05 / 3 for r in rows)
    assert rows[0]["statistic"]["cells"] == 28 and rows[0]["decision"] == "reject H0 (supported)"
    assert set(rows[0]["leave_one_instrument_out_t"]) == set(reports)
    assert rows[1]["statistic"]["spearman"] > .8 and rows[1]["decision"] == "reject H0 (supported)"
    other = evaluate_hypotheses(specs, method_profile="p", protocol="audited2", reports=reports, manifest={},
                                settings={"population_size":8})
    assert other[0]["decision"] == "not evaluated" and "population_size" in other[0]["note"]
    assert evaluate_hypotheses(specs, method_profile="p", protocol="audited", reports=reports,
                               manifest={})[0]["decision"] == "not evaluated"
