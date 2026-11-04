"""Tests for the labeled theory/audit EXTENSIONS (docs/EXTENSIONS.md).

Paper-faithful defaults are covered by the existing suites; every test here
switches an extension on explicitly and checks its defining property.
"""
from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from gen_trade.backtest import BacktestResult, ExecutionConfig, Trade, run_backtest
from gen_trade.experiments import (ExperimentJob, IMPROVED_CANDIDATES, finish_experiment, pooled_choice,
                                   run_trials, validation_score)
from gen_trade.instruments import InstrumentSpec, benchmark_spec
from gen_trade.market_data import merge_fx_weekend_stubs
from gen_trade.market_execution import MarketArrays, execute_batch
from gen_trade.methods.igtsp_ring import (GroupChromosome, IGTSPRing, bitwise_mutate, facility_location_bank,
                                          group_crossover, random_chromosome)
from gen_trade.methods.indicators import FeatureScaler, feature_families, raw_features
from gen_trade.methods.moo3 import (MOO3, _Individual, das_dennis, nsga3_truncation, rank_fronts,
                                    reference_divisions, sequential_truncation)
from gen_trade.methods.mstgam import MSTGAM, paper_sharpe, trade_fitness
from gen_trade.methods.ring_fitness import portfolio_fitness
from gen_trade.runner import synthetic_markets
from gen_trade.statistics import (alpha_beta, deflated_sharpe_ratio, exposure_matched_blend, exposure_path,
                                  pbo_cscv, probabilistic_sharpe_ratio, random_interval_signals, spa_test)


def candles(n=160, seed=3, scale=1.0):
    rng = np.random.default_rng(seed)
    times = pd.bdate_range("2021-01-04", periods=n, tz="UTC")
    close = scale * 100 * np.exp(.03 * np.sin(np.arange(n) / 4) + np.cumsum(rng.normal(0, .01, n)))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp":times, "available_at":times + pd.Timedelta(hours=8), "open":opening,
                         "high":np.maximum(opening, close) * 1.01, "low":np.minimum(opening, close) * .99,
                         "close":close, "volume":rng.uniform(100, 200, n)})


def result_from(returns, trades=(), start=100.0):
    equity = start * np.concatenate([[1.0], np.cumprod(1 + np.asarray(returns, dtype=float))])
    times = tuple(pd.date_range("2022-01-03", periods=len(returns), freq="D", tz="UTC"))
    return BacktestResult(equity, np.asarray([t.return_fraction for t in trades]), tuple(trades),
                          ExecutionConfig(initial_cash=start), times)


def trade(entry, exit_, allocation=1.0, forced=False):
    t = pd.Timestamp("2022-01-03", tz="UTC")
    return Trade(entry, exit_, t, t, 1.0, 1.0, 0.0, forced, "sample_end" if forced else "signal",
                 allocation_fraction=allocation)


# ----------------------------------------------------------------------- MOO3

def individual(objectives, eligible=True, count=10, violation=0.0):
    return _Individual(None, objectives, eligible, count, violation=violation)


def test_constrained_domination_orders_infeasible_rules_by_violation_only_when_enabled():
    population = [individual((1, 1, 1)), individual((-1e12, -1e12, 1e12), False, 1, 5.0),
                  individual((-1e12, -1e12, 1e12), False, 3, 1.0)]
    paper = rank_fronts(population)
    assert paper == [[0], [1, 2]]
    constrained = rank_fronts(population, constrained=True)
    assert constrained == [[0], [2], [1]]


def test_sequential_truncation_recomputes_crowding_after_each_removal():
    # Three clustered points in the middle: one-shot crowding would drop two
    # neighbours together; sequential removal keeps the cluster represented.
    values = np.array([[0., 10.], [4.9, 5.1], [5., 5.], [5.1, 4.9], [10., 0.]])
    kept = sequential_truncation(values, 3)
    assert 0 in kept and 4 in kept and len(kept) == 3
    assert sum(i in kept for i in (1, 2, 3)) == 1


def test_nsga3_reference_points_and_niching_fill_empty_niches_first():
    points = das_dennis(3, 4)
    assert len(points) == math.comb(6, 2) and np.allclose(points.sum(axis=1), 1)
    assert math.comb(reference_divisions(64) + 2, 2) >= 64
    selected = np.array([[0., 0., 1.]])
    last = np.array([[0., 0., .95], [1., 0., 0.], [0., 1., 0.]])
    chosen = nsga3_truncation(selected, last, 2, das_dennis(3, 2), np.random.default_rng(0))
    assert sorted(chosen) == [1, 2]


def test_robust_objectives_are_compounded_and_lower_bounded():
    model = MOO3(population_size=4, generations=0, objectives="robust", lcb_z=1.0)
    q = np.array([.1, -.05, .02])
    total, lower, std = model._objectives(3, 999., float(q.mean()), float(q.std()), float(np.log1p(q).sum()))
    assert total == pytest.approx(100 * (np.prod(1 + q) - 1))
    assert lower == pytest.approx(q.mean() - q.std(ddof=1) / np.sqrt(3))
    assert std == pytest.approx(q.std())


def test_minimum_trade_rate_constraint_is_enforced_on_the_selected_rule():
    bars = candles(240)
    model = MOO3(population_size=16, generations=3, seed=1, objectives="robust", min_trades_per_year=8,
                 survival="nsga3", feature_transform="scale_free", threshold_volatility_multiple=1.0,
                 execution=ExecutionConfig(fee=.0002, max_holding_days=5, take_profit=.05))
    model.fit(bars)
    years = (bars["available_at"].iloc[-1] - bars["timestamp"].iloc[0]).total_seconds() / (365.25 * 86400)
    minimum = max(2, math.ceil(8 * years))
    assert model.diagnostics["minimum_completed_trades"] == minimum
    selected = model.diagnostics["pareto_front"][model.diagnostics["selected_index"]]
    assert selected["completed_trades"] >= minimum
    assert model.diagnostics["variant"]["constrained_domination"]
    with pytest.raises(ValueError, match="completed trades"):
        MOO3(population_size=8, generations=1, min_trades_per_year=10_000).fit(candles(60))


def test_scale_free_features_remove_price_level_and_share_family_bounds():
    low, high = candles(120), candles(120, scale=10.0)
    paper = raw_features(low, .01), raw_features(high, .01)
    free = raw_features(low, .01, transform="scale_free"), raw_features(high, .01, transform="scale_free")
    np.testing.assert_allclose(paper[1]["ta_ma_10"], 10 * paper[0]["ta_ma_10"])
    for name in ("ta_ma_10", "ta_ema_5", "ta_psar", "ta_atr_3", "ta_obv"):
        np.testing.assert_allclose(free[0][name], free[1][name], rtol=1e-9, equal_nan=True)
    groups = feature_families(free[0])
    assert groups["ta_ma_10"] == groups["ta_ema_3"] == "price_ratio"
    scaler = FeatureScaler.fit(free[0], groups)
    assert scaler.bounds["ta_ma_10"] == scaler.bounds["ta_ma_50"] == scaler.bounds["ta_psar"]


def test_volatility_threshold_is_fitted_on_training_rows_only():
    bars = candles(200)
    model = MOO3(population_size=4, generations=0, threshold_volatility_multiple=2.0).fit(bars.iloc[:120])
    sigma = np.std(np.diff(np.log(bars["close"].iloc[:120])), ddof=1)
    assert model.fitted_threshold_ == pytest.approx(2 * sigma)
    shocked = bars.copy()
    shocked.loc[150:, ["open", "high", "low", "close"]] *= 1.5
    np.testing.assert_array_equal(model.predict(bars)[:120], model.predict(shocked)[:120])


# --------------------------------------------------------------------- MSTGAM

def test_tstat_fitness_and_sign_adjustment():
    q = np.array([.02, -.01, .03, .01])
    assert trade_fitness(4, q.sum(), q.std()) == pytest.approx(paper_sharpe(q))
    expected = (q.sum() - .025 * 2) / (q.std(ddof=1) * 2)
    assert trade_fitness(4, q.sum(), q.std(), variant="tstat", years=2) == pytest.approx(expected)
    calm, wild = trade_fitness(4, -.1, .01), trade_fitness(4, -.1, .05)
    assert wild > calm  # the paper pathology
    calm_fixed = trade_fitness(4, -.1, .01, sign_adjusted=True)
    wild_fixed = trade_fitness(4, -.1, .05, sign_adjusted=True)
    assert calm_fixed > wild_fixed
    assert trade_fitness(1, .1, .1) < -1e11


def test_lognormal_mutation_changes_one_gene_multiplicatively_and_snapshots_track_best():
    model = MSTGAM(mutation="lognormal")
    child = np.full(70, 1 / 70)
    model._mutate(child, np.random.default_rng(4))
    changed = np.flatnonzero(child != 1 / 70)
    assert len(changed) == 1 and child[changed[0]] > 0
    fitted = MSTGAM(population_size=6, generations=4, seed=2, fitness="tstat", sign_adjusted=True,
                    mutation="lognormal", execution=ExecutionConfig(fee=.0002)).fit(candles(200))
    snapshots = fitted.generation_snapshots(every=2)
    assert [g for g, _ in snapshots] == [0, 2, 4]
    np.testing.assert_allclose(snapshots[-1][1].weights_, fitted.weights_)
    assert fitted.diagnostics["variant"]["fitness"] == "tstat"


# ----------------------------------------------------------------------- Ring

def test_paper_ring_ratio_is_bounded_by_best_group_but_mixture_rewards_hedging():
    up = np.array([1., 1.2, 1.0, 1.2, 1.0])
    down = np.array([1., 1.0, 1.2, 1.0, 1.2])
    paths = np.vstack([up, down])
    counts = np.array([4, 4])
    split = GroupChromosome((0, 1), ((1,), (1,)), (0,), (0,))
    paper = portfolio_fitness(split, paths, counts)
    ratios = (paths[:, -1] - 1) / np.max(np.maximum.accumulate(paths, axis=1) - paths, axis=1)
    assert paper["return_risk_ratio"] <= ratios.max() + 1e-12
    mixture = portfolio_fitness(split, paths, counts, variant="mixture_drawdown")
    assert mixture["paper_portfolio_risk"] < paper["paper_portfolio_risk"]
    assert mixture["group_balance"] == pytest.approx(np.log(2))
    calm, wild = np.array([[1., .95, .9], [1., 1., 1.]]), np.array([[1., .7, .9], [1., 1., 1.]])
    first_only = GroupChromosome((0, 1), ((1,), (0,)), (0,), (0,))
    paper_calm, paper_wild = (portfolio_fitness(first_only, x, counts)["return_risk_ratio"] for x in (calm, wild))
    assert paper_wild > paper_calm  # more drawdown "helps" a losing portfolio in Eq.6
    fixed_calm, fixed_wild = (portfolio_fitness(first_only, x, counts, sign_adjusted=True)["return_risk_ratio"]
                              for x in (calm, wild))
    assert fixed_calm > fixed_wild


def test_group_crossover_injects_whole_groups_and_bitwise_mutation_stays_valid():
    rng = np.random.default_rng(5)
    first, second = random_chromosome(rng, 12, 4), random_chromosome(rng, 12, 4)
    second_groups = {frozenset(i for i, label in enumerate(second.assignments) if label == g) for g in range(4)}
    first_groups = {frozenset(i for i, label in enumerate(first.assignments) if label == g) for g in range(4)}
    transmitted = 0
    for _ in range(50):
        child = group_crossover(first, second, rng)
        assert set(child.assignments) == set(range(4))
        assert all(child.weight_bits[g] in (first.weight_bits[g], second.weight_bits[g]) for g in range(4))
        groups = {frozenset(i for i, label in enumerate(child.assignments) if label == g) for g in range(4)}
        transmitted += bool(groups & (second_groups - first_groups))
        mutated = bitwise_mutate(child, rng)
        assert set(mutated.assignments) == set(range(4)) and len(mutated.assignments) == 12
    assert transmitted > 25  # whole second-parent groups usually survive intact


def test_facility_location_bank_covers_distinct_return_clusters():
    rng = np.random.default_rng(6)
    base = rng.normal(0, 1, (2, 30))
    monthly = np.vstack([base[0] + rng.normal(0, .01, (6, 30)), base[1] + rng.normal(0, .01, (6, 30))])
    returns, risks = np.linspace(1, .5, 12), np.full(12, .1)
    chosen = facility_location_bank(monthly, returns, risks, np.arange(12), 2, pool_multiple=6)
    assert {int(i) < 6 for i in chosen} == {True, False}


def test_ring_extensions_fit_end_to_end_without_duplicate_children():
    bars = synthetic_markets(320)["BTC/USDT"]
    model = IGTSPRing(population_size=16, generations=4, seed=3, island_count=2, strategy_count=8,
                      execution=ExecutionConfig(fee=.0002), **IMPROVED_CANDIDATES["igtsp_ring"][0] | {"groups":4})
    model.fit(bars)
    variant = model.diagnostics["variant"]
    assert [s["generations"] for s in variant["stages"]] == [2, 2]
    assert all(event["generation"] < 2 for event in model.diagnostics["migration_events"])
    assert variant["unique_children"] and variant["crossover_mode"] == "group"
    assert len(model.generation_snapshots(every=1)) == 3
    plan = model.predict(bars)
    assert len(plan.names) == 8


# ------------------------------------------------------------------- Protocol

def test_pooled_choice_needs_a_majority_of_seeds_and_uses_mean_scores():
    sets = [[{"validation_score":1.0}, {"validation_score":None}],
            [{"validation_score":0.0}, {"validation_score":3.0}],
            [{"validation_score":2.0}, {"validation_score":None}]]
    winner, pooled = pooled_choice(sets)
    assert winner == 0 and pooled == [1.0, None]


def test_validation_score_requires_the_minimum_trade_count():
    bars = candles(40)
    signals = np.zeros(40, dtype=np.int8); signals[[3, 10]] = 1; signals[[6, 13]] = -1
    result = run_backtest(bars, signals, ExecutionConfig(fee=0))
    assert len(result.trades) == 2
    assert validation_score(result, 2) is not None and validation_score(result, 3) is None


def test_validated_deployment_never_fits_on_validation_rows(monkeypatch):
    from gen_trade import experiments
    fits = []
    class Active:
        def __init__(self, **kwargs): pass
        def fit(self, bars, extra_features=None): fits.append(len(bars)); return self
        def predict(self, bars, extra_features=None):
            s = np.zeros(len(bars), dtype=np.int8); s[::4] = 1; s[2::4] = -1; return s
        def parameters(self): return {}
        @property
        def diagnostics(self): return {}
    monkeypatch.setattr(experiments, "method_class", lambda name: Active)
    bars = synthetic_markets(160)["BTC/USDT"]
    job = ExperimentJob("mstgam", bars, {}, 60, 65, 105, 110, 4, 0, 1, ExecutionConfig(), [{}, {}],
                        deploy="validated", min_validation_trades=2)
    trials = run_trials(job)
    report, _ = finish_experiment(job, trials, 0)
    assert fits == [60, 60, 60]
    assert report["selection"]["protocol"] == "train/validation/deploy-validated/test"
    assert report["selection"]["refit_rows"] == 60


# ----------------------------------------------------------------- Statistics

def test_exposure_alpha_beta_and_blend():
    rng = np.random.default_rng(7)
    market = rng.normal(.001, .01, 200)
    benchmark = result_from(market)
    half = result_from(.5 * market, [trade(0, 100), trade(120, 199, forced=True)])
    exposure = exposure_path(half)
    assert exposure[:100].all() and not exposure[100:120].any() and exposure[120:].all()
    fit = alpha_beta(half, benchmark, periods_per_year=252)
    assert fit["beta"] == pytest.approx(.5) and abs(fit["alpha_per_period"]) < 1e-12
    blend = exposure_matched_blend(benchmark, .5)
    np.testing.assert_allclose(blend.equity, half.equity)


def test_random_interval_signals_replay_durations_and_trade_count():
    bars = candles(120)
    signals = np.zeros(120, dtype=np.int8); signals[[9, 40, 80]] = 1; signals[[19, 45, 99]] = -1
    actual = run_backtest(bars, signals, ExecutionConfig(fee=0))
    random = random_interval_signals(actual.trades, 120, 25, np.random.default_rng(1))
    durations = sorted(t.exit_index - t.entry_index for t in actual.trades)
    for i in range(random.shape[1]):
        replay = run_backtest(bars, random[:, i], ExecutionConfig(fee=0))
        assert sorted(t.exit_index - t.entry_index for t in replay.trades) == durations


def test_psr_dsr_pbo_and_spa_behave_as_documented():
    rng = np.random.default_rng(8)
    good = rng.normal(.001, .01, 1500)
    psr = probabilistic_sharpe_ratio(good)
    dsr = deflated_sharpe_ratio(good, 20, 4e-4)["deflated_sharpe_ratio"]
    assert 0 < dsr < psr < 1
    assert deflated_sharpe_ratio(good, 1, 0.)["deflated_sharpe_ratio"] == pytest.approx(psr)
    noise = rng.normal(0, .01, (800, 8))
    assert .2 < pbo_cscv(noise)["pbo"] < .8
    skilled = noise.copy(); skilled[:, 0] += .004
    assert pbo_cscv(skilled)["pbo"] == 0
    benchmark = rng.normal(0, .01, 800)
    assert spa_test(benchmark, np.column_stack([benchmark + .003, benchmark]), reps=300)["pvalue_consistent"] < .05


# --------------------------------------------------------------- Costs / data

def test_tick_spread_floor_binds_only_for_low_prices_and_defaults_unchanged():
    bars = candles(5)
    bars["execution_open"], bars["execution_close"] = bars["open"], bars["close"]
    cheap = bars.copy(); cheap[["execution_open", "execution_close"]] = 2.0
    spec = InstrumentSpec("X", "stock", spread_bps=2, minimum_spread=.01)
    floor = MarketArrays.from_bars(cheap, spec)
    np.testing.assert_allclose(floor.ask_open - floor.bid_open, .01)
    rich = MarketArrays.from_bars(bars, spec)
    np.testing.assert_allclose(rich.ask_open - rich.bid_open, np.maximum(bars["open"] * 2e-4, .01))
    plain = MarketArrays.from_bars(bars, InstrumentSpec("X", "stock", spread_bps=2))
    np.testing.assert_array_equal(plain.ask_open, bars["open"].to_numpy() * (1 + 1e-4))


def test_benchmark_cost_scenarios_original_and_audited():
    assert benchmark_spec("EUR/USD").financing_long == .02
    assert benchmark_spec("EUR/USD", stress=True).financing_long == .04
    assert benchmark_spec("EUR/USD", fx_financing=0).financing_long == 0
    assert benchmark_spec("EUR/USD", stress=True, fx_financing=0).financing_long == .02
    assert benchmark_spec("GRPN", stock_minimum_spread=.01, stress=True).minimum_spread == .02
    assert benchmark_spec("GRPN").minimum_spread == 0


def test_weekend_stub_merge_is_causal_and_preserves_extremes():
    times = pd.to_datetime(["2024-01-05", "2024-01-07", "2024-01-08", "2024-01-09"], utc=True)
    frame = pd.DataFrame({"symbol":"EUR/USD", "timestamp":times, "available_at":times + pd.Timedelta(days=1),
                          "open":[1., 1.1, 1.2, 1.3], "high":[1.05, 1.5, 1.25, 1.35],
                          "low":[.95, 1.05, .9, 1.25], "close":[1.02, 1.15, 1.22, 1.31], "volume":np.nan,
                          "session_date":["2024-01-05", "2024-01-07", "2024-01-08", "2024-01-09"]})
    merged, count = merge_fx_weekend_stubs(frame)
    assert count == 1 and len(merged) == 3
    row = merged.iloc[1]
    assert row["timestamp"] == times[1] and row["available_at"] == times[2] + pd.Timedelta(days=1)
    assert (row["open"], row["high"], row["low"], row["close"]) == (1.1, 1.5, .9, 1.22)
    assert row["session_date"] == "2024-01-08"


def test_benchmark_code_hash_covers_native_sources(tmp_path, monkeypatch):
    from gen_trade import benchmark
    first = benchmark._code_hash()
    import inspect
    assert "native" in inspect.getsource(benchmark._code_hash)
    assert first == benchmark._code_hash()
