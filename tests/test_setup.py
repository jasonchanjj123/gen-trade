"""Setup v1: required declaration fields, liquidity screen, sessions, lookbacks and the v1 benchmark."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import talib

from gen_trade.benchmark import PROTOCOLS, run_benchmark
from gen_trade.data import ChromosomeContext
from gen_trade.experiments import CANDIDATE_PROFILES
from gen_trade.instruments import benchmark_spec
from gen_trade.market_data import merge_fx_weekend_stubs
from gen_trade.runner import MOO3_EXIT, compare_methods, synthetic_markets, walk_forward_folds
from gen_trade.setup import (BENCHMARK_SLIPPAGE_BPS, CANONICAL_FILE, FEATURE_LOOKBACKS, MAX_FEATURE_LOOKBACK_BARS,
                             REQUIRED_FIELDS, LiquidityRule, build_setup, canonical_setup, liquidity_screen,
                             setup_hash, trading_sessions, validate_setup)

ROOT = Path(__file__).resolve().parents[1]


def _setup(**overrides):
    options = dict(symbols=["AAPL", "GM", "GRPN", "EUR/USD", "GBP/USD", "USD/JPY"], method_profile="paper",
                   candidates=CANDIDATE_PROFILES["paper"], rules=PROTOCOLS["v1"])
    return build_setup(**{**options, **overrides})


def test_v1_declares_every_required_field_and_refuses_gaps_or_stale_hashes():
    setup = validate_setup(_setup())
    assert setup["setup_version"] == "v1"
    for section, names in REQUIRED_FIELDS.items():
        for name in names:
            assert setup[section][name] not in (None, "", [], {}), f"{section}.{name}"
    for section, names in REQUIRED_FIELDS.items():
        broken = json.loads(json.dumps(setup))
        del broken[section][names[0]]
        broken["setup_sha256"] = setup_hash(broken)
        with pytest.raises(ValueError, match=f"{section}.{names[0]}"):
            validate_setup(broken)
    edited = json.loads(json.dumps(setup))
    edited["evaluation"]["embargo"]["trading_sessions"] = 1
    with pytest.raises(ValueError, match="hash"):
        validate_setup(edited)


def test_committed_v1_declaration_matches_the_code():
    """Change control: costs, timing, windows or screens cannot change silently.

    If this fails, decide whether the trading or evaluation rules changed. If
    they did, declare a new setup version instead of editing v1. If only the
    wording changed, regenerate: gen-trade setup --output docs/setup-v1.json
    """
    committed = json.loads((ROOT / CANONICAL_FILE).read_text())
    assert validate_setup(committed) == json.loads(json.dumps(canonical_setup()))


def test_declaration_is_built_from_the_constants_the_run_uses():
    setup = _setup()
    stock, fx = benchmark_spec("XYZ", stock_minimum_spread=.01, fx_financing=0.), benchmark_spec("EUR/USD", fx_financing=0.)
    costs = setup["cost_model"]
    assert costs["commissions"] == {"stock_usd_per_share":stock.commission_per_unit,
                                    "stock_minimum_usd":stock.minimum_commission, "fx_bps_per_side":fx.commission_bps}
    assert costs["slippage_bps_per_side"] == BENCHMARK_SLIPPAGE_BPS
    assert costs["spreads"]["stock_full_spread"] == "max(2 bps, USD 0.01)"
    assert f"{MOO3_EXIT['max_holding_days']} calendar days" in setup["score_to_trade_mapping"]["exit"]["moo3"]
    assert setup["evaluation"]["max_feature_lookback"]["bars"] == MAX_FEATURE_LOOKBACK_BARS
    assert setup["trading_universe"]["liquidity_filters"]["policy"] == "exclude"
    assert [row["test_year"] for row in setup["evaluation"]["fold_calendar"]] == list(range(2019, 2026))
    # A different protocol or profile is a different declaration.
    original = _setup(rules=PROTOCOLS["original"])
    assert original["setup_sha256"] != setup["setup_sha256"]
    assert original["trading_universe"]["liquidity_filters"]["policy"] == "flag"
    monthly = _setup(method_profile="robust_monthly", candidates=CANDIDATE_PROFILES["robust_monthly"])
    assert monthly["decision_schedule"]["decision_cadence"]["mstgam"] == ["monthly"]


def _record_talib(monkeypatch):
    warmups = []
    for name in talib.get_functions():
        original = getattr(talib, name)
        def wrapper(*args, _original=original, **kwargs):
            output = _original(*args, **kwargs)
            for values in (output if isinstance(output, tuple) else (output,)):
                finite = np.flatnonzero(np.isfinite(np.asarray(values, dtype=float)))
                warmups.append(int(finite[0]) if len(finite) else len(values))
            return output
        monkeypatch.setattr(talib, name, wrapper)
    return warmups


def test_declared_lookbacks_match_the_longest_talib_warmup(monkeypatch):
    from gen_trade.methods import indicators, ring_bank
    bars = synthetic_markets(300, seed=3)["BTC/USDT"]
    warmups = _record_talib(monkeypatch)
    indicators.physical_features(bars)
    assert max(warmups) == FEATURE_LOOKBACKS["moo3"]["finite_window_bars"]
    warmups.clear()
    ring_bank.indicator_conditions(bars)
    assert max(warmups) == FEATURE_LOOKBACKS["igtsp_ring"]["finite_window_bars"] == MAX_FEATURE_LOOKBACK_BARS


def _stock(symbol, sessions, price, volume, seed=0):
    rng = np.random.default_rng(seed)
    n = len(sessions)
    close = np.asarray(price, dtype=float) * np.exp(.03 * np.sin(np.arange(n) * .3) + np.cumsum(rng.normal(0, .004, n)))
    opening = np.r_[close[0], close[:-1]]
    opens = pd.DatetimeIndex(sessions).tz_localize("UTC") + pd.Timedelta(hours=14, minutes=30)
    volume = np.broadcast_to(np.asarray(volume, dtype=float), (n,)).copy()
    return pd.DataFrame({"symbol":symbol, "session_date":pd.DatetimeIndex(sessions).strftime("%Y-%m-%d"),
                         "timestamp":opens, "available_at":opens + pd.Timedelta(hours=6, minutes=30),
                         "open":opening, "high":np.maximum(opening, close) * 1.003,
                         "low":np.minimum(opening, close) * .997, "close":close, "volume":volume,
                         "execution_close":close, "execution_volume":volume, "split_ratio":1., "dividend":0.,
                         "asset_class":"stock"})


def _fx(symbol, days, with_quotes=True):
    opens = pd.DatetimeIndex(days, tz="UTC")
    close = 1.1 + .001 * np.sin(np.arange(len(opens)))
    frame = pd.DataFrame({"symbol":symbol, "session_date":opens.strftime("%Y-%m-%d"), "timestamp":opens,
                          "available_at":opens + pd.Timedelta(days=1), "open":close, "high":close + .002,
                          "low":close - .002, "close":close, "volume":np.nan, "asset_class":"fx"})
    if with_quotes:
        for side, shift in (("bid", -.0001), ("ask", .0001)):
            for field in ("open", "high", "low", "close"):
                frame[f"{side}_{field}"] = frame[field] + shift
    return frame


def test_liquidity_screen_is_point_in_time_and_applies_each_threshold():
    sessions = pd.bdate_range("2020-01-01", periods=120)
    liquid = _stock("LIQ", sessions, 20., 1e6)
    result = liquidity_screen(liquid, 100, capital=100_000)
    assert result["eligible"] and result["reasons"] == []
    assert result["window_last_session"] == liquid["session_date"].iloc[99]
    assert result["metrics"]["participation"] == pytest.approx(100_000 / result["metrics"]["median_dollar_volume"])
    later = liquid.copy()
    later.loc[100:, ["execution_close", "execution_volume"]] = [.5, 1.]
    assert liquidity_screen(later, 100, capital=100_000) == result
    penny = liquidity_screen(_stock("PNY", sessions, 3., 1e7), 100, capital=100_000)
    assert not penny["eligible"] and len(penny["reasons"]) == 1
    assert penny["reasons"][0].startswith("median price 3.") and penny["reasons"][0].endswith("below 5.00")
    thin = liquidity_screen(_stock("THN", sessions, 20., 2e4), 100, capital=100_000)
    assert not thin["eligible"] and len(thin["reasons"]) == 2
    assert "dollar volume" in thin["reasons"][0] and "above 1%" in thin["reasons"][1]
    assert "fewer than 63" in liquidity_screen(liquid, 40, capital=100_000)["reasons"][0]
    unknown = liquid.copy()
    unknown.loc[50, "execution_volume"] = np.nan
    assert "session volume unknown" in liquidity_screen(unknown, 100, capital=100_000)["reasons"]
    loose = LiquidityRule(min_median_price=1.)
    assert liquidity_screen(_stock("PNY", sessions, 3., 1e7), 100, capital=100_000, rule=loose)["eligible"]
    days = pd.bdate_range("2020-01-01", periods=80)
    assert liquidity_screen(_fx("EUR/USD", days), 70, capital=100_000)["eligible"]
    assert liquidity_screen(_fx("EUR/GBP", days), 70, capital=100_000)["reasons"] == ["not a G10 currency against USD"]
    assert liquidity_screen(_fx("USD/TRY", days), 70, capital=100_000)["reasons"] == ["not a G10 currency against USD"]
    assert liquidity_screen(_fx("EUR/USD", days, False), 70, capital=100_000)["reasons"] == ["no observed bid/ask quotes"]


def test_trading_sessions_skip_raw_fx_sunday_stubs_and_count_merged_rows_once():
    days = [d for d in pd.date_range("2024-01-07", "2024-01-19") if d.day_of_week != 5]
    raw = _fx("EUR/USD", days)
    assert len(raw) == 12 and trading_sessions(raw, 0, len(raw)) == 10
    merged, count = merge_fx_weekend_stubs(raw)
    assert count == 2 and len(merged) == trading_sessions(merged, 0, len(merged)) == 10
    without_dates = merged.drop(columns="session_date")
    assert trading_sessions(without_dates, 0, len(without_dates)) == 10
    stock = _stock("LIQ", pd.bdate_range("2024-01-01", periods=10), 20., 1e6)
    assert trading_sessions(stock, 2, 7) == 5


def _fake_methods(monkeypatch):
    from gen_trade.methods import igtsp_ring, moo3, mstgam

    class Method:
        def __init__(self, *, population_size, generations, seed, execution, threshold=.001, min_votes=3, groups=17,
                     take_profit_limit=.1, stop_loss_limit=.1, **options):
            self.execution, self.seed = execution, seed

        def fit(self, bars, extra_features=None):
            self.rows = len(bars)
            return self

        def predict(self, bars, extra_features=None):
            index = np.arange(len(bars))
            return np.where(index % 4 == 0, 1, np.where(index % 4 == 2, -1, 0))

        def parameters(self):
            return {"seed":self.seed, "training_rows":self.rows}

        @property
        def diagnostics(self):
            return {"training_rows":self.rows}

    for module, name in ((moo3, "MOO3"), (mstgam, "MSTGAM"), (igtsp_ring, "IGTSPRing")):
        monkeypatch.setattr(module, name, Method)


def test_fold_numbers_keep_calendar_seeds_when_folds_are_left_out(monkeypatch):
    _fake_methods(monkeypatch)
    markets = synthetic_markets(220, seed=4)
    folds = walk_forward_folds(220, 60, 30, embargo_bars=3, validation_bars=30)
    options = dict(population_size=4, generations=0, workers=1, bootstrap_reps=100, seeds=(10,),
                   train_bars=60, validation_bars=30, test_bars=30, embargo_bars=3)
    report = compare_methods(markets, ChromosomeContext("BTC/USDT"), fold_plan=[folds[0], folds[2]],
                             fold_numbers=[0, 2], **options)
    assert [(f["index"], f["effective_seed"]) for f in report["folds"]] == [(0, 10), (2, 12)]
    for numbers in ([2, 0], [0], [0, 0], [-1, 2]):
        with pytest.raises(ValueError, match="fold_numbers"):
            compare_methods(markets, ChromosomeContext("BTC/USDT"), fold_plan=[folds[0], folds[2]],
                            fold_numbers=numbers, **options)


def _dataset(directory):
    sessions = pd.bdate_range("2015-01-01", "2020-12-31")
    low_volume = np.where((sessions >= "2019-10-01") & (sessions < "2020-01-01"), 1e4, 1e6)
    frames = {"LIQ":_stock("LIQ", sessions, 50., 1e6, seed=1),
              "ILL":_stock("ILL", sessions, 50., low_volume, seed=2),
              "PNY":_stock("PNY", sessions, 2., 1e8, seed=3)}
    directory.mkdir()
    instruments = {}
    for symbol, frame in frames.items():
        path = directory / f"{symbol}.parquet"
        frame.to_parquet(path, index=False)
        instruments[symbol] = {"file":path.name, "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                               "periods_per_year":252.}
    (directory / "manifest.json").write_text(json.dumps({"universe_selection":"test fixture", "instruments":instruments}))


def _verifier():
    spec = importlib.util.spec_from_file_location("verify_benchmark", ROOT / "scripts" / "verify-benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify


@pytest.mark.parametrize("protocol", ["v1", "original"])
def test_v1_benchmark_removes_illiquid_years_and_verifies(tmp_path, monkeypatch, protocol):
    _fake_methods(monkeypatch)
    data, output = tmp_path / "data", tmp_path / "out"
    _dataset(data)
    summary = run_benchmark(data, output, population_size=4, generations=0, seeds=(1,), workers=1,
                            first_test_year=2019, last_test_year=2020, bootstrap_reps=100, protocol=protocol,
                            random_timing_reps=20, spa_reps=50, progress="off")
    screen = {(row["symbol"], row["test_year"]):row for row in summary["liquidity_screen"]["instrument_years"]}
    assert not screen[("ILL", 2020)]["eligible"] and screen[("ILL", 2019)]["eligible"]
    assert not screen[("PNY", 2019)]["eligible"] and not screen[("PNY", 2020)]["eligible"]
    report_text = (output / "report.md").read_text()
    assert "## Liquidity screen" in report_text
    if protocol == "v1":
        assert summary["universe"] == ["LIQ", "ILL"] and summary["excluded_instruments"] == ["PNY"]
        assert not (output / "PNY.json").exists()
        ill = json.loads((output / "ILL.json").read_text())
        assert [row["traded"] for row in ill["fold_audit"]] == [True, False]
        assert [(f["test_first_open"][:4], f["effective_seed"]) for f in ill["folds"]] == [("2019", 1)]
        liq = json.loads((output / "LIQ.json").read_text())
        assert [f["effective_seed"] for f in liq["folds"]] == [1, 2]
        setup = validate_setup(summary["setup"])
        assert summary["settings"]["setup_sha256"] == setup["setup_sha256"] == ill["setup_sha256"]
        assert all(min(row["embargo_sessions"].values()) == 5 for row in ill["fold_audit"])
        assert "## Setup v1" in report_text and "| ILL | 2020 | no |" in report_text
    else:
        assert summary["universe"] == ["LIQ", "ILL", "PNY"] and "setup" not in summary
        assert "| PNY | 2019 | yes |" in report_text
    verify = _verifier()
    assert verify(output, data)["status"] == "passed"
    if protocol == "v1":
        ill["fold_audit"][1]["traded"] = True
        (output / "ILL.json").write_text(json.dumps(ill))
        with pytest.raises(ValueError, match="traded flag"):
            verify(output, data)
