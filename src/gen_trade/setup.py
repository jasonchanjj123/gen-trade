"""Trading setup and evaluation protocol as data (``setup_version`` v1).

One declaration covers the trading universe, decision schedule, admissible
information, score-to-trade mapping, risk constraints, cost model and the
walk-forward evaluation. ``build_setup`` assembles it from the constants the
code runs with (protocol rules, instrument cost specs, exit rules, fold
settings), so the copy stamped into a report cannot drift from the run.
``validate_setup`` refuses a declaration with a missing required field or a
stale hash.

Benchmark protocol ``v1`` enforces this setup: the audited2 rules, a
point-in-time liquidity screen that removes ineligible instrument-years, and
a check that every embargo spans the declared number of trading sessions.
Earlier protocols record the same screen as a flag only. See docs/SETUP_V1.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math

import numpy as np
import pandas as pd

SETUP_VERSION = "v1"

# Daily slippage per side added to modeled or observed spreads (benchmark).
BENCHMARK_SLIPPAGE_BPS = {"stock": 1, "fx": .1}

# Longest finite indicator warm-up, in bars, per method (tests check these
# against TA-Lib). Recursive indicators also carry state from the fold's
# history start; see FEATURE_LOOKBACKS.
FEATURE_LOOKBACKS = {
    "moo3": {
        "finite_window_bars": 50,
        "calendar_days": 50,
        "detail": "SMA, CCI, RSI and Williams %R over 10-50 bars; ATR and EMA over 3-10 bars; scale-free OBV "
                  "change over 20 bars; directional-change statistics over 3-50 calendar days",
    },
    "mstgam": {
        "finite_window_bars": 0,
        "calendar_days": 0,
        "detail": "directional-change state machines at ten fixed thresholds; no fixed window",
    },
    "igtsp_ring": {
        "finite_window_bars": 88,
        "calendar_days": 0,
        "detail": "21 TA-Lib rules; longest warm-up TRIX(30) = 88 bars, then MACD(12,26,9) = 33 and ADX(14) = 27",
    },
}
MAX_FEATURE_LOOKBACK_BARS = max(item["finite_window_bars"] for item in FEATURE_LOOKBACKS.values())
RECURSIVE_STATE = ("EMA, Wilder RSI/ATR/ADX, OBV, Parabolic SAR and directional-change trend state are recursive: "
                   "they start at the fold's training start and run forward, so a test-year decision depends on "
                   "every bar from the training start up to the decision bar and on none after it")

G10_CURRENCIES = ("USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "NOK", "SEK")


@dataclass(frozen=True)
class LiquidityRule:
    """Tradability thresholds for one instrument-year (v1 values)."""

    lookback_sessions: int = 63
    min_median_price: float = 5.0
    min_median_dollar_volume: float = 10_000_000.0
    max_participation: float = .01
    fx_currencies: tuple[str, ...] = G10_CURRENCIES


LIQUIDITY_V1 = LiquidityRule()

REQUIRED_FIELDS = {
    "trading_universe": ("asset_selection", "liquidity_filters"),
    "decision_schedule": ("signal_calculation_time", "execution_time", "delay"),
    "admissible_information": ("feature_lookbacks", "data_publication_lags"),
    "score_to_trade_mapping": ("entry", "exit", "sizing", "no_trade_state"),
    "risk_constraints": ("leverage", "concentration", "exposure_limits"),
    "cost_model": ("commissions", "spreads", "other_costs"),
    "evaluation": ("scheme", "training_window", "validation_window", "step_size", "retraining_frequency",
                   "label_horizon", "max_feature_lookback", "test_dates", "purging", "embargo"),
}


def _asset_class(bars: pd.DataFrame) -> str:
    if "asset_class" in bars and len(bars):
        return str(bars["asset_class"].iloc[0])
    return "fx" if "/" in str(bars["symbol"].iloc[0]) else "stock"


def _session_dates(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Trading date of each row: ``session_date`` when present; otherwise the
    UTC day the candle covers (FX: the day before ``available_at``, which a
    merged Sunday stub shares with Monday) or the bar's opening day."""
    if "session_date" in bars:
        return pd.DatetimeIndex(pd.to_datetime(bars["session_date"]))
    if len(bars) and _asset_class(bars) == "fx":
        stamps = pd.to_datetime(bars["available_at"], utc=True) - pd.Timedelta(days=1)
    else:
        stamps = pd.to_datetime(bars["timestamp"], utc=True)
    return pd.DatetimeIndex(stamps.dt.tz_localize(None).dt.normalize())


def trading_sessions(bars: pd.DataFrame, start: int, stop: int) -> int:
    """Weekday trading sessions among rows ``[start, stop)``.

    Exchange-calendar stock rows are all sessions. A raw FX Sunday stub (the
    few hours after the weekly open) is not; once ``merge_fx_weekend_stubs``
    has folded it into Monday, the merged row keeps Monday's session date and
    counts once.
    """
    return int((_session_dates(bars.iloc[start:stop]).day_of_week < 5).sum())


def _finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def liquidity_screen(bars: pd.DataFrame, stop: int, *, capital: float, rule: LiquidityRule = LIQUIDITY_V1) -> dict:
    """Tradability of one instrument for the test year that follows row ``stop``.

    Uses only the ``rule.lookback_sessions`` rows before ``stop`` (benchmarks
    pass the validation end, so these rows are known before the test
    embargo), which makes the decision point-in-time. A stock needs a median
    raw close of at least ``min_median_price``, a median session dollar
    volume of at least ``min_median_dollar_volume``, and a full-capital order
    no larger than ``max_participation`` of that volume: the cost model has no
    market-impact term, which is defensible only at small participation. An
    FX pair needs both currencies in the G10 with USD on one side and observed
    bid/ask quotes; spot-FX volume is not consolidated, so no volume test
    applies.
    """
    symbol = str(bars["symbol"].iloc[0])
    kind = _asset_class(bars)
    window = bars.iloc[max(0, stop - rule.lookback_sessions):stop]
    if not len(window):
        raise ValueError("Liquidity screen needs rows before the cutoff")
    dates = _session_dates(window)
    record = {"symbol":symbol, "asset_class":kind, "window_first_session":dates[0].strftime("%Y-%m-%d"),
              "window_last_session":dates[-1].strftime("%Y-%m-%d"),
              "known_at":pd.Timestamp(window["available_at"].iloc[-1]).isoformat(), "sessions":len(window)}
    reasons, metrics = [], {}
    if kind == "fx":
        base, quote = symbol.split("/")
        if {base, quote} - set(rule.fx_currencies) or "USD" not in (base, quote):
            reasons.append("not a G10 currency against USD")
        if any(name not in bars for name in ("bid_open", "ask_open", "bid_close", "ask_close")):
            reasons.append("no observed bid/ask quotes")
    else:
        if len(window) < rule.lookback_sessions:
            reasons.append(f"fewer than {rule.lookback_sessions} sessions before the cutoff")
        price = window["execution_close" if "execution_close" in window else "close"].to_numpy(dtype=float)
        volume = window["execution_volume" if "execution_volume" in window else "volume"].to_numpy(dtype=float)
        median_price = float(np.median(price))
        metrics["median_price"] = median_price
        if median_price < rule.min_median_price:
            reasons.append(f"median price {median_price:.2f} below {rule.min_median_price:.2f}")
        if not np.isfinite(volume).all():
            reasons.append("session volume unknown")
            metrics.update(median_dollar_volume=None, participation=None)
        else:
            dollar_volume = float(np.median(price * volume))
            participation = capital / dollar_volume if dollar_volume > 0 else None
            metrics.update(median_dollar_volume=dollar_volume, participation=_finite(participation))
            if dollar_volume < rule.min_median_dollar_volume:
                reasons.append(f"median dollar volume {dollar_volume / 1e6:.1f}M below "
                               f"{rule.min_median_dollar_volume / 1e6:.1f}M")
            if participation is None or participation > rule.max_participation:
                share = "undefined" if participation is None else f"{participation:.2%}"
                reasons.append(f"order is {share} of median dollar volume, above {rule.max_participation:.0%}")
    return {**record, "metrics":metrics, "eligible":not reasons, "reasons":reasons}


def fold_audit(bars: pd.DataFrame, folds, years, *, embargo_sessions: int, capital: float,
               rule: LiquidityRule = LIQUIDITY_V1, max_lookback: int = MAX_FEATURE_LOOKBACK_BARS) -> list[dict]:
    """Dates, session counts, embargo, warm-up and liquidity checks per test year."""
    dates = _session_dates(bars)
    def day(index):
        return dates[index].strftime("%Y-%m-%d")
    rows = []
    for year, fold in zip(years, folds):
        gaps = {"after_training":trading_sessions(bars, fold.train_stop, fold.validation_start),
                "after_validation":trading_sessions(bars, fold.validation_stop, fold.test_start)}
        rows.append({
            "test_year":int(year),
            "training":[day(fold.train_start), day(fold.train_stop - 1)],
            "validation":[day(fold.validation_start), day(fold.validation_stop - 1)],
            "test":[day(fold.test_start), day(fold.test_stop - 1)],
            "training_bars":fold.train_stop - fold.train_start,
            "validation_bars":fold.validation_stop - fold.validation_start,
            "test_bars":fold.test_stop - fold.test_start,
            "embargo_sessions":gaps,
            "embargo_ok":min(gaps.values()) >= embargo_sessions,
            "training_bars_after_warmup":fold.train_stop - fold.train_start - max_lookback,
            "warmup_ok":fold.train_stop - fold.train_start > max_lookback,
            "liquidity":liquidity_screen(bars, fold.validation_stop, capital=capital, rule=rule),
        })
    return rows


def setup_hash(setup: dict) -> str:
    body = {key:value for key, value in setup.items() if key != "setup_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def validate_setup(setup: dict) -> dict:
    """Refuse a declaration that misses a required field or has a stale hash."""
    if setup.get("setup_version") != SETUP_VERSION:
        raise ValueError(f"setup_version must be {SETUP_VERSION!r}")
    missing = [f"{section}.{name}" for section, names in REQUIRED_FIELDS.items() for name in names
               if not isinstance(setup.get(section), dict) or setup[section].get(name) in (None, "", [], {})]
    if missing:
        raise ValueError(f"Setup {SETUP_VERSION} is missing required fields: {missing}")
    if setup.get("setup_sha256") != setup_hash(setup):
        raise ValueError("Setup hash does not match its content")
    return setup


def _calendar(first_test_year, last_test_year, train_years, validation_years, embargo):
    return [{"test_year":year,
             "training":f"{year - train_years - validation_years}-01-01 to the last session before "
                        f"{year - validation_years}-01-01, minus {embargo} sessions",
             "validation":f"{year - validation_years}-01-01 to the last session of {year - 1}, minus {embargo} sessions",
             "test":f"{year}-01-01 to {year}-12-31"}
            for year in range(first_test_year, last_test_year + 1)]


CANONICAL_FILE = "docs/setup-v1.json"


def canonical_setup(method_profile: str = "paper", *, data_directory=None, first_test_year: int = 2019,
                    last_test_year: int = 2025) -> dict:
    """The v1 declaration for a dataset's universe (default: the six research
    instruments), the repository lockbox and protocol ``v1``. With the
    defaults it is the committed reference copy in ``docs/setup-v1.json``."""
    from pathlib import Path
    from .benchmark import PROTOCOLS
    from .experiments import CANDIDATE_PROFILES
    from .market_data import DEFAULT_FX, DEFAULT_STOCKS, UNIVERSES
    manifest = Path(data_directory) / "manifest.json" if data_directory else None
    if manifest is not None and manifest.exists():
        record = json.loads(manifest.read_text())
        symbols, rule = list(record["instruments"]), record.get("universe_selection")
    else:
        symbols, rule = [*DEFAULT_STOCKS, *DEFAULT_FX], UNIVERSES["default"][2]
    lockbox = Path(__file__).resolve().parents[2] / "lockbox.json"
    return validate_setup(build_setup(symbols=symbols, method_profile=method_profile,
                                      candidates=CANDIDATE_PROFILES[method_profile], rules=PROTOCOLS["v1"],
                                      first_test_year=first_test_year, last_test_year=last_test_year,
                                      universe_rule=rule,
                                      lockbox=json.loads(lockbox.read_text()) if lockbox.exists() else None))


def build_setup(*, symbols, method_profile: str, candidates: dict, rules: dict, first_test_year: int = 2019,
                last_test_year: int = 2025, train_years: int = 3, validation_years: int = 1,
                embargo_sessions: int = 5, initial_cash: float = 100_000, seeds=(11, 22, 33),
                population_size: int = 64, generations: int = 20, universe_rule: str | None = None,
                lockbox: dict | None = None, liquidity: LiquidityRule = LIQUIDITY_V1) -> dict:
    """The v1 declaration for one benchmark run, from the code's own constants."""
    from .experiments import METHOD_NAMES
    from .instruments import benchmark_spec
    from .methods.igtsp_ring import IGTSPRing
    from .runner import MOO3_EXIT

    def spec(symbol, stress=False):
        return benchmark_spec(symbol, stress=stress, fx_financing=rules["fx_financing"],
                              stock_minimum_spread=rules["stock_minimum_spread"])
    stock, stock_stress, fx, fx_stress = spec("XYZ"), spec("XYZ", True), spec("EUR/USD"), spec("EUR/USD", True)
    ring = inspect.signature(IGTSPRing).parameters
    take_profit, stop_loss = ring["take_profit_limit"].default, ring["stop_loss_limit"].default
    compare = rules["compare"]
    cadences = {name:sorted({c.get("decision_cadence", "daily") for c in candidates[name]}) for name in METHOD_NAMES}
    hurdles = sorted({f"vote_margin {c['vote_margin']}" if "vote_margin" in c else f"min_votes {c.get('min_votes', 3)}"
                      for c in candidates["mstgam"]})
    policy = rules.get("liquidity_policy", "flag")
    stocks = sorted(s for s in symbols if "/" not in s)
    pairs = sorted(s for s in symbols if "/" in s)
    deploy = compare.get("deploy", "refit")
    setup = {
        "setup_version":SETUP_VERSION,
        "method_profile":method_profile,
        "trading_universe":{
            "asset_selection":{
                "instruments":{"stocks":stocks, "fx":pairs},
                "rule":universe_rule or "fixed instrument list supplied with the dataset",
                "accounts":"one independent USD account per instrument; no cross-asset portfolio",
                "data_requirements":"complete XNYS regular sessions for stocks; observed Dukascopy bid/ask candles "
                                    "for FX, with crossed quotes quarantined",
                "survivorship":"a fixed list chosen in advance; not a survivorship-free sample",
            },
            "liquidity_filters":{
                **asdict(liquidity),
                "measured_on":f"the {liquidity.lookback_sessions} bars before the validation end, all known "
                              "before the test embargo",
                "stocks":"median raw close >= min_median_price, median close x volume >= min_median_dollar_volume, "
                         "fold capital / median dollar volume <= max_participation",
                "fx":"both currencies in fx_currencies with USD on one side, observed bid/ask quotes",
                "policy":policy,
                "on_failure":("the instrument-year is not traded or evaluated" if policy == "exclude" else
                              "recorded as a flag; the instrument-year is still traded"),
            },
        },
        "decision_schedule":{
            "bar":"one daily bar per session: XNYS regular session for stocks (09:30-16:00 ET, early closes "
                  "respected); UTC calendar day for FX",
            "signal_calculation_time":"at the bar's available_at (stocks: the session close; FX: 00:00 UTC after "
                                      "the candle), from bars whose available_at is at or before it",
            "execution_time":"the next bar's open, and only if that open is at or after the signal bar's "
                             "available_at; stocks fill at the next 09:30 ET open with spread and slippage, FX buys "
                             "at the next candle's first ask and sells at its first bid",
            "delay":{"bars":1, "stocks":"one overnight, about 17.5 hours",
                     "fx":"none after the candle completes: the next candle's first quote at 00:00 UTC is "
                          "assumed executable"},
            "decision_cadence":cadences,
            "exit_monitoring":"targets, stops and holding deadlines are checked at each close and filled at the "
                              "next open; no intrabar fills",
            "retraining_time":"once per test year, before its first session",
            "entry_hours":"unrestricted",
        },
        "admissible_information":{
            "point_in_time_rule":"a decision at bar t uses only rows with available_at <= available_at(t); "
                                 "cross-market features use a backward as-of join with a finite maximum age; lags "
                                 "and rolling windows are never backfilled",
            "feature_lookbacks":{**FEATURE_LOOKBACKS, "max_finite_window_bars":MAX_FEATURE_LOOKBACK_BARS,
                                 "recursive_state":RECURSIVE_STATE},
            "data_publication_lags":{
                "stock_ohlcv":"known at the session close (available_at = XNYS close); no further vendor delay "
                              "is modeled",
                "fx_ohlc":"known at 00:00 UTC after the candle (available_at = open + 1 day)",
                "splits":"applied on the effective date; later split factors only undo the vendor's "
                         "back-adjustment of earlier prices (regression-tested) and are never features",
                "dividends":"cash entitlement on the ex-date open; adjusted closes are not used",
                "vendor_revisions":"not reconstructed",
            },
            "fit_time_information":"scalers, directional-change thresholds and triggers, and the Ring rule bank "
                                   "are fitted on training rows only; configuration choice uses validation rows "
                                   "only; test rows are never used to fit or select",
            "excluded":["dividend-adjusted closes", "FX volume (not consolidated); volume rules leave the Ring pool",
                        "any row after the decision bar"],
            "reporting_only":"annualization factors count bars per year over the whole dataset; they scale "
                             "reported statistics and never enter fitting or selection",
        },
        "score_to_trade_mapping":{
            "signal_alphabet":"+1 buy: enter long if flat; -1 sell: exit if long; 0: no action",
            "entry":{
                "moo3":"buy when the evolved Boolean rule tree is true at the close",
                "mstgam":f"buy when the buy side wins the weighted vote of 70 directional-change genes "
                         f"(hurdle: {', '.join(hurdles)})",
                "igtsp_ring":"each sleeve buys when its TA-Lib buy rule fires",
            },
            "exit":{
                "moo3":f"holding deadline of {MOO3_EXIT['max_holding_days']} calendar days, or a "
                       f"+{MOO3_EXIT['take_profit']:.0%} target observed at a close; no sell rule",
                "mstgam":"sell when the sell side wins the vote",
                "igtsp_ring":f"the sleeve's sell rule, or an evolved take-profit in [0, {take_profit:.0%}] or "
                             f"stop-loss in [0, {stop_loss:.0%}] observed at a close",
                "all":"filled at the next open; a position still open at a window's last close is liquidated "
                      "there with exit costs (sample_end)",
            },
            "sizing":{
                "rule":"all allocated capital in fractional units at the fill price after costs",
                "moo3_mstgam":"one position using the fold's whole capital",
                "igtsp_ring":"capital split across groups by the evolved weights (equal weights in robust "
                             "profiles), equally among each group's members, fixed for the year",
                "fold_capital_usd":initial_cash,
            },
            "no_trade_state":[
                "signal 0 keeps the current state; a flat account stays in cash earning nothing",
                "during indicator warm-up, undefined features never trigger",
                "bars between weekly or monthly decision bars emit no action",
                "conflicting Ring buy and sell rules hold",
                f"when no configuration has >= {compare.get('min_validation_trades', 1)} validation trades, the "
                "method holds cash for the whole test year",
                "an instrument-year that fails the liquidity screen is not traded" if policy == "exclude" else
                "liquidity screen failures are flagged only",
            ],
        },
        "risk_constraints":{
            "leverage":f"{stock.leverage:g}x cash account; no borrowing",
            "direction":"long only" if not (stock.allow_short or fx.allow_short) else "long and short",
            "exposure_limits":"gross exposure <= 100% of fold equity; no pyramiding; Ring sleeve allocations fixed "
                              "at the fold start with no rebalancing or borrowing between sleeves",
            "concentration":"100% in one instrument per account by design; Ring diversifies only across rules on "
                            "that instrument",
            "liquidity_participation":f"order <= {liquidity.max_participation:.0%} of median session dollar "
                                      "volume (stocks)",
            "margin":f"maintenance-margin exit at {stock.maintenance_margin_fraction:.0%} (inactive at 1x cash)",
            "stops":"only Ring evolves stop losses; no account drawdown stop, volatility target or kill switch",
        },
        "cost_model":{
            "commissions":{"stock_usd_per_share":stock.commission_per_unit,
                           "stock_minimum_usd":stock.minimum_commission, "fx_bps_per_side":fx.commission_bps},
            "spreads":{"stock_full_spread":f"max({stock.spread_bps:g} bps, USD {stock.minimum_spread:g})",
                       "fx":"observed bid/ask: buy at ask, sell at bid"},
            "slippage_bps_per_side":BENCHMARK_SLIPPAGE_BPS,
            "other_costs":{
                "fx_financing_annual":fx.financing_long, "fx_financing_source":fx.financing_source,
                "dividends":"credited on the ex-date, no withholding tax",
                "currency_conversion":"side-aware at contemporaneous bid/ask (USD/JPY)",
                "market_impact":"not modeled; relies on the participation cap",
                "taxes":"not modeled", "idle_cash_interest":0,
            },
            "stress_scenario":{"stock_full_spread":f"max({stock_stress.spread_bps:g} bps, USD "
                                                  f"{stock_stress.minimum_spread:g})",
                               "stock_usd_per_share":stock_stress.commission_per_unit,
                               "fx_bps_per_side":fx_stress.commission_bps,
                               "fx_financing_annual":fx_stress.financing_long,
                               "slippage_bps_per_side":{k:2 * v for k, v in BENCHMARK_SLIPPAGE_BPS.items()},
                               "models":"the same frozen strategies; nothing is refitted"},
            "applied_in":"training fitness, validation, test and the buy-and-hold baseline, through one engine",
        },
        "evaluation":{
            "scheme":"rolling walk-forward on calendar years: train, embargo, validation, embargo, test",
            "training_window":f"{train_years} calendar years before the validation year, minus the embargo",
            "validation_window":f"{validation_years} calendar year before the test year, minus the embargo",
            "test_window":"one complete calendar year; every fold starts flat with fresh capital",
            "step_size":"1 calendar year",
            "retraining_frequency":"annual: a fresh fit per method, seed and test year; deployed model: "
                                   + ("the validated training-only model" if deploy == "validated"
                                      else "a new fit through the validation end"),
            "label_horizon":{
                "definition":"bars from a decision to the end of the return it earns in fitness or validation",
                "execution_delay_bars":1,
                "moo3":f"at most {MOO3_EXIT['max_holding_days']} calendar days (holding deadline)",
                "mstgam":"unbounded until a sell vote; truncated at the window end",
                "igtsp_ring":"unbounded until a sell rule, target or stop; truncated at the window end",
                "daily_fitness":"one bar for marked-to-market daily fitness (robust profiles)",
            },
            "max_feature_lookback":{"bars":MAX_FEATURE_LOOKBACK_BARS, "recursive_state":RECURSIVE_STATE,
                                    "check":"training windows must exceed it (warmup_ok in fold_audit)"},
            "test_dates":{"first":f"{first_test_year}-01-01", "last":f"{last_test_year}-12-31",
                          "years":list(range(first_test_year, last_test_year + 1)),
                          "lockbox":None if not lockbox else
                          {"first_test_year":lockbox.get("first_test_year"),
                           "held_out_instruments":len(lockbox.get("symbols", []))}},
            "purging":{
                "method":"label truncation at every window boundary",
                "rule":"fit receives only rows before the training end and validation scores only validation "
                       "rows; positions open at a window's last close are liquidated there, so no training or "
                       "validation label ends after its window",
                "why_not_fixed_width":"MSTGAM and Ring holding periods are unbounded (p95 12-85 bars, maximum a "
                                      "full year in the audited2 runs), so a fixed purge would remove a year",
                "tests":["tests/test_runner.py::test_future_price_changes_cannot_change_training_inputs_or_frozen_parameters",
                         "tests/test_runner.py::test_validation_selection_and_refit_never_see_held_out_prices",
                         "tests/test_backtest.py::test_terminal_liquidation_uses_final_close_and_records_boundary_convention"],
            },
            "embargo":{"trading_sessions":embargo_sessions,
                       "where":["training to validation", "validation to test"],
                       "unit":"weekday trading sessions; FX Sunday stubs are merged first"
                              if rules.get("merge_fx_weekend_stubs") else
                              "observations; raw FX Sunday stubs count, so FX gaps can hold one fewer session",
                       "covers":"the one-bar execution delay and MOO3's maximum holding period",
                       "check":"counted per fold in fold_audit" + ("; a shorter gap stops the run"
                                                                  if rules.get("setup_version") else "")},
            "fold_calendar":_calendar(first_test_year, last_test_year, train_years, validation_years,
                                      embargo_sessions),
            "model_selection":{
                "candidates_per_method":{name:len(candidates[name]) for name in METHOD_NAMES},
                "selection_scope":compare.get("selection", "per_seed"),
                "validation_metric":compare.get("validation_metric", "log_wealth_minus_drawdown"),
                "min_validation_trades":compare.get("min_validation_trades", 1),
                "deploy":deploy,
                "test_used_for_selection":False,
            },
            "search_budget":{"population_size":population_size, "generations":generations, "seeds":list(seeds)},
        },
    }
    setup["setup_sha256"] = setup_hash(setup)
    return setup
