"""Chronological, common-setting evaluation without held-out strategy selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from . import __version__
from .backtest import ExecutionConfig, run_backtest
from .data import ChromosomeContext, align_features, validate_bars
from .statistics import detailed_statistics, paired_comparison


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_stop: int
    test_start: int
    test_stop: int


def walk_forward_folds(n: int, train_bars: int, test_bars: int, *, embargo_bars: int = 0,
                       max_folds: int | None = None) -> list[Fold]:
    for name, value in (("n", n), ("train_bars", train_bars), ("test_bars", test_bars)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(embargo_bars) is not int or embargo_bars < 0:
        raise ValueError("embargo_bars must be a nonnegative integer")
    if max_folds is not None and (type(max_folds) is not int or max_folds < 1):
        raise ValueError("max_folds must be a positive integer or None")
    folds = []
    end = train_bars
    while end + embargo_bars + test_bars <= n:
        start = end + embargo_bars
        folds.append(Fold(end - train_bars, end, start, start + test_bars))
        if max_folds is not None and len(folds) >= max_folds:
            break
        end += test_bars
    if not folds:
        raise ValueError("Insufficient observations for one full training/embargo/test fold")
    return folds


def synthetic_markets(n: int = 480, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Deterministic TEST FIXTURE with invented gold/FX/crypto prices and volumes."""
    if n < 100:
        raise ValueError("Synthetic demo requires at least 100 rows")
    rng = np.random.default_rng(seed)
    timestamps = pd.bdate_range("2020-01-01 09:00", periods=n, tz="UTC")
    result = {}
    for symbol, base, phase in (("BTC/USDT", 30000., 0.), ("XAU/USD", 1800., .8), ("EUR/USD", 1.1, 1.7)):
        # Cycles deliberately exercise both reversals and exits; they are not market estimates.
        close = base * np.exp(.04 * np.sin(np.arange(n) * .31 + phase) + np.cumsum(rng.normal(0, .002, n)))
        opening = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, .001, n))
        frame = pd.DataFrame({
            "symbol":symbol, "timestamp":timestamps,
            "available_at":timestamps + pd.Timedelta(hours=8),
            "open":opening, "high":np.maximum(opening,close)*1.002,
            "low":np.minimum(opening,close)*.998, "close":close,
            "volume":rng.uniform(100,1000,n),
        })
        result[symbol] = validate_bars(frame)
    return result


def compare_methods(
    markets: dict[str, pd.DataFrame], context: ChromosomeContext, *,
    train_bars: int, test_bars: int, embargo_bars: int = 5, max_folds: int = 3,
    population_size: int = 16, generations: int = 2, seed: int = 42,
    fee: float = .0025, slippage_bps: float = 0.0, periods_per_year: float = 252.,
    bootstrap_reps: int = 300, synthetic: bool = False,
) -> dict:
    """Compare documented adaptations; common costs/settings, frozen training fits.

    Fresh cash/positions per fold. Indicators see past training/embargo observations
    as warm-up, but the training scaler/calibration/selection never sees test rows.
    Neither folds nor seeds are ranked to pick a winner from held-out returns.
    """
    from .methods.moo3 import MOO3
    from .methods.mstgam import MSTGAM

    if context.pair not in markets:
        raise ValueError(f"Unknown trading pair: {context.pair}")
    for symbol, source in markets.items():
        checked = validate_bars(source)
        if "symbol" not in checked or str(checked["symbol"].iloc[0]) != symbol:
            raise ValueError("Market mapping keys must match each frame's symbol")
    bars = validate_bars(markets[context.pair])
    folds = walk_forward_folds(len(bars), train_bars, test_bars, embargo_bars=embargo_bars, max_folds=max_folds)
    data_hash = hashlib.sha256()
    for symbol, source in sorted(markets.items()):
        data_hash.update(symbol.encode())
        data_hash.update(pd.util.hash_pandas_object(source, index=False).values.tobytes())
    source_hash = hashlib.sha256()
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    code_files = list(package.rglob("*.py"))
    code_files.extend((root / "native" / "src").rglob("*.rs"))
    code_files.extend(path for path in (root/"pyproject.toml",root/"uv.lock",root/"native"/"Cargo.toml",
                                       root/"native"/"Cargo.lock") if path.exists())
    for path in sorted(code_files):
        source_hash.update(str(path.relative_to(root)).encode())
        source_hash.update(path.read_bytes())
    from . import dc
    native_hash = None
    if dc._native_dc_arrays is not None:
        import gen_trade_native
        location = Path(gen_trade_native.__file__)
        libraries = [location] if location.suffix == ".so" else sorted(location.parent.glob("*.so"))
        digest = hashlib.sha256()
        for library in libraries:
            digest.update(library.read_bytes())
        native_hash = digest.hexdigest() if libraries else None
    report = {
        "schema_version":1, "software_version":__version__,
        "source_and_lock_sha256":source_hash.hexdigest(),
        "runtime":{"python":platform.python_version(),"platform":platform.platform(),
                   "dc_backend":"rust" if dc._native_dc_arrays is not None else "python",
                   "native_binary_sha256":native_hash,
                   "packages":{name:version(name) for name in ("numpy","pandas","scipy","statsmodels","arch","TA-Lib")}},
        "data_kind":"synthetic test fixture" if synthetic else "user supplied; not independently audited",
        "dataset_sha256":data_hash.hexdigest(), "seed":seed, "chromosome_context":context.to_dict(),
        "protocol": {"train_bars":train_bars,"test_bars":test_bars,"embargo_bars":embargo_bars,
                     "population_size":population_size,"generations":generations,
                     "periods_per_year":periods_per_year,"common_fee_per_side":fee,
                     "slippage_bps":slippage_bps,"folds_start_flat":True,
                     "budget_note":"Equal evolutionary population/generations, not equal total compute; MSTGAM also calibrates its strategy cutoffs.",
                     "comparison":"research adaptations; not numerical paper reproductions"},
        "folds":[],
    }
    for fold_index, fold in enumerate(folds):
        train = bars.iloc[fold.train_start:fold.train_stop].reset_index(drop=True)
        test = bars.iloc[fold.test_start:fold.test_stop].reset_index(drop=True)
        history = bars.iloc[fold.train_start:fold.test_stop].reset_index(drop=True)
        features = align_features(history, markets, context.features)
        train_len = len(train)
        extra_train = {key:values[:train_len] for key,values in features.items()}
        fold_seed = seed + fold_index
        executions = {
            "moo3":ExecutionConfig(fee=fee,slippage_bps=slippage_bps,max_holding_days=5,
                                   take_profit=.05,entry_hours=context.entry_hours),
            "mstgam":ExecutionConfig(fee=fee,slippage_bps=slippage_bps,entry_hours=context.entry_hours),
        }
        models = {
            "moo3":MOO3(population_size=population_size,generations=generations,seed=fold_seed,
                        execution=executions["moo3"]),
            "mstgam":MSTGAM(population_size=population_size,generations=generations,seed=fold_seed,
                            execution=executions["mstgam"]),
        }
        fold_report = {"index":fold_index,"bounds":asdict(fold),
                       "train_last_known_at":train["available_at"].iloc[-1].isoformat(),
                       "test_first_open":test["timestamp"].iloc[0].isoformat(), "methods":{}}
        results = {}
        for name, model in models.items():
            # Cross-market/volume terminals extend MOO3. MSTGAM deliberately retains its
            # published 70 weights and accepts no additional terminal representation.
            extra_fit = extra_train if name == "moo3" else None
            extra_predict = features if name == "moo3" else None
            model.fit(train, extra_features=extra_fit)
            signals = np.asarray(model.predict(history, extra_features=extra_predict))
            test_signals = signals[fold.test_start-fold.train_start:]
            results[name] = run_backtest(test, test_signals, executions[name])
            fold_report["methods"][name] = {
                "parameters":model.parameters(), "training":model.diagnostics,
                "statistics":detailed_statistics(results[name],periods_per_year=periods_per_year,
                                                 bootstrap_reps=bootstrap_reps,seed=fold_seed),
            }
        comparison = paired_comparison(results["moo3"],results["mstgam"],
                                       periods_per_year=periods_per_year,
                                       bootstrap_reps=bootstrap_reps,seed=fold_seed)
        fold_report["paired_moo3_minus_mstgam"] = comparison
        baseline = run_backtest(
            test, np.ones(len(test), dtype=np.int8),
            ExecutionConfig(fee=fee,slippage_bps=slippage_bps,entry_hours=context.entry_hours),
        )
        fold_report["baselines"] = {
            "next_open_buy_and_hold": detailed_statistics(
                baseline,periods_per_year=periods_per_year,bootstrap_reps=bootstrap_reps,seed=fold_seed
            )
        }
        report["folds"].append(fold_report)
    # No pooled significance claim: correlated rolling training folds are reported
    # individually. The report preserves their identities for multi-asset analysis.
    report["limitations"] = [
        "Synthetic runs validate software only; real-market profitability is untested." if synthetic
        else "Data quality, market calendars, survivorship, and execution realism need independent audit.",
        "One seed per fold does not reproduce the papers' 50-run multi-asset experiments.",
        "Pair and entry-session choices are experiment context, not newly evolved paper genes.",
        "MOO3 can consume lagged cross-market OHLCV terminals; MSTGAM retains the original 70-weight representation.",
        "No production order routing, partial fills, funding/borrow costs, or cross-asset portfolio allocation.",
        "No claim of deflated Sharpe, probability of backtest overfitting, or full paper baseline reproduction.",
    ]
    # Fail before emitting a misleading report containing NaN/Infinity.
    return json.loads(json.dumps(report, allow_nan=False))
