"""Chronological, common-setting evaluation without held-out strategy selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from . import __version__
from .backtest import ExecutionConfig, run_backtest
from .data import ChromosomeContext, align_features, validate_bars
from .experiments import METHOD_NAMES, ExperimentJob, candidates_for, run_experiment, run_experiment_or_cash
from .parallel import process_pool
from .instruments import InstrumentSpec
from .statistics import detailed_statistics, holm_adjust, paired_comparison


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_stop: int
    test_start: int
    test_stop: int
    validation_start: int | None = None
    validation_stop: int | None = None


def walk_forward_folds(n: int, train_bars: int, test_bars: int, *, embargo_bars: int = 0,
                       max_folds: int | None = None, validation_bars: int = 0) -> list[Fold]:
    for name, value in (("n", n), ("train_bars", train_bars), ("test_bars", test_bars)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(embargo_bars) is not int or embargo_bars < 0:
        raise ValueError("embargo_bars must be a nonnegative integer")
    if type(validation_bars) is not int or validation_bars < 0:
        raise ValueError("validation_bars must be a nonnegative integer")
    if max_folds is not None and (type(max_folds) is not int or max_folds < 1):
        raise ValueError("max_folds must be a positive integer or None")
    folds = []
    end = train_bars
    while True:
        validation_start = end + embargo_bars if validation_bars else None
        validation_stop = validation_start + validation_bars if validation_bars else None
        start = (validation_stop if validation_bars else end) + embargo_bars
        if start + test_bars > n:
            break
        folds.append(Fold(end - train_bars, end, start, start + test_bars,
                          validation_start, validation_stop))
        if max_folds is not None and len(folds) >= max_folds:
            break
        end += test_bars
    if not folds:
        raise ValueError("Insufficient observations for one full train/validation/embargo/test fold")
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
    validation_bars: int = 60, workers: int = 3, island_workers: int = 1,
    method_configs: dict | None = None, seeds: tuple[int, ...] | None = None,
    instrument: InstrumentSpec | None = None, initial_cash: float = 10000.,
    fold_plan: list[Fold] | None = None, include_paths: bool = False,
    stress_instrument: InstrumentSpec | None = None,
    on_no_eligible: str = 'raise',
) -> dict:
    """Compare documented adaptations; common costs/settings, frozen training fits.

    Fresh cash/positions per fold. Indicators see past training/embargo observations
    as warm-up, but the training scaler/calibration/selection never sees test rows.
    Neither folds nor seeds are ranked to pick a winner from held-out returns.
    """
    for name, value, lower in (("population_size", population_size, 4), ("generations", generations, 0),
                               ("seed", seed, 0), ("workers", workers, 1), ("island_workers", island_workers, 1)):
        if type(value) is not int or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if on_no_eligible not in ('raise','cash'):
        raise ValueError('on_no_eligible must be raise or cash')
    if workers > 1 and island_workers > 1:
        raise ValueError("Use workers=1 for parallel islands; avoid nested experiment/island process pools")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    if type(bootstrap_reps) is not int or bootstrap_reps < 100:
        raise ValueError("bootstrap_reps must be an integer of at least 100")
    run_seeds = tuple(seeds) if seeds is not None else (seed,)
    if (not run_seeds or len(set(run_seeds)) != len(run_seeds)
            or any(type(s) is not int or s < 0 for s in run_seeds)):
        raise ValueError("seeds must be distinct nonnegative integers")
    configs = {name:candidates_for(name, method_configs, validation_bars > 0) for name in METHOD_NAMES}
    if context.pair not in markets:
        raise ValueError(f"Unknown trading pair: {context.pair}")
    for symbol, source in markets.items():
        checked = validate_bars(source)
        if "symbol" not in checked or str(checked["symbol"].iloc[0]) != symbol:
            raise ValueError("Market mapping keys must match each frame's symbol")
    bars = validate_bars(markets[context.pair])
    folds = fold_plan if fold_plan is not None else walk_forward_folds(len(bars),train_bars,test_bars,
                embargo_bars=embargo_bars,max_folds=max_folds,validation_bars=validation_bars)
    if not folds:
        raise ValueError('At least one complete fold is required')
    for fold in folds:
        if not 0 <= fold.train_start < fold.train_stop <= fold.test_start < fold.test_stop <= len(bars):
            raise ValueError('Invalid fold bounds')
        if fold.validation_start is not None and not fold.train_stop <= fold.validation_start < fold.validation_stop <= fold.test_start:
            raise ValueError('Invalid validation bounds')
    if any(a.test_stop>b.test_start for a,b in zip(folds,folds[1:])):
        raise ValueError('Held-out folds must not overlap')
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
        "schema_version":3, "software_version":__version__,
        "source_and_lock_sha256":source_hash.hexdigest(),
        "runtime":{"python":platform.python_version(),"platform":platform.platform(),
                   "dc_backend":"rust" if dc._native_dc_arrays is not None else "python",
                   "native_binary_sha256":native_hash,
                   "packages":{name:version(name) for name in ("numpy","pandas","scipy","statsmodels","arch","TA-Lib")}},
        "data_kind":"synthetic test fixture" if synthetic else "user supplied; not independently audited",
        "dataset_sha256":data_hash.hexdigest(), "seed":seed, "seeds":list(run_seeds),
        "methods":list(METHOD_NAMES), "chromosome_context":context.to_dict(),
        "protocol": {"train_bars":train_bars if fold_plan is None else None,
                     "test_bars":test_bars if fold_plan is None else None,"embargo_bars":embargo_bars,
                     "validation_bars":validation_bars if fold_plan is None else None,
                     "on_no_eligible":on_no_eligible,
                     "fold_plan_source":"explicit calendar plan" if fold_plan is not None else "rolling bar counts",
                     "instrument":asdict(instrument) if instrument else None,"initial_cash":initial_cash,
                     "stress_instrument":asdict(stress_instrument) if stress_instrument else None,
                     "population_size":population_size,"generations":generations,
                     "periods_per_year":periods_per_year,"common_fee_per_side":fee,
                     "slippage_bps":slippage_bps,"folds_start_flat":True,
                     "parallelism":{"experiment_workers":workers,"island_workers":island_workers,
                                    "backend":"spawn_processes" if workers > 1 or island_workers > 1 else "serial",
                                    "numerical_threads_per_process_worker":1},
                     "candidate_configurations":configs,
                     "budget_note":"Equal population/generations per fit; ring population is TOTAL across islands. Calibration, strategy-bank backtests, candidate counts and runtime differ.",
                     "comparison":"research adaptations; not numerical paper reproductions"},
        "folds":[],
    }
    pending = []
    records = []
    for fold_index, fold in enumerate(folds):
        train = bars.iloc[fold.train_start:fold.train_stop].reset_index(drop=True)
        test = bars.iloc[fold.test_start:fold.test_stop].reset_index(drop=True)
        history = bars.iloc[fold.train_start:fold.test_stop].reset_index(drop=True)
        features = align_features(history, markets, context.features)
        executions = {
            "moo3":ExecutionConfig(fee=fee,slippage_bps=slippage_bps,max_holding_days=5,
                                   take_profit=.05,entry_hours=context.entry_hours,initial_cash=initial_cash,instrument=instrument),
            "mstgam":ExecutionConfig(fee=fee,slippage_bps=slippage_bps,entry_hours=context.entry_hours,initial_cash=initial_cash,instrument=instrument),
            "igtsp_ring":ExecutionConfig(fee=fee,slippage_bps=slippage_bps,entry_hours=context.entry_hours,initial_cash=initial_cash,instrument=instrument),
        }
        for base_seed in run_seeds:
            fold_seed = base_seed + fold_index
            fold_report = {"index":fold_index,"base_seed":base_seed,"effective_seed":fold_seed,"bounds":asdict(fold),
                       "train_last_known_at":train["available_at"].iloc[-1].isoformat(),
                       "test_first_open":test["timestamp"].iloc[0].isoformat(), "methods":{},
                       "validation_first_open":bars["timestamp"].iloc[fold.validation_start].isoformat() if validation_bars else None,
                       "refit_last_known_at":bars["available_at"].iloc[fold.validation_stop-1].isoformat() if validation_bars else None}
            record_index = len(records)
            records.append((fold_report, test, executions, fold_seed))
            for name in METHOD_NAMES:
                job = ExperimentJob(name, history, features, len(train),
                                    fold.validation_start-fold.train_start if validation_bars else None,
                                    fold.validation_stop-fold.train_start if validation_bars else None,
                                    fold.test_start-fold.train_start, population_size, generations, fold_seed,
                                    executions[name],configs[name],island_workers,
                                    replace(executions[name],instrument=stress_instrument,slippage_bps=slippage_bps*2)
                                    if stress_instrument else None)
                pending.append((record_index, name, job))
    pool = process_pool(min(workers, len(pending))) if workers > 1 else None
    run_job=run_experiment_or_cash if on_no_eligible=='cash' else run_experiment
    results_by_record = [{} for _ in records]
    try:
        # Submit all independent method/fold/seed jobs before awaiting results.
        tasks = [(index, name, pool.submit(run_job, job)) for index, name, job in pending] if pool else pending
        for index, name, task in tasks:
            method_report, result = task.result() if pool else run_job(task)
            fold_report, _, _, fold_seed = records[index]
            method_report["statistics"] = detailed_statistics(result,periods_per_year=periods_per_year,
                                                             bootstrap_reps=bootstrap_reps,seed=fold_seed)
            if include_paths:
                method_report['backtest']=export_backtest(result)
            stress_result=method_report.pop('_stress_result',None)
            if stress_result is not None:
                method_report['cost_stress']={
                    'statistics':detailed_statistics(stress_result,periods_per_year=periods_per_year,
                                                    bootstrap_reps=bootstrap_reps,seed=fold_seed),
                    'frozen_model':True,'execution':asdict(stress_result.config)}
                if include_paths:method_report['cost_stress']['backtest']=export_backtest(stress_result)
            fold_report["methods"][name] = method_report
            results_by_record[index][name] = result
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    for (fold_report, test, _, fold_seed), results in zip(records, results_by_record):
        pairs = {}
        for first, second in combinations(METHOD_NAMES, 2):
            pairs[f"{first}_minus_{second}"] = paired_comparison(
                results[first],results[second],periods_per_year=periods_per_year,
                bootstrap_reps=bootstrap_reps,seed=fold_seed,
            )
        adjusted = holm_adjust(pair["paired_return_difference_inference"]["hac_pvalue_mean_zero"] for pair in pairs.values())
        for pair, pvalue in zip(pairs.values(), adjusted):
            pair["holm_hac_pvalue_within_fold_seed"] = pvalue
            pair["holm_family"] = "Three pairwise method comparisons within this fold/seed; excludes adaptive search and other runs."
        fold_report["paired_comparisons"] = pairs
        # Keep the former two-method key for existing report consumers.
        fold_report["paired_moo3_minus_mstgam"] = pairs["moo3_minus_mstgam"]
        baseline = run_backtest(test, np.ones(len(test),dtype=np.int8),
                                ExecutionConfig(fee=fee,slippage_bps=slippage_bps,entry_hours=context.entry_hours,
                                                instrument=instrument,initial_cash=initial_cash))
        fold_report["baselines"] = {"next_open_buy_and_hold":detailed_statistics(
            baseline,periods_per_year=periods_per_year,bootstrap_reps=bootstrap_reps,seed=fold_seed)}
        if include_paths:fold_report['baseline_backtest']=export_backtest(baseline)
        if stress_instrument:
            stress_baseline=run_backtest(test,np.ones(len(test),dtype=np.int8),replace(baseline.config,
                instrument=stress_instrument,slippage_bps=slippage_bps*2))
            fold_report['baseline_cost_stress']={'statistics':detailed_statistics(stress_baseline,
                periods_per_year=periods_per_year,bootstrap_reps=bootstrap_reps,seed=fold_seed),
                'backtest':export_backtest(stress_baseline) if include_paths else None}
        report["folds"].append(fold_report)
    # No pooled significance claim: correlated rolling training folds are reported
    # individually. The report preserves their identities for multi-asset analysis.
    report["limitations"] = [
        "Synthetic runs validate software only; real-market profitability is untested." if synthetic
        else "Data quality, market calendars, survivorship, and execution realism need independent audit.",
        "The supplied seed set and rolling validation protocol do not reproduce the papers' full multi-asset experiments.",
        "Pair and entry-session choices are experiment context, not newly evolved paper genes.",
        "MOO3 can consume lagged cross-market OHLCV terminals; MSTGAM retains the original 70-weight representation.",
        "IGTSP-Ring uses full-text Eqs.6-16 and Table 5 defaults; the paper's alternative Section 3.1 objective, omitted rule/ranking details and execution assumptions are documented.",
        "Ring portfolios divide one instrument's capital among fixed strategy sleeves; there is no cross-asset allocation.",
        "Instrument costs and financing are declared research assumptions; actual broker swap histories are not inferred." if instrument
        else "No production order routing, partial fills, or funding/borrow costs.",
        "No claim of deflated Sharpe, probability of backtest overfitting, or full paper baseline reproduction.",
    ]
    # Fail before emitting a misleading report containing NaN/Infinity.
    return json.loads(json.dumps(report, allow_nan=False))


def export_backtest(result) -> dict:
    trades=[]
    for trade in result.trades:
        row=asdict(trade)
        for key in ('entry_time','exit_time'):row[key]=row[key].isoformat()
        trades.append(row)
    return {'initial_cash':result.config.initial_cash,'timestamps':[t.isoformat() for t in result.timestamps],
            'equity':result.equity.tolist(),'trades':trades,'ledger':result.ledger}
