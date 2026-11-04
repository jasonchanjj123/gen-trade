"""Independent evolutionary sub-frameworks under one frozen-selection protocol."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import inspect
import os
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd

from .backtest import (BacktestResult, ExecutionConfig, PortfolioSignals,
                       run_backtest, run_portfolio_backtest)

METHOD_NAMES = ("moo3", "mstgam", "igtsp_ring")
DEFAULT_CANDIDATES = {
    "moo3": ({"threshold":.001}, {"threshold":.005}),
    "mstgam": ({"min_votes":3}, {"min_votes":2}),
    "igtsp_ring": ({"groups":12}, {"groups":17}),
}


class NoEligibleConfiguration(ValueError):
    def __init__(self,message,trials):
        super().__init__(message)
        self.trials=trials


class ResearchMethod(Protocol):
    """The bank/representation/objective/operators stay inside each framework."""

    def fit(self, bars: pd.DataFrame, extra_features=None) -> ResearchMethod: ...
    def predict(self, bars: pd.DataFrame, extra_features=None) -> np.ndarray | PortfolioSignals: ...
    def parameters(self) -> dict: ...
    @property
    def diagnostics(self) -> dict: ...


def method_class(name: str):
    from .methods.moo3 import MOO3
    from .methods.mstgam import MSTGAM
    from .methods.igtsp_ring import IGTSPRing
    classes = dict(zip(METHOD_NAMES, (MOO3, MSTGAM, IGTSPRing)))
    if name not in classes:
        raise ValueError(f"Unknown method: {name}")
    return classes[name]


def candidates_for(name: str, supplied: Mapping | None, validation: bool) -> list[dict]:
    if supplied is not None and (not isinstance(supplied, Mapping) or set(supplied) - set(METHOD_NAMES)):
        raise ValueError(f"method_configs must map only these method names: {METHOD_NAMES}")
    raw = supplied.get(name) if supplied and name in supplied else DEFAULT_CANDIDATES[name] if validation else ({},)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError(f"{name} needs a nonempty sequence of candidate configurations")
    if not validation and len(raw) != 1:
        raise ValueError("Multiple configurations require a separate validation window")
    reserved = {"population_size", "generations", "seed", "execution", "island_workers", "bank"}
    allowed = set(inspect.signature(method_class(name)).parameters) - reserved
    candidates = []
    for config in raw:
        if not isinstance(config, Mapping) or set(config) - allowed:
            raise ValueError(f"Invalid/reserved {name} candidate fields; allowed fields: {sorted(allowed)}")
        candidates.append(dict(config))
    return candidates


@dataclass
class ExperimentJob:
    method: str
    history: pd.DataFrame
    features: dict[str, np.ndarray]
    train_stop: int
    validation_start: int | None
    validation_stop: int | None
    test_start: int
    population_size: int
    generations: int
    seed: int
    execution: ExecutionConfig
    candidates: list[dict]
    island_workers: int = 1
    stress_execution: ExecutionConfig | None = None


def _features(job: ExperimentJob, stop: int) -> dict | None:
    return {key:values[:stop] for key,values in job.features.items()} if job.method == "moo3" else None


def evaluate_window(model: ResearchMethod, job: ExperimentJob, start: int, stop: int,
                    execution: ExecutionConfig | None = None) -> BacktestResult:
    # Never pass a future suffix to prediction during validation. Test starts
    # flat, but the indicators replay all preceding, already available candles.
    history = job.history.iloc[:stop].reset_index(drop=True)
    prediction = model.predict(history, extra_features=_features(job, stop))
    bars = history.iloc[start:stop].reset_index(drop=True)
    if isinstance(prediction, PortfolioSignals):
        return run_portfolio_backtest(bars, prediction.slice(start, stop), execution or job.execution)
    return run_backtest(bars, np.asarray(prediction)[start:stop], execution or job.execution)


def validation_score(result: BacktestResult) -> float | None:
    """Shared configuration score, separate from each method's training fitness."""
    if not result.trades:
        return None
    drawdown = np.max(1 - result.equity / np.maximum.accumulate(result.equity))
    return float(np.log(result.equity[-1] / result.equity[0]) - drawdown)


def run_experiment(job: ExperimentJob) -> tuple[dict, BacktestResult]:
    start_time = perf_counter()
    factory = method_class(job.method)
    trials = []
    winner, best_score, selected_model = None, -np.inf, None
    for index, config in enumerate(job.candidates):
        options = dict(population_size=job.population_size, generations=job.generations,
                       seed=job.seed, execution=job.execution, **config)
        if job.method == "igtsp_ring":
            options.setdefault("island_count", min(4, job.population_size // 2))
            options["island_workers"] = job.island_workers
        model = factory(**options)
        trial = {"index":index, "configuration":config}
        try:
            model.fit(job.history.iloc[:job.train_stop].reset_index(drop=True),
                      extra_features=_features(job, job.train_stop))
        except ValueError as error:
            # Failed search candidates are retained, never silently replaced.
            trial.update(status="failed_fit", reason=str(error), validation_score=None)
            trials.append(trial)
            continue
        trial["training"] = model.diagnostics
        trial["parameters"] = model.parameters()
        if job.validation_start is not None:
            result = evaluate_window(model, job, job.validation_start, job.validation_stop)
            score = validation_score(result)
            trial.update(validation_score=score, validation_net_return=result.total_return,
                         validation_trade_count=len(result.trades),
                         status="eligible" if score is not None else "no_validation_trades")
        else:
            score = 0.0
            trial.update(status="fixed_configuration", validation_score=None)
        trials.append(trial)
        # Stable first-candidate tie handling; every method is reported, no
        # framework or seed is selected using held-out outcomes.
        if score is not None and score > best_score:
            winner, best_score, selected_model = index, score, model
    if selected_model is None:
        reasons = "; ".join(trial.get("reason", trial["status"]) for trial in trials)
        raise NoEligibleConfiguration(f"{job.method}: no eligible configuration ({reasons})",trials)
    if job.validation_stop is not None:
        # Freeze configuration, then start a NEW fit on past train+validation.
        config = job.candidates[winner]
        options = dict(population_size=job.population_size, generations=job.generations,
                       seed=job.seed, execution=job.execution, **config)
        if job.method == "igtsp_ring":
            options.setdefault("island_count", min(4, job.population_size // 2))
            options["island_workers"] = job.island_workers
        selected_model = factory(**options).fit(
            job.history.iloc[:job.validation_stop].reset_index(drop=True),
            extra_features=_features(job, job.validation_stop),
        )
    result = evaluate_window(selected_model, job, job.test_start, len(job.history))
    method_report = {
        "parameters":selected_model.parameters(), "training":selected_model.diagnostics,
        "selection":{"protocol":"train/validation/refit/test" if job.validation_stop is not None else "fixed/train/test",
                     "score":"log(net wealth multiple) minus max drawdown; >=1 validation trade",
                     "selected_candidate":winner, "trials":trials,
                     "refit_rows":job.validation_stop if job.validation_stop is not None else job.train_stop,
                     "test_used_for_selection":False},
        "worker_pid":os.getpid(), "elapsed_seconds":perf_counter() - start_time,
        "trade_statistics_note":"Unweighted completed-trade fractions pooled across active fixed-capital sleeves."
                                if job.method == "igtsp_ring" else "Completed single-position trade fractions.",
    }
    if job.method == "igtsp_ring" and hasattr(selected_model, "evaluate_paper"):
        method_report["paper_held_out"] = selected_model.evaluate_paper(job.history, start=job.test_start)
    if job.stress_execution is not None:
        method_report['_stress_result']=evaluate_window(selected_model,job,job.test_start,len(job.history),job.stress_execution)
    return method_report, result


def run_experiment_or_cash(job: ExperimentJob):
    """Predeclared deployment fallback; retain inactive/failed trials and test cash."""
    try:
        return run_experiment(job)
    except NoEligibleConfiguration as error:
        recognized=('no eligible','no valid','two completed','two closed','no_validation_trades')
        if any(t['status']=='failed_fit' and not any(word in t.get('reason','').lower() for word in recognized)
               for t in error.trials):
            raise
        test=job.history.iloc[job.test_start:].reset_index(drop=True)
        result=run_backtest(test,np.zeros(len(test),dtype=np.int8),job.execution)
        report={'status':'cash_fallback_no_eligible_configuration','reason':str(error),
                'parameters':{'method':job.method,'seed':job.seed,'population_size':job.population_size,
                              'generations':job.generations,'execution':__import__('dataclasses').asdict(job.execution)},
                'training':{'fitted':False},'selection':{'selected_candidate':None,'trials':error.trials,
                'test_used_for_selection':False,'fallback':'flat cash; no test-based retuning'},
                'worker_pid':os.getpid(),'elapsed_seconds':None}
        if job.stress_execution is not None:
            report['_stress_result']=run_backtest(test,np.zeros(len(test),dtype=np.int8),job.stress_execution)
        return report,result
