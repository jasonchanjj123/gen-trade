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
# Labeled EXTENSION profile (docs/EXTENSIONS.md). Each candidate keeps the
# paper's validated knob and switches on the theory/audit-motivated variants.
_MOO3_EXT = {"objectives":"robust", "min_trades_per_year":3., "survival":"nsga3", "feature_transform":"scale_free"}
_MSTGAM_EXT = {"fitness":"tstat", "sign_adjusted":True, "mutation":"lognormal"}
_RING_EXT = {"fitness":"mixture_drawdown", "sign_adjusted":True, "crossover_mode":"group", "mutation_mode":"bitwise",
             "unique_children":True, "bank_selection":"diverse", "bank_ranking":"two_stage", "migration_schedule":"adaptive"}
IMPROVED_CANDIDATES = {
    "moo3": ({**_MOO3_EXT, "threshold_volatility_multiple":1.}, {**_MOO3_EXT, "threshold_volatility_multiple":2.}),
    "mstgam": ({**_MSTGAM_EXT, "min_votes":3}, {**_MSTGAM_EXT, "min_votes":2}),
    "igtsp_ring": ({**_RING_EXT, "groups":12}, {**_RING_EXT, "groups":17}),
}
# Review 2026-11-05 profile: every method's search scores excess over an
# exposure-matched blend on resampled training months (methods/robust_fitness.py);
# Ring uses equal group weights. Validation knobs are unchanged.
_MOO3_ROBUST = {"objectives":"blend_excess", "complexity_penalty":.01, "min_trades_per_year":3.,
                "feature_transform":"scale_free"}
_MSTGAM_ROBUST = {"fitness":"blend_excess", "mutation":"lognormal", "min_trades_per_year":3.}
_RING_ROBUST = {**_RING_EXT, "fitness":"blend_excess", "weight_mode":"equal"}
ROBUST_CANDIDATES = {
    "moo3": ({**_MOO3_ROBUST, "threshold_volatility_multiple":1.}, {**_MOO3_ROBUST, "threshold_volatility_multiple":2.}),
    "mstgam": ({**_MSTGAM_ROBUST, "min_votes":3}, {**_MSTGAM_ROBUST, "min_votes":2}),
    "igtsp_ring": ({**_RING_ROBUST, "groups":12}, {**_RING_ROBUST, "groups":17}),
}
CANDIDATE_PROFILES = {"paper":DEFAULT_CANDIDATES, "improved":IMPROVED_CANDIDATES, "robust":ROBUST_CANDIDATES}
VALIDATION_METRICS = ("log_wealth_minus_drawdown", "blend_excess")
DEPLOY_POLICIES = ("refit", "validated")


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
    deploy: str = "refit"
    min_validation_trades: int = 1
    validation_curve: bool = False
    validation_metric: str = "log_wealth_minus_drawdown"


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


def validation_score(result: BacktestResult, min_trades: int = 1) -> float | None:
    """Shared configuration score, separate from each method's training fitness."""
    if len(result.trades) < max(1, min_trades):
        return None
    drawdown = np.max(1 - result.equity / np.maximum.accumulate(result.equity))
    return float(np.log(result.equity[-1] / result.equity[0]) - drawdown)


def blend_excess_score(result: BacktestResult, bars: pd.DataFrame, execution: ExecutionConfig,
                       min_trades: int = 1) -> float | None:
    """EXTENSION (review 2026-11-05): validation information ratio of the
    excess over a cash/buy-and-hold blend with the strategy's mean exposure."""
    from .methods.robust_fitness import benchmark_returns, excess_information_ratio, period_returns, periods_per_year
    from .statistics import exposure_path
    if len(result.trades) < max(1, min_trades):
        return None
    score = excess_information_ratio(period_returns(result.equity), exposure_path(result),
                                     benchmark_returns(bars, execution), periods_per_year=periods_per_year(bars))[0]
    return float(score)


def _score(job: ExperimentJob, result: BacktestResult) -> float | None:
    if job.validation_metric == "blend_excess":
        bars = job.history.iloc[job.validation_start:job.validation_stop].reset_index(drop=True)
        return blend_excess_score(result, bars, job.execution, job.min_validation_trades)
    return validation_score(result, job.min_validation_trades)


def _model(job: ExperimentJob, config: dict):
    options = dict(population_size=job.population_size, generations=job.generations,
                   seed=job.seed, execution=job.execution, **config)
    if job.method == "igtsp_ring":
        options.setdefault("island_count", min(4, job.population_size // 2))
        options["island_workers"] = job.island_workers
    return method_class(job.method)(**options)


def _validation_curve(model, job: ExperimentJob) -> list[dict] | None:
    """Score the best individual of selected generations on the validation window."""
    if not job.validation_curve or job.validation_start is None or not hasattr(model, "generation_snapshots"):
        return None
    curve = []
    for generation, snapshot in model.generation_snapshots():
        result = evaluate_window(snapshot, job, job.validation_start, job.validation_stop)
        curve.append({"generation":generation, "validation_score":_score(job, result),
                      "validation_net_return":result.total_return, "validation_trade_count":len(result.trades)})
    return curve


def run_trials(job: ExperimentJob, models: list | None = None) -> list[dict]:
    """Fit every candidate on training rows and score it on validation only.

    ``models``, when supplied, receives each fitted model (or None) in order so
    an in-process caller can deploy the validated model without refitting.
    """
    trials = []
    for index, config in enumerate(job.candidates):
        model = _model(job, config)
        trial = {"index":index, "configuration":config}
        try:
            model.fit(job.history.iloc[:job.train_stop].reset_index(drop=True),
                      extra_features=_features(job, job.train_stop))
        except ValueError as error:
            # Failed search candidates are retained, never silently replaced.
            trial.update(status="failed_fit", reason=str(error), validation_score=None)
            trials.append(trial)
            if models is not None:
                models.append(None)
            continue
        if models is not None:
            models.append(model)
        trial["training"] = model.diagnostics
        trial["parameters"] = model.parameters()
        if job.validation_start is not None:
            result = evaluate_window(model, job, job.validation_start, job.validation_stop)
            score = _score(job, result)
            trial.update(validation_score=score, validation_net_return=result.total_return,
                         validation_trade_count=len(result.trades),
                         status="eligible" if score is not None else "no_validation_trades")
            curve = _validation_curve(model, job)
            if curve is not None:
                trial["validation_curve"] = curve
        else:
            trial.update(status="fixed_configuration", validation_score=None)
        trials.append(trial)
    return trials


def choose_candidate(trials: list[dict], validation: bool) -> int | None:
    """Stable first-candidate tie handling; held-out outcomes are never used."""
    winner, best = None, -np.inf
    for trial in trials:
        if trial["status"] == "failed_fit":
            continue
        score = trial.get("validation_score") if validation else 0.0
        if score is not None and score > best:
            winner, best = trial["index"], score
    return winner


def pooled_choice(trial_sets: list[list[dict]]) -> tuple[int | None, list[float | None]]:
    """EXTENSION: choose one configuration per method/fold from seed-mean scores.

    A candidate is eligible when a strict majority of seeds produced a
    validation score; its pooled score is the mean over those seeds.
    """
    candidates = len(trial_sets[0])
    pooled = []
    for index in range(candidates):
        scores = [trials[index].get("validation_score") for trials in trial_sets]
        observed = [score for score in scores if score is not None]
        pooled.append(float(np.mean(observed)) if 2 * len(observed) > len(scores) else None)
    winner, best = None, -np.inf
    for index, score in enumerate(pooled):
        if score is not None and score > best:
            winner, best = index, score
    return winner, pooled


def finish_experiment(job: ExperimentJob, trials: list[dict], winner: int | None,
                      pooled_scores: list | None = None, fitted=None) -> tuple[dict, BacktestResult]:
    start_time = perf_counter()
    if winner is None:
        reasons = "; ".join(trial.get("reason", trial["status"]) for trial in trials)
        raise NoEligibleConfiguration(f"{job.method}: no eligible configuration ({reasons})",trials)
    config = job.candidates[winner]
    if job.validation_stop is not None and job.deploy == "refit":
        # Freeze configuration, then start a NEW fit on past train+validation.
        stop, protocol = job.validation_stop, "train/validation/refit/test"
    else:
        # Deterministic re-creation of the exact validated training-only model.
        stop = job.train_stop
        protocol = "train/validation/deploy-validated/test" if job.validation_stop is not None else "fixed/train/test"
    if fitted is not None and stop == job.train_stop:
        selected_model = fitted
    else:
        selected_model = _model(job, config).fit(job.history.iloc[:stop].reset_index(drop=True),
                                                 extra_features=_features(job, stop))
    result = evaluate_window(selected_model, job, job.test_start, len(job.history))
    score_text = (("information ratio of excess over the exposure-matched blend" if job.validation_metric == "blend_excess"
                   else "log(net wealth multiple) minus max drawdown")
                  + f"; >={max(1, job.min_validation_trades)} validation trade"
                  + ("s" if job.min_validation_trades > 1 else ""))
    method_report = {
        "parameters":selected_model.parameters(), "training":selected_model.diagnostics,
        "selection":{"protocol":protocol, "score":score_text,
                     "selected_candidate":winner, "trials":trials,
                     "refit_rows":stop, "test_used_for_selection":False},
        "worker_pid":os.getpid(), "elapsed_seconds":perf_counter() - start_time,
        "trade_statistics_note":"Unweighted completed-trade fractions pooled across active fixed-capital sleeves."
                                if job.method == "igtsp_ring" else "Completed single-position trade fractions.",
    }
    if pooled_scores is not None:
        method_report["selection"]["pooled_seed_mean_scores"] = pooled_scores
        method_report["selection"]["selection_scope"] = "configuration chosen once per method/fold from seed-mean validation scores"
    if job.method == "igtsp_ring" and hasattr(selected_model, "evaluate_paper"):
        method_report["paper_held_out"] = selected_model.evaluate_paper(job.history, start=job.test_start)
    if job.stress_execution is not None:
        method_report['_stress_result']=evaluate_window(selected_model,job,job.test_start,len(job.history),job.stress_execution)
    return method_report, result


def run_experiment(job: ExperimentJob) -> tuple[dict, BacktestResult]:
    start_time = perf_counter()
    models: list = []
    trials = run_trials(job, models)
    winner = choose_candidate(trials, job.validation_start is not None)
    report, result = finish_experiment(job, trials, winner, fitted=models[winner] if winner is not None else None)
    report["elapsed_seconds"] = perf_counter() - start_time
    return report, result


def _cash_report(job: ExperimentJob, error: NoEligibleConfiguration):
    recognized=('no eligible','no valid','two completed','two closed','no_validation_trades','completed trades')
    if any(t['status']=='failed_fit' and not any(word in t.get('reason','').lower() for word in recognized)
           for t in error.trials):
        raise error
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


def finish_or_cash(job: ExperimentJob, trials: list[dict], winner: int | None, pooled_scores=None):
    try:
        return finish_experiment(job, trials, winner, pooled_scores)
    except NoEligibleConfiguration as error:
        return _cash_report(job, error)


def run_experiment_or_cash(job: ExperimentJob):
    """Predeclared deployment fallback; retain inactive/failed trials and test cash."""
    try:
        return run_experiment(job)
    except NoEligibleConfiguration as error:
        return _cash_report(job, error)
