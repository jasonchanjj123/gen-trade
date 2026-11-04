"""MSTGAM: a causal research adaptation of Salman et al. (2025).

DOI: 10.1007/s10462-025-11419-z, sections 4--5. The original chromosome
contains 70 weights over eight DC strategies, not arbitrary OHLCV features.
This implementation therefore accepts but does not use external GP terminals.

Adaptations are explicit: online confirmed DC states replace retrospective OS
labels; ambiguous per-gene holdings become raw opinions with a final shared
execution guard; action ties hold; ``min_votes=3`` follows the detailed text's
"more than two" rule. Calendar durations use observation availability times.
St6 uses fixed timestamp-keyed random draws across fitness evaluations.
The shared simulator fills next opens and reports terminal liquidations.

Labeled EXTENSIONS (off by default; docs/EXTENSIONS.md): ``fitness="tstat"``
divides by s*sqrt(n) instead of the population std, so the score is the
t-statistic of training trades in excess of an annual risk-free rate prorated
over the training span (audit item 10); ``sign_adjusted`` applies Israelsen's
(2005) refinement for negative numerators; ``mutation="lognormal"`` replaces
the uniform one-gene redraw (expected new vote share 1 - ln 2) with a
multiplicative heavy-tailed step.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gen_trade.backtest import ExecutionConfig, run_backtest
from gen_trade.data import validate_bars
from gen_trade.dc import DCProfile, directional_changes


THRESHOLDS = (0.00098, 0.0022, 0.0048, 0.0072, 0.0098,
              0.0122, 0.0155, 0.0170, 0.0200, 0.0255)
GENES = tuple((strategy, index) for strategy in range(1, 9)
              for index in range(10 if strategy <= 6 else 5))
INVALID_FITNESS = -1e12


def normalise_weights(weights: np.ndarray) -> np.ndarray:
    """Return a finite simplex vector; an all-zero proposal becomes uniform."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be a nonempty finite nonnegative vector")
    total = float(values.sum())
    if not np.isfinite(total):
        raise ValueError("weight sum must be finite")
    return values / total if total > 0 else np.full(values.size, 1.0 / values.size)


def aggregate_signals(opinions: np.ndarray, weights: np.ndarray, min_votes: int = 3) -> np.ndarray:
    """Weighted Buy/Sell/Hold votes; active-count override and ties are explicit.

    Zero-weight genes still count toward the paper's active-gene threshold.
    Opinions are observation-by-gene and use the shared +1/-1/0 convention.
    """
    opinion = np.asarray(opinions)
    w = normalise_weights(weights)
    if opinion.ndim != 2 or opinion.shape[1] != w.size or not np.isin(opinion, [-1, 0, 1]).all():
        raise ValueError("opinions must be an n-by-gene matrix containing only -1,0,1")
    if not isinstance(min_votes, (int, np.integer)) or isinstance(min_votes, bool) or min_votes < 1:
        raise ValueError("min_votes must be a positive integer")
    buy = (opinion == 1) @ w
    sell = (opinion == -1) @ w
    hold = (opinion == 0) @ w
    active = np.count_nonzero(opinion, axis=1) >= min_votes
    return _vote_totals(buy, sell, hold, active)


def _vote_totals(buy: np.ndarray, sell: np.ndarray, hold: np.ndarray, active: np.ndarray) -> np.ndarray:
    # A rounding residue in a unit-sum vector must not turn a mathematical tie
    # into an order. Use the same rule for batch fitting and single inference.
    eps = 16 * np.finfo(float).eps
    signals = np.zeros(buy.shape, dtype=np.int8)
    signals[(buy > sell + eps) & (active | (buy > hold + eps))] = 1
    signals[(sell > buy + eps) & (active | (sell > hold + eps))] = -1
    return signals


FITNESS_VARIANTS = ("paper", "tstat")
MUTATION_VARIANTS = ("paper", "lognormal")


def trade_fitness(count: int, total: float, population_std: float, *, risk_free_return: float = 0.025,
                  variant: str = "paper", sign_adjusted: bool = False, years: float = 1.0) -> float:
    """Shared paper/extension trade score from sufficient statistics.

    paper: (sum q - Rf) / population std, Eqs.9/12.
    tstat: (sum q - Rf_annual * years) / (sample std * sqrt(n)), the t-statistic.
    sign_adjusted: for a negative numerator multiply by the denominator instead
    (Israelsen 2005), so more risk can no longer improve a losing score.
    """
    if count < 2 or not np.isfinite(total) or not np.isfinite(population_std) or population_std <= 1e-12:
        return INVALID_FITNESS
    if variant == "paper":
        numerator, denominator = total - risk_free_return, population_std
    elif variant == "tstat":
        sample_std = population_std * np.sqrt(count / (count - 1))
        numerator, denominator = total - risk_free_return * years, sample_std * np.sqrt(count)
    else:
        raise ValueError(f"fitness variant must be one of {FITNESS_VARIANTS}")
    score = numerator * denominator if sign_adjusted and numerator < 0 else numerator / denominator
    return float(np.clip(score, -1e9, 1e9)) if np.isfinite(score) else INVALID_FITNESS


def paper_sharpe(trade_returns: np.ndarray, risk_free_return: float = 0.025) -> float:
    """Paper Eqs.9/12: (sum net trade returns - Rf)/population trade STD.

    This is unannualized and is deliberately not called conventional Sharpe.
    Fewer than two trades or zero variance receive a finite dominated score.
    """
    values = np.asarray(trade_returns, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        return INVALID_FITNESS
    std = float(np.std(values, ddof=0))
    if std <= 1e-12:
        return INVALID_FITNESS
    score = (float(values.sum()) - risk_free_return) / std
    return float(np.clip(score, -1e9, 1e9)) if np.isfinite(score) else INVALID_FITNESS


@dataclass(frozen=True)
class ThresholdCalibration:
    osv_down: float | None
    osv_up: float | None
    tmv_down: float | None
    tmv_up: float | None
    duration_ratio: float
    event_ratio: float
    dc_events: int
    completed_os_events: int


@dataclass(frozen=True)
class _Features:
    dc: DCProfile
    osv: np.ndarray
    tmv: np.ndarray
    relative_move: np.ndarray
    duration_ratio: np.ndarray
    st2_ready: np.ndarray
    pattern_up: np.ndarray
    pattern_down: np.ndarray


def _profile_features(prices: np.ndarray, times: np.ndarray, threshold: float) -> _Features:
    dc = directional_changes(prices, threshold)
    valid = dc.direction != 0
    anchor = dc.anchor_price
    theoretical = anchor * (1.0 + dc.direction * threshold)
    osv = np.full(len(prices), np.nan)
    tmv = np.full(len(prices), np.nan)
    relative = np.full(len(prices), np.nan)
    np.divide(prices - theoretical, threshold * theoretical, out=osv, where=valid)
    np.divide(prices - anchor, threshold * anchor, out=tmv, where=valid)
    np.divide(prices - anchor, anchor, out=relative, where=valid)
    dc_duration = np.zeros(len(prices))
    elapsed = np.zeros(len(prices))
    known = np.flatnonzero(valid)
    dc_duration[known] = times[dc.confirmation_index[known]] - times[dc.anchor_index[known]]
    elapsed[known] = times[known] - times[dc.confirmation_index[known]]
    ratio = np.full(len(prices), np.nan)
    np.divide(elapsed, dc_duration, out=ratio, where=dc_duration > 0)

    # Prior trend OS existence is finalized only when the next reversal confirms.
    # The current trend needs a favorable extension already observed, never its
    # eventual retrospective extremum or a backdated OS-end label.
    up_pattern = np.zeros(len(prices), dtype=bool)
    down_pattern = np.zeros(len(prices), dtype=bool)
    history: list[tuple[int, bool]] = []
    last_confirmation = -1
    last_direction = 0
    for i in range(len(prices)):
        if dc.confirmation[i]:
            if last_confirmation >= 0:
                history.append((last_direction, bool(dc.anchor_index[i] > last_confirmation)))
                history = history[-4:]
            last_confirmation, last_direction = i, int(dc.direction[i])
        if dc.has_overshoot[i] and len(history) == 4:
            up_pattern[i] = dc.direction[i] == 1 and history == [(1, True), (-1, False), (1, True), (-1, False)]
            down_pattern[i] = dc.direction[i] == -1 and history == [(-1, True), (1, False), (-1, True), (1, False)]
    return _Features(dc, np.abs(osv), np.abs(tmv), np.abs(relative), ratio,
                     (dc_duration > 0) & (elapsed >= 2.0 * dc_duration), up_pattern, down_pattern)


def _fixed_uniform(times_ns: np.ndarray, seed: int, threshold_index: int) -> np.ndarray:
    """SplitMix64 keyed by timestamp: stable across prefixes and candidate weights."""
    mask = (1 << 64) - 1
    salt = np.uint64((int(seed) + (threshold_index + 1) * 0x9E3779B97F4A7C15) & mask)
    with np.errstate(over="ignore"):
        z = times_ns.astype(np.uint64) ^ salt
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(float) * (1.0 / (1 << 53))


def _threshold_signal(features: _Features, values: np.ndarray,
                      down: float | None, up: float | None) -> np.ndarray:
    signal = np.zeros(len(values), dtype=np.int8)
    if down is not None:
        signal[(features.dc.direction == -1) & (values >= down)] = 1
    if up is not None:
        signal[(features.dc.direction == 1) & (values >= up)] = -1
    return signal


class MSTGAM:
    """Optimize 70 strategy/threshold weights on training data only.

    ``predict`` replays causal DC state on the supplied context but never refits
    quartile cutoffs, RD/RN, or weights. Supply preceding observations when a
    held-out window should inherit its earlier DC state. The inference result
    is prefix-invariant for a fixed fitted model.
    """

    def __init__(self, population_size: int = 150, generations: int = 50,
                 seed: int = 0, min_votes: int = 3,
                 execution: ExecutionConfig | None = None,
                 crossover_probability: float = 0.95, risk_free_return: float = 0.025, *,
                 fitness: str = "paper", sign_adjusted: bool = False, mutation: str = "paper",
                 mutation_scale: float = 0.5):
        if fitness not in FITNESS_VARIANTS:
            raise ValueError(f"fitness must be one of {FITNESS_VARIANTS}")
        if mutation not in MUTATION_VARIANTS:
            raise ValueError(f"mutation must be one of {MUTATION_VARIANTS}")
        if not np.isfinite(mutation_scale) or mutation_scale <= 0:
            raise ValueError("mutation_scale must be positive")
        self.fitness, self.sign_adjusted, self.mutation = fitness, bool(sign_adjusted), mutation
        self.mutation_scale = float(mutation_scale)
        for name, value, lower in (("population_size", population_size, 2),
                                   ("generations", generations, 0), ("seed", seed, 0),
                                   ("min_votes", min_votes, 1)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if min_votes > len(GENES):
            raise ValueError("min_votes cannot exceed 70")
        if not np.isfinite(crossover_probability) or not 0 <= crossover_probability <= 1:
            raise ValueError("crossover_probability must be in [0,1]")
        if not np.isfinite(risk_free_return):
            raise ValueError("risk_free_return must be finite")
        self.population_size, self.generations = int(population_size), int(generations)
        self.seed, self.min_votes = int(seed), int(min_votes)
        self.crossover_probability = float(crossover_probability)
        self.risk_free_return = float(risk_free_return)
        self.execution = execution or ExecutionConfig(fee=0.0025)
        self._fitted = False
        self._diagnostics: dict = {}

    @property
    def diagnostics(self) -> dict:
        return dict(self._diagnostics)

    @staticmethod
    def _external_names(extra_features, n: int) -> list[str]:
        if extra_features is None:
            return []
        if not isinstance(extra_features, Mapping):
            raise ValueError("extra_features must be a mapping of names to aligned arrays")
        for name, values in extra_features.items():
            if np.asarray(values).shape != (n,):
                raise ValueError(f"External feature {name!r} is not aligned to the candles")
        return sorted(str(name) for name in extra_features)

    @staticmethod
    def _clock(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        times_ns = pd.DatetimeIndex(frame["available_at"]).as_unit("ns").asi8
        elapsed_seconds = (times_ns - times_ns[0]).astype(float) / 1e9
        return elapsed_seconds, times_ns

    def _score(self, frame: pd.DataFrame, signals: np.ndarray) -> tuple[float, int]:
        if self.execution.instrument is not None:
            return self._score_many(frame,np.asarray(signals)[None,:])[0]
        key = np.asarray(signals, dtype=np.int8).tobytes()
        if key not in self._fitness_cache:
            result = run_backtest(frame, signals, self.execution)
            values = np.asarray(result.trade_returns, dtype=float)
            self._fitness_cache[key] = (self._trade_score(len(values), float(values.sum()),
                                                          float(np.std(values, ddof=0)) if len(values) else 0.0),
                                        int(len(values)))
        else:
            self._cache_hits += 1
        return self._fitness_cache[key]

    def _score_many(self,frame,signals):
        if self.execution.instrument is None:
            return [self._score(frame,signal) for signal in signals]
        from ..market_execution import execute_batch
        keys=[np.asarray(signal,dtype=np.int8).tobytes() for signal in signals]
        missing={key:signal for key,signal in zip(keys,signals) if key not in self._fitness_cache}
        if missing:
            result=execute_batch(self._prepared_market,np.column_stack(list(missing.values())),self.execution)
            for i,key in enumerate(missing):
                count=int(result.trade_counts[i])
                self._fitness_cache[key]=(self._trade_score(count,float(result.trade_return_sum[i]),
                                                            float(result.trade_return_std[i])),count)
        self._cache_hits+=len(keys)-len(missing)
        return [self._fitness_cache[key] for key in keys]

    def _trade_score(self, count: int, total: float, population_std: float) -> float:
        return trade_fitness(count, total, population_std, risk_free_return=self.risk_free_return,
                             variant=self.fitness, sign_adjusted=self.sign_adjusted,
                             years=getattr(self, "_training_years", 1.0))

    def _mutate(self, child: np.ndarray, rng: np.random.Generator) -> None:
        gene = int(rng.integers(len(GENES)))
        if self.mutation == "paper":
            child[gene] = rng.uniform(0.0, 1.0)
            return
        # Heavy-tailed step size: P(alpha=k) proportional to k^-1.5 on 1..8,
        # the power-law strength of Doerr et al.'s fast GA, applied in log space.
        strengths = np.arange(1, 9, dtype=float)
        probabilities = strengths ** -1.5 / np.sum(strengths ** -1.5)
        alpha = float(rng.choice(strengths, p=probabilities))
        child[gene] = max(float(child[gene]), 1e-12) * float(np.exp(self.mutation_scale * alpha * rng.normal()))

    def generation_snapshots(self, every: int = 5) -> list[tuple[int, MSTGAM]]:
        """Frozen copies holding the best weights after selected generations."""
        import copy as _copy
        if not self._fitted:
            raise RuntimeError("Fit MSTGAM before requesting snapshots")
        snapshots = []
        for generation, weights in enumerate(self._generation_weights):
            if generation % every == 0 or generation == len(self._generation_weights) - 1:
                model = _copy.copy(self)
                model.weights_ = normalise_weights(weights)
                snapshots.append((generation, model))
        return snapshots

    def _cutoffs(self, frame: pd.DataFrame, f: _Features, values: np.ndarray) -> tuple[float | None, float | None]:
        candidates = []
        for direction in (-1, 1):
            observed = values[(f.dc.direction == direction) & f.dc.has_overshoot & np.isfinite(values)]
            if not len(observed):
                candidates.append([None])
            else:
                # Equal-count quartiles, then their medians, as in section4.1.2.
                candidates.append(sorted(set(float(np.median(part))
                                             for part in np.array_split(np.sort(observed), 4) if len(part))))
        best = (candidates[0][0], candidates[1][0])
        best_score = INVALID_FITNESS
        pairs=[(down,up) for down in candidates[0] for up in candidates[1]]
        signals=[_threshold_signal(f,values,down,up) for down,up in pairs]
        for pair,(score,_) in zip(pairs,self._score_many(frame,signals)):
            if score>best_score:
                best,best_score=pair,score
        return best

    def _calibrate(self, frame: pd.DataFrame, features: list[_Features], times: np.ndarray) -> list[ThresholdCalibration]:
        calibrations = []
        for f in features:
            confirmations = np.flatnonzero(f.dc.confirmation)
            dc_time = sum(float(times[i] - times[f.dc.anchor_index[i]]) for i in confirmations)
            os_times = [max(0.0, float(times[f.dc.anchor_index[current]] - times[previous]))
                        for previous, current in zip(confirmations[:-1], confirmations[1:])]
            osv_down, osv_up = self._cutoffs(frame, f, f.osv)
            tmv_down, tmv_up = self._cutoffs(frame, f, f.tmv)
            os_count = sum(duration > 0 for duration in os_times)
            calibrations.append(ThresholdCalibration(
                osv_down, osv_up, tmv_down, tmv_up,
                float(sum(os_times) / dc_time) if dc_time > 0 else 0.0,
                float(os_count / len(confirmations)) if len(confirmations) else 0.0,
                int(len(confirmations)), int(os_count)))
        return calibrations

    def _opinions(self, features: list[_Features], times_ns: np.ndarray) -> np.ndarray:
        per_threshold = []
        for index, (theta, f, c) in enumerate(zip(THRESHOLDS, features, self.calibration_)):
            direction = f.dc.direction
            opposite = -direction
            matrix = np.zeros((len(direction), 8), dtype=np.int8)
            matrix[:, 0] = np.where(f.relative_move >= 2 * theta, opposite, 0)
            matrix[:, 1] = np.where(f.st2_ready, opposite, 0)
            matrix[:, 2] = _threshold_signal(f, f.osv, c.osv_down, c.osv_up)
            matrix[:, 3] = _threshold_signal(f, f.tmv, c.tmv_down, c.tmv_up)
            matrix[:, 4] = np.where(f.duration_ratio >= c.duration_ratio, opposite, 0)
            matrix[(direction == -1) & f.dc.confirmation &
                   (_fixed_uniform(times_ns, self.seed, index) >= c.event_ratio), 5] = 1
            matrix[(direction == 1) & f.dc.confirmation, 5] = -1
            matrix[f.pattern_up, 6] = 1
            matrix[(direction == -1) & f.dc.confirmation, 6] = -1
            matrix[f.pattern_down, 7] = 1
            matrix[(direction == 1) & f.dc.confirmation, 7] = -1
            per_threshold.append(matrix)
        return np.column_stack([per_threshold[index][:, strategy - 1] for strategy, index in GENES])

    def fit(self, bars: pd.DataFrame, extra_features=None) -> MSTGAM:
        self._fitted = False
        self._diagnostics = {}
        frame = validate_bars(bars)
        if len(frame) < 3:
            raise ValueError("MSTGAM training needs at least three candles")
        ignored = self._external_names(extra_features, len(frame))
        self._fitness_cache: dict[bytes, tuple[float, int]] = {}
        self._cache_hits = 0
        if self.execution.instrument is not None:
            from ..market_execution import MarketArrays
            self._prepared_market=MarketArrays.from_bars(frame,self.execution.instrument)
        times, times_ns = self._clock(frame)
        self._training_years = max(float(times[-1] - times[0]) / (365.25 * 86400), 1e-9)
        prices = frame["close"].to_numpy()
        features = [_profile_features(prices, times, theta) for theta in THRESHOLDS]
        self.calibration_ = self._calibrate(frame, features, times)
        opinions = self._opinions(features, times_ns)
        rng = np.random.default_rng(self.seed)
        population = rng.uniform(0.0, 1.0, size=(self.population_size, len(GENES)))
        population /= population.sum(axis=1, keepdims=True)
        best_score, best_weights, best_trades = INVALID_FITNESS, None, 0
        history = []
        self._generation_weights = []
        buy_mask, sell_mask, hold_mask = (opinions == 1).T, (opinions == -1).T, (opinions == 0).T
        active = np.count_nonzero(opinions, axis=1) >= self.min_votes
        for generation in range(self.generations + 1):
            # Batch all votes, then reuse simulator results for identical actions.
            buy, sell, hold = population @ buy_mask, population @ sell_mask, population @ hold_mask
            all_signals = _vote_totals(buy, sell, hold, active)
            evaluations = self._score_many(frame,all_signals)
            scores = np.asarray([value[0] for value in evaluations])
            elite_index = int(np.argmax(scores))
            if scores[elite_index] > best_score:
                best_score = float(scores[elite_index])
                best_weights = population[elite_index].copy()
                best_trades = evaluations[elite_index][1]
            history.append(float(scores[elite_index]))
            self._generation_weights.append(population[elite_index].copy())
            if generation == self.generations:
                break

            def tournament() -> np.ndarray:
                indices = rng.choice(self.population_size, size=2, replace=False)
                return population[int(indices[np.argmax(scores[indices])])]

            children = [population[elite_index].copy()]
            while len(children) < self.population_size:
                first = tournament().copy()
                if rng.random() < self.crossover_probability:
                    second = tournament().copy()
                    left, right = sorted(rng.choice(np.arange(1, len(GENES)), size=2, replace=False))
                    first[left:right], second[left:right] = second[left:right].copy(), first[left:right].copy()
                    child = first if rng.integers(2) == 0 else second
                else:
                    child = first
                    self._mutate(child, rng)
                children.append(normalise_weights(child))
            population = np.asarray(children)
        self._diagnostics = {
            "training_rows": len(frame), "training_end": frame["available_at"].iloc[-1].isoformat(),
            "fitness": "paper_trade_sharpe_eq9_ddof0" if self.fitness == "paper" else "extension_trade_tstat_prorated_rf",
            "variant": {"fitness": self.fitness, "sign_adjusted": self.sign_adjusted, "mutation": self.mutation,
                        "training_years": self._training_years},
            "best_fitness": float(best_score),
            "best_closed_trades": best_trades, "generation_best": history,
            "unique_signal_evaluations": len(self._fitness_cache), "fitness_cache_hits": self._cache_hits,
            "ignored_external_features": ignored,
        }
        if best_weights is None:
            raise ValueError("MSTGAM found no valid candidate: need at least two closed trades with nonzero return variance")
        self.weights_ = normalise_weights(best_weights)
        self._fitted = True
        # Fitness cache is fit-local, not an unbounded process-level market cache.
        self._fitness_cache.clear()
        if hasattr(self,"_prepared_market"):
            del self._prepared_market
        return self

    def predict(self, bars: pd.DataFrame, extra_features=None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit MSTGAM before predicting")
        frame = validate_bars(bars)
        self._external_names(extra_features, len(frame))
        times, times_ns = self._clock(frame)
        prices = frame["close"].to_numpy()
        features = [_profile_features(prices, times, theta) for theta in THRESHOLDS]
        return aggregate_signals(self._opinions(features, times_ns), self.weights_, self.min_votes)

    def parameters(self) -> dict:
        if not self._fitted:
            raise RuntimeError("Fit MSTGAM before requesting parameters")
        return {
            "method": "mstgam", "paper": "10.1007/s10462-025-11419-z",
            "population_size": self.population_size, "generations": self.generations,
            "seed": self.seed, "min_votes": self.min_votes,
            "crossover_probability": self.crossover_probability,
            "risk_free_return": self.risk_free_return, "execution": asdict(self.execution),
            "fitness": self.fitness, "sign_adjusted": self.sign_adjusted, "mutation": self.mutation,
            "mutation_scale": self.mutation_scale,
            "weights": self.weights_.tolist(), "thresholds": list(THRESHOLDS),
            "genes": [{"strategy": strategy, "threshold_index": index} for strategy, index in GENES],
            "calibration": [asdict(item) for item in self.calibration_],
            "uses_external_features": False,
            "adaptations": ["causal_confirmed_dc", "unknown_initial_direction",
                            "calendar_availability_durations", "training_quartile_pair_grid",
                            "raw_opinions_shared_execution_guard", "hold_on_tie",
                            "timestamp_keyed_st6_draws", "simplex_normalization_after_variation",
                            "next_open_execution", "finite_invalid_fitness_penalty"],
        }
