"""MOO3: directional-change typed GP, NSGA-II, final-front modified Sharpe.

Research implementation of Long et al., DOI 10.1007/s10462-025-11390-9,
Sections 4--5. GP evolves only the buy/hold condition; the shared simulator owns
long-only exits. Objectives are paper Eq.3 total return, mean completed-trade
return and completed-trade volatility. The first is NOT equity compounding.

Documented adaptations: train-only scaling, causal confirmed DC terminals,
standard NSGA-II infinite endpoint crowding, minimum two completed trades,
population-variance convention, explicit end liquidation/next-open execution,
and one deterministic always-buy coverage individual. Paper population/depth/
generation/operator defaults are retained. Small runs validate software only.

Labeled EXTENSIONS (all off by default; see docs/EXTENSIONS.md):
``objectives="robust"`` replaces Eq.3 with compounded trade return (scale and
split invariant) and mean trade return with its lower confidence bound;
``min_trades_per_year`` adds a constrained-domination activity constraint
(Deb 2000); ``survival="nsga2_sequential"|"nsga3"`` changes last-front
truncation; ``feature_transform="scale_free"`` removes price levels and shares
scaling within feature families; ``threshold_volatility_multiple`` fits the DC
threshold to training volatility.

Review 2026-11-05: ``objectives="single"`` is a scalar baseline, the lower
confidence bound of annualized log growth, (sum log(1+q) - z*sqrt(n)*s)/years,
with s the sample trade-return std (close to the log-return std for small
trades); ``objectives="blend_excess"`` scores the information ratio of daily
excess over an exposure-matched blend on a fresh random half of the training
months each generation, minus ``complexity_penalty`` per tree node. Scalar
objectives fill all three slots, so Pareto rank equals score order, survival
is plain NSGA-II truncation and the final pick is the best score.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import pandas as pd

from ..backtest import ExecutionConfig, run_backtest
from .indicators import FEATURE_TRANSFORMS, FeatureScaler, feature_families, raw_features

OBJECTIVE_SETS = ("paper", "robust", "single", "blend_excess")
SCALAR_OBJECTIVES = ("single", "blend_excess")
SURVIVAL_RULES = ("nsga2", "nsga2_sequential", "nsga3")


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple[Node, ...] = ()
    value: str | float | None = None

    def __post_init__(self):
        if self.op == "feature":
            if self.children or not isinstance(self.value, str) or not self.value:
                raise ValueError("Feature terminal needs a name and no children")
        elif self.op == "constant":
            if (self.children or isinstance(self.value, bool) or not isinstance(self.value, Real)
                    or not np.isfinite(self.value) or not 0 <= self.value <= 1):
                raise ValueError("Constant terminal must be finite and in [0,1]")
        elif self.op in ("and", "or", "gt", "lt"):
            expected = "boolean" if self.op in ("and", "or") else "number"
            if (self.value is not None or len(self.children) != 2
                    or any(not isinstance(child, Node) or child.kind != expected for child in self.children)):
                raise ValueError(f"{self.op} requires two {expected} children")
        else:
            raise ValueError(f"Unknown tree operator {self.op!r}")

    @property
    def kind(self) -> str:
        return "number" if self.op in ("feature", "constant") else "boolean"

    @property
    def depth(self) -> int:
        return 0 if not self.children else 1 + max(child.depth for child in self.children)

    def to_dict(self) -> dict:
        if self.op in ("feature", "constant"):
            return {self.op: self.value}
        return {"operator": self.op, "children": [child.to_dict() for child in self.children]}


def evaluate_tree(node: Node, features: dict[str, np.ndarray], n: int) -> np.ndarray:
    if node.op == "feature":
        return features[str(node.value)]
    if node.op == "constant":
        return np.full(n, float(node.value))
    left, right = (evaluate_tree(child, features, n) for child in node.children)
    if node.op == "and":
        return np.logical_and(left, right)
    if node.op == "or":
        return np.logical_or(left, right)
    valid = np.isfinite(left) & np.isfinite(right)
    if node.op == "gt":
        return valid & (left > right)
    if node.op == "lt":
        return valid & (left < right)
    raise ValueError(f"Unknown tree operator {node.op!r}")


def node_count(node: Node) -> int:
    return 1 + sum(node_count(child) for child in node.children)


def _paths(node: Node, path=()):
    yield path, node
    for i, child in enumerate(node.children):
        yield from _paths(child, (*path, i))


def _replace(node: Node, path: tuple[int, ...], replacement: Node) -> Node:
    if not path:
        return replacement
    children = list(node.children)
    children[path[0]] = _replace(children[path[0]], path[1:], replacement)
    return Node(node.op, tuple(children), node.value)


@dataclass
class _Individual:
    tree: Node
    objectives: tuple[float, float, float]
    eligible: bool
    trade_count: int
    rank: int = 0
    crowding: float = 0.0
    violation: float = 0.0


def _minimization(population) -> np.ndarray:
    return np.asarray([(-p.objectives[0], -p.objectives[1], p.objectives[2]) for p in population], dtype=float)


def _crowding(values: np.ndarray) -> np.ndarray:
    """Standard NSGA-II crowding of one front given minimization values."""
    distance = np.zeros(len(values))
    if len(values) <= 2:
        distance[:] = np.inf
        return distance
    for metric in range(values.shape[1]):
        ordered = np.argsort(values[:, metric], kind="stable")
        distance[ordered[0]] = distance[ordered[-1]] = np.inf
        span = values[ordered[-1], metric] - values[ordered[0], metric]
        if span > 0:
            distance[ordered[1:-1]] += (values[ordered[2:], metric] - values[ordered[:-2], metric]) / span
    return distance


def rank_fronts(population: list[_Individual], *, constrained: bool = False) -> list[list[int]]:
    """Fast vectorized dominance; invalid candidates can never beat valid ones.

    ``constrained=True`` is Deb's constrained domination: among infeasible
    candidates, the smaller constraint violation dominates (paper mode leaves
    infeasible candidates mutually non-dominated).
    """
    values = _minimization(population)
    valid = np.asarray([p.eligible for p in population])
    dominance = ((values[:, None, :] <= values[None, :, :]).all(axis=2)
                 & (values[:, None, :] < values[None, :, :]).any(axis=2)
                 & valid[:, None] & valid[None, :])
    dominance |= valid[:, None] & ~valid[None, :]
    if constrained:
        violation = np.asarray([p.violation for p in population], dtype=float)
        dominance |= (~valid[:, None] & ~valid[None, :]) & (violation[:, None] < violation[None, :])
    remaining = dominance.sum(axis=0)
    fronts = []
    current = np.flatnonzero(remaining == 0)
    while current.size:
        front = current.tolist()
        fronts.append(front)
        for index in front:
            population[index].rank = len(fronts) - 1
            population[index].crowding = 0.0
        for metric in range(3):
            ordered = sorted(front, key=lambda i: values[i, metric])
            population[ordered[0]].crowding = float("inf")
            population[ordered[-1]].crowding = float("inf")
            span = values[ordered[-1], metric] - values[ordered[0], metric]
            if span > 0:
                for position in range(1, len(ordered) - 1):
                    population[ordered[position]].crowding += float(
                        (values[ordered[position + 1], metric] - values[ordered[position - 1], metric]) / span
                    )
        remaining[current] = -1
        remaining -= dominance[current].sum(axis=0)
        current = np.flatnonzero(remaining == 0)
    return fronts


def modified_sharpe(
    objectives: np.ndarray, preferences=(1 / 3, 1 / 3, 1 / 3),
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Paper Eq.6 for ONE training Pareto front, not conventional Sharpe."""
    values = np.asarray(objectives, dtype=float)
    weights = np.asarray(preferences, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite (n,3) objective matrix")
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Preferences must be three nonnegative finite weights summing to one")
    lo, hi = values.min(axis=0), values.max(axis=0)
    span = hi - lo
    normalized = np.divide(values - lo, span, out=np.zeros_like(values), where=span > 0)
    scores = np.exp(np.log1p(normalized) @ (weights * np.array([1, 1, -1])))
    return scores, {"minimum": lo.tolist(), "maximum": hi.tolist()}


def sequential_truncation(values: np.ndarray, keep: int) -> list[int]:
    """EXTENSION: remove the least-crowded member one at a time, recomputing
    crowding after every removal (Kukkonen and Deb 2006). Zheng and Doerr
    (IEEE TEVC 2025) prove front gaps then stay within a constant factor of the
    minimum, unlike the one-shot crowding computation of classic NSGA-II."""
    remaining = list(range(len(values)))
    while len(remaining) > keep:
        distance = _crowding(values[remaining])
        # Stable tie-break: the last-listed member with the smallest distance.
        worst = int(np.flatnonzero(distance == distance.min())[-1])
        remaining.pop(worst)
    return remaining


def das_dennis(objectives: int, divisions: int) -> np.ndarray:
    """Uniform simplex reference points (Das and Dennis 1998) for NSGA-III."""
    points = []
    def build(prefix, left, depth):
        if depth == objectives - 1:
            points.append(prefix + [left])
            return
        for value in range(left + 1):
            build(prefix + [value], left - value, depth + 1)
    build([], divisions, 0)
    return np.asarray(points, dtype=float) / divisions


def reference_divisions(population_size: int, objectives: int = 3) -> int:
    divisions = 1
    while math.comb(divisions + objectives - 1, objectives - 1) < population_size:
        divisions += 1
    return divisions


def nsga3_truncation(selected: np.ndarray, last: np.ndarray, keep: int,
                     references: np.ndarray, rng: np.random.Generator) -> list[int]:
    """EXTENSION: NSGA-III niche-preserving choice of ``keep`` rows of ``last``.

    ``selected`` holds minimization values already admitted from better fronts.
    Normalization uses the ideal point and per-objective extreme (maximum)
    values of the combined set, a simple robust variant of Deb and Jain (2014).
    Wietheger and Doerr (IJCAI 2023) prove NSGA-III never loses a found
    Pareto-optimal value on 3-objective OneMinMax, where NSGA-II provably can.
    """
    combined = np.vstack([selected, last]) if len(selected) else last
    ideal = combined.min(axis=0)
    span = combined.max(axis=0) - ideal
    span[span <= 0] = 1.0
    normalized = (combined - ideal) / span
    unit = references / np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1e-12)
    projection = normalized @ unit.T
    distance = np.sqrt(np.maximum((normalized ** 2).sum(axis=1, keepdims=True) - projection ** 2, 0.0))
    niche = distance.argmin(axis=1)
    niche_distance = distance[np.arange(len(combined)), niche]
    count = np.bincount(niche[:len(selected)], minlength=len(references)).astype(float)
    candidates = list(range(len(selected), len(combined)))
    chosen: list[int] = []
    excluded = np.zeros(len(references), dtype=bool)
    while len(chosen) < keep:
        open_counts = np.where(excluded, np.inf, count)
        minimum = open_counts.min()
        options = np.flatnonzero(open_counts == minimum)
        reference = int(rng.choice(options))
        members = [i for i in candidates if niche[i] == reference]
        if not members:
            excluded[reference] = True
            continue
        if count[reference] == 0:
            pick = min(members, key=lambda i: niche_distance[i])
        else:
            pick = members[int(rng.integers(len(members)))]
        candidates.remove(pick)
        chosen.append(pick - len(selected))
        count[reference] += 1
    return chosen


class MOO3:
    def __init__(
        self, population_size=500, generations=50, seed=0, threshold=0.001,
        max_depth=6, preferences=(1 / 3, 1 / 3, 1 / 3),
        execution: ExecutionConfig | None = None, *,
        objectives: str = "paper", min_trades_per_year: float | None = None,
        lcb_z: float = 1.0, survival: str = "nsga2", feature_transform: str = "paper",
        threshold_volatility_multiple: float | None = None, complexity_penalty: float = 0.0,
        subwindow_fraction: float = 0.5,
    ):
        if objectives not in OBJECTIVE_SETS:
            raise ValueError(f"objectives must be one of {OBJECTIVE_SETS}")
        if survival not in SURVIVAL_RULES:
            raise ValueError(f"survival must be one of {SURVIVAL_RULES}")
        if feature_transform not in FEATURE_TRANSFORMS:
            raise ValueError(f"feature_transform must be one of {FEATURE_TRANSFORMS}")
        if min_trades_per_year is not None and (not np.isfinite(min_trades_per_year) or min_trades_per_year < 0):
            raise ValueError("min_trades_per_year must be finite and nonnegative or None")
        if not np.isfinite(lcb_z) or lcb_z < 0:
            raise ValueError("lcb_z must be finite and nonnegative")
        if threshold_volatility_multiple is not None and (
                not np.isfinite(threshold_volatility_multiple) or threshold_volatility_multiple <= 0):
            raise ValueError("threshold_volatility_multiple must be positive or None")
        self.objectives, self.survival, self.feature_transform = objectives, survival, feature_transform
        self.min_trades_per_year = None if min_trades_per_year is None else float(min_trades_per_year)
        self.lcb_z = float(lcb_z)
        if not np.isfinite(complexity_penalty) or complexity_penalty < 0:
            raise ValueError("complexity_penalty must be finite and nonnegative")
        if not np.isfinite(subwindow_fraction) or not 0 < subwindow_fraction <= 1:
            raise ValueError("subwindow_fraction must be in (0,1]")
        self.complexity_penalty, self.subwindow_fraction = float(complexity_penalty), float(subwindow_fraction)
        self.threshold_volatility_multiple = (None if threshold_volatility_multiple is None
                                              else float(threshold_volatility_multiple))
        if type(population_size) is not int or population_size < 4:
            raise ValueError("population_size must be an integer >=4")
        if type(generations) is not int or generations < 0:
            raise ValueError("generations must be a nonnegative integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(max_depth) is not int or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must be in (0,1)")
        modified_sharpe(np.zeros((1, 3)), preferences)
        self.population_size = population_size
        self.generations = generations
        self.seed = seed
        self.threshold = float(threshold)
        self.max_depth = max_depth
        self.preferences = tuple(float(x) for x in preferences)
        self.execution = execution or ExecutionConfig(fee=0.00025, max_holding_days=5, take_profit=0.05)
        self._selected: Node | None = None
        self._diagnostics: dict = {}

    def _number(self, rng: np.random.Generator, names: tuple[str, ...]) -> Node:
        return Node("constant", value=float(rng.random())) if rng.random() < 0.25 else Node("feature", value=str(rng.choice(names)))

    def _tree(self, rng: np.random.Generator, names: tuple[str, ...], remaining: int) -> Node:
        if remaining == 1 or rng.random() < 0.45:
            return Node(str(rng.choice(("gt", "lt"))), (self._number(rng, names), self._number(rng, names)))
        return Node(str(rng.choice(("and", "or"))), (self._tree(rng, names, remaining - 1), self._tree(rng, names, remaining - 1)))

    def _offspring(self, first: Node, second: Node, rng, names) -> Node:
        locations = list(_paths(first))
        path, chosen = locations[int(rng.integers(len(locations)))]
        if rng.random() < 0.95:
            compatible = [(p, node) for p, node in _paths(second) if node.kind == chosen.kind]
            replacement = compatible[int(rng.integers(len(compatible)))][1]
        elif chosen.kind == "number":
            replacement = self._number(rng, names)
        else:
            # Point mutation preserves arity and strong typing.
            alternative = {"gt": "lt", "lt": "gt", "and": "or", "or": "and"}[chosen.op]
            replacement = Node(alternative, chosen.children)
        changed = _replace(first, path, replacement)
        return changed if changed.depth <= self.max_depth else first

    def _training_threshold(self, bars: pd.DataFrame) -> float:
        if self.threshold_volatility_multiple is None:
            return self.threshold
        close = pd.to_numeric(bars["close"], errors="raise").to_numpy(dtype=float)
        sigma = float(np.std(np.diff(np.log(close)), ddof=1)) if len(close) > 2 else 0.0
        return float(np.clip(self.threshold_volatility_multiple * sigma, 1e-4, 0.5))

    def _objectives(self, count: int, total_paper: float, mean: float, std: float,
                    log_growth: float) -> tuple[float, float, float]:
        if self.objectives == "paper":
            return (total_paper, mean, std)
        if self.objectives == "single":
            sample_std = std * math.sqrt(count / (count - 1)) if count > 1 else 0.0
            score = (log_growth - self.lcb_z * math.sqrt(count) * sample_std) / self._training_years
            return (score, score, -score)
        # Scale-free compounded trade return (%), lower confidence bound of the
        # mean trade return using the sample standard error, and trade std.
        sample_std = std * math.sqrt(count / (count - 1)) if count > 1 else 0.0
        lower = mean - self.lcb_z * sample_std / math.sqrt(count) if count else 0.0
        return (100.0 * math.expm1(log_growth), lower, std)

    def fit(self, bars: pd.DataFrame, extra_features: dict[str, np.ndarray] | None = None) -> MOO3:
        self._selected = None
        self._diagnostics = {}
        self.fitted_threshold_ = self._training_threshold(bars)
        raw = raw_features(bars, self.fitted_threshold_, extra_features, transform=self.feature_transform)
        groups = feature_families(raw) if self.feature_transform == "scale_free" else None
        self._scaler = FeatureScaler.fit(raw, groups)
        features = self._scaler.transform(raw)
        names = tuple(features)
        rng = np.random.default_rng(self.seed)
        cache = {}
        times = pd.to_datetime(bars["timestamp"], utc=True)
        known = pd.to_datetime(bars["available_at"], utc=True)
        years = max((known.iloc[-1] - times.iloc[0]).total_seconds() / (365.25 * 86400), 0.0)
        minimum = 2 if self.min_trades_per_year is None else max(2, math.ceil(self.min_trades_per_year * years))
        constrained = self.min_trades_per_year is not None
        self._training_years = max(years, 1e-9)
        scalar = self.objectives in SCALAR_OBJECTIVES
        survival_rule = "nsga2" if scalar else self.survival

        def record(tree, count, objectives):
            finite = bool(np.isfinite(objectives).all())
            eligible = count >= minimum and finite
            violation = 0.0 if eligible else float(max(minimum - count, 0) + (0 if finite else minimum + 1))
            cache[tree] = (objectives if eligible else (-1e12, -1e12, 1e12), eligible, count, violation)

        def evaluate_many(trees):
            if self.execution.instrument is None:
                return [individual(tree) for tree in trees]
            from ..market_execution import MarketArrays,execute_batch
            nonlocal prepared_market
            if prepared_market is None:
                prepared_market=MarketArrays.from_bars(bars,self.execution.instrument)
            missing=list(dict.fromkeys(tree for tree in trees if tree not in cache))
            if missing:
                decisions=np.column_stack([evaluate_tree(tree,features,len(bars)).astype(np.int8) for tree in missing])
                outcomes=execute_batch(prepared_market,decisions,self.execution)
                for i,tree in enumerate(missing):
                    count=int(outcomes.trade_counts[i])
                    objectives=self._objectives(count,float(outcomes.paper_total_return[i]),
                        float(outcomes.trade_return_sum[i]/count) if count else 0.,float(outcomes.trade_return_std[i]),
                        float(outcomes.trade_log_growth[i]))
                    record(tree,count,objectives)
            return [self._individual(tree,cache) for tree in trees]

        prepared_market=None

        def individual(tree):
            if tree not in cache:
                signals = evaluate_tree(tree, features, len(bars)).astype(np.int8)
                result = run_backtest(bars, signals, self.execution)
                count = len(result.trade_returns)
                returns = np.asarray(result.trade_returns, dtype=float)
                objectives = self._objectives(
                    count, result.paper_total_return, float(np.mean(returns)) if count else 0.0,
                    float(np.std(returns, ddof=0)) if count else 0.0,
                    float(np.log1p(np.maximum(returns, -1 + 1e-12)).sum()))
                # Finite placeholder plus eligibility-aware dominance avoids NaN/inf objectives.
                record(tree, count, objectives)
            return self._individual(tree, cache)

        excess_state = {}
        if self.objectives == "blend_excess":
            from .robust_fitness import SubWindows, bar_months, benchmark_returns, periods_per_year, period_returns
            excess_state = {"paths": {}, "mask": None, "ppy": periods_per_year(bars),
                            "benchmark": benchmark_returns(bars, self.execution),
                            "sampler": SubWindows(bar_months(bars), self.seed, self.subwindow_fraction)}

            def excess_many(trees):
                """Cache full-window paths once; re-score every requested tree on the current mask."""
                from .robust_fitness import excess_information_ratio
                nonlocal prepared_market
                paths = excess_state["paths"]
                missing = list(dict.fromkeys(tree for tree in trees if tree not in paths))
                if missing:
                    decisions = np.column_stack([evaluate_tree(tree, features, len(bars)).astype(np.int8) for tree in missing])
                    if self.execution.instrument is not None:
                        from ..market_execution import MarketArrays, execute_batch
                        if prepared_market is None:
                            prepared_market = MarketArrays.from_bars(bars, self.execution.instrument)
                        outcome = execute_batch(prepared_market, decisions, self.execution, record_equity=True,
                                                record_exposure=True)
                        returns, held, counts = period_returns(outcome.equity), outcome.held, outcome.trade_counts
                    else:
                        from ..statistics import exposure_path
                        results = [run_backtest(bars, decisions[:, i], self.execution) for i in range(len(missing))]
                        returns = np.column_stack([period_returns(r.equity) for r in results])
                        held = np.column_stack([exposure_path(r) for r in results])
                        counts = [len(r.trades) for r in results]
                    for i, tree in enumerate(missing):
                        paths[tree] = (returns[:, i].astype(np.float32), held[:, i].astype(np.float32), int(counts[i]))
                unique = list(dict.fromkeys(trees))
                scores = excess_information_ratio(np.column_stack([paths[tree][0] for tree in unique]),
                                                  np.column_stack([paths[tree][1] for tree in unique]),
                                                  excess_state["benchmark"], excess_state["mask"],
                                                  periods_per_year=excess_state["ppy"])
                for tree, score in zip(unique, scores):
                    value = float(score) - self.complexity_penalty * node_count(tree)
                    record(tree, paths[tree][2], (value, value, -value))
                return [self._individual(tree, cache) for tree in trees]

            evaluate_many = excess_many
        references = das_dennis(3, reference_divisions(self.population_size)) if survival_rule == "nsga3" else None
        coverage = Node("gt", (Node("constant", value=1.0), Node("constant", value=0.0)))
        population = evaluate_many([coverage]+[self._tree(rng,names,self.max_depth) for _ in range(self.population_size-1)])
        history = []
        for generation in range(self.generations + 1):
            if excess_state:
                # A fresh training-month sample; parents are re-scored on it too.
                excess_state["mask"] = excess_state["sampler"].mask(generation)
                population = evaluate_many([p.tree for p in population])
            fronts = rank_fronts(population, constrained=constrained)
            history.append({"generation": generation, "eligible": sum(p.eligible for p in population),
                            "pareto_size": sum(population[i].eligible for i in fronts[0])})
            if generation == self.generations:
                break

            def tournament():
                left, right = (population[int(i)] for i in rng.integers(len(population), size=2))
                return min((left, right), key=lambda p: (p.rank, -p.crowding))

            children = evaluate_many([self._offspring(tournament().tree,tournament().tree,rng,names)
                                      for _ in range(self.population_size)])
            combined = population + children
            survival = []
            for front in rank_fronts(combined, constrained=constrained):
                room = self.population_size - len(survival)
                if survival_rule == "nsga2":
                    # Paper path, unchanged: every admitted front in crowding order
                    # (population order feeds the index-based tournaments).
                    ordered = sorted(front, key=lambda i: combined[i].crowding, reverse=True)
                    survival.extend(ordered[:room])
                elif len(front) <= room:
                    survival.extend(sorted(front, key=lambda i: combined[i].crowding, reverse=True))
                elif survival_rule == "nsga2_sequential":
                    kept = sequential_truncation(_minimization([combined[i] for i in front]), room)
                    survival.extend(front[k] for k in kept)
                else:
                    chosen = nsga3_truncation(_minimization([combined[i] for i in survival]),
                                              _minimization([combined[i] for i in front]), room, references, rng)
                    survival.extend(front[k] for k in chosen)
                if len(survival) == self.population_size:
                    break
            population = [combined[i] for i in survival]
        if excess_state:
            excess_state["mask"] = None  # final choice on the full training window
            population = evaluate_many([p.tree for p in population])
        final_front = [population[i] for i in rank_fronts(population, constrained=constrained)[0] if population[i].eligible]
        # Remove duplicate genotypes from the reported front without changing evolution.
        final_front = list({p.tree: p for p in final_front}.values())
        if not final_front:
            raise ValueError("MOO3 found no eligible strategy with at least "
                             + ("two" if minimum == 2 else str(minimum)) + " completed trades")
        objectives = np.asarray([p.objectives for p in final_front])
        if scalar:
            scores, normalization = objectives[:, 0], None
        else:
            scores, normalization = modified_sharpe(objectives, self.preferences)
        winner = int(np.argmax(scores))
        self._selected = final_front[winner].tree
        names_of_objectives = {"paper":("paper_total_return_pct", "mean_trade_return", "trade_return_std"),
                               "robust":("compounded_trade_return_pct", "mean_trade_return_lower_bound", "trade_return_std"),
                               "single":("lcb_annual_log_growth", "lcb_annual_log_growth_copy", "negative_score"),
                               "blend_excess":("penalized_excess_information_ratio", "score_copy", "negative_score"),
                               }[self.objectives]
        haircut = None
        if excess_state:
            from .robust_fitness import excess_paths, selection_haircut
            pool = [tree for tree, value in excess_state["paths"].items() if value[2] >= minimum]
            if self._selected in pool:
                paths = excess_paths(np.column_stack([excess_state["paths"][tree][0] for tree in pool]),
                                     np.column_stack([excess_state["paths"][tree][1] for tree in pool]),
                                     excess_state["benchmark"])
                haircut = selection_haircut(paths, pool.index(self._selected), periods_per_year=excess_state["ppy"],
                                            seed=self.seed)
                haircut["scope"] = "every distinct evaluated tree meeting the trade floor"
        self._diagnostics = {
            "method": "moo3", "source_doi": "10.1007/s10462-025-11390-9",
            "unique_evaluations": len(cache), "minimum_completed_trades": minimum,
            "variant": {"objectives": self.objectives, "survival": self.survival,
                        "feature_transform": self.feature_transform,
                        "constrained_domination": constrained, "fitted_threshold": self.fitted_threshold_,
                        **({"scalar_objective": True, "effective_survival": survival_rule,
                            "complexity_penalty_per_node": self.complexity_penalty} if scalar else {}),
                        **({"subwindows": excess_state["sampler"].describe(), "selection_haircut": haircut}
                           if excess_state else {})},
            "objective_names": list(names_of_objectives),
            "history": history, "selected_index": winner, "training_front_normalization": normalization,
            "pareto_front": [{"rule": p.tree.to_dict(), **dict(zip(names_of_objectives, p.objectives)),
                              "completed_trades": p.trade_count,
                              ("selection_score" if scalar else "modified_sharpe"): float(scores[i])}
                             for i, p in enumerate(final_front)],
            "feature_count": len(names), "training_feature_bounds": {k: list(v) for k, v in self._scaler.bounds.items()},
        }
        return self

    @staticmethod
    def _individual(tree, cache) -> _Individual:
        objectives, eligible, count, violation = cache[tree]
        return _Individual(tree, objectives, eligible, count, violation=violation)

    def predict(self, bars: pd.DataFrame, extra_features: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if self._selected is None:
            raise RuntimeError("MOO3 must be fitted before prediction")
        raw = raw_features(bars, self.fitted_threshold_, extra_features, transform=self.feature_transform)
        features = self._scaler.transform(raw)
        return evaluate_tree(self._selected, features, len(bars)).astype(np.int8)

    def parameters(self) -> dict:
        return {"method": "moo3", "population_size": self.population_size, "generations": self.generations,
                "seed": self.seed, "threshold": self.threshold, "max_depth": self.max_depth,
                "preferences": list(self.preferences), "execution": asdict(self.execution),
                "objectives": self.objectives, "min_trades_per_year": self.min_trades_per_year,
                "lcb_z": self.lcb_z, "survival": self.survival, "feature_transform": self.feature_transform,
                "complexity_penalty": self.complexity_penalty, "subwindow_fraction": self.subwindow_fraction,
                "threshold_volatility_multiple": self.threshold_volatility_multiple,
                "fitted_threshold": getattr(self, "fitted_threshold_", None),
                "entry_rule": self._selected.to_dict() if self._selected is not None else None}

    @property
    def diagnostics(self) -> dict:
        return copy.deepcopy(self._diagnostics)
