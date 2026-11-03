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
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..backtest import ExecutionConfig, run_backtest
from .indicators import FeatureScaler, raw_features


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple[Node, ...] = ()
    value: str | float | None = None

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


def rank_fronts(population: list[_Individual]) -> list[list[int]]:
    """Fast vectorized dominance; invalid candidates can never beat valid ones."""
    values = np.asarray([(-p.objectives[0], -p.objectives[1], p.objectives[2]) for p in population])
    valid = np.asarray([p.eligible for p in population])
    dominance = ((values[:, None, :] <= values[None, :, :]).all(axis=2)
                 & (values[:, None, :] < values[None, :, :]).any(axis=2)
                 & valid[:, None] & valid[None, :])
    dominance |= valid[:, None] & ~valid[None, :]
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


class MOO3:
    def __init__(
        self, population_size=500, generations=50, seed=0, threshold=0.001,
        max_depth=6, preferences=(1 / 3, 1 / 3, 1 / 3),
        execution: ExecutionConfig | None = None,
    ):
        if type(population_size) is not int or population_size < 4:
            raise ValueError("population_size must be an integer >=4")
        if type(generations) is not int or generations < 0:
            raise ValueError("generations must be a nonnegative integer")
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

    def fit(self, bars: pd.DataFrame, extra_features: dict[str, np.ndarray] | None = None) -> MOO3:
        self._selected = None
        self._diagnostics = {}
        raw = raw_features(bars, self.threshold, extra_features)
        self._scaler = FeatureScaler.fit(raw)
        features = self._scaler.transform(raw)
        names = tuple(features)
        rng = np.random.default_rng(self.seed)
        cache = {}

        def individual(tree):
            if tree not in cache:
                signals = evaluate_tree(tree, features, len(bars)).astype(np.int8)
                result = run_backtest(bars, signals, self.execution)
                count = len(result.trade_returns)
                objectives = (result.paper_total_return, float(np.mean(result.trade_returns)) if count else 0.0,
                              float(np.std(result.trade_returns, ddof=0)) if count else 0.0)
                eligible = count >= 2 and bool(np.isfinite(objectives).all())
                # Finite placeholder plus eligibility-aware dominance avoids NaN/inf objectives.
                cache[tree] = (objectives if eligible else (-1e12, -1e12, 1e12), eligible, count)
            return _Individual(tree, *cache[tree])

        coverage = Node("gt", (Node("constant", value=1.0), Node("constant", value=0.0)))
        population = [individual(coverage)] + [individual(self._tree(rng, names, self.max_depth)) for _ in range(self.population_size - 1)]
        history = []
        for generation in range(self.generations + 1):
            fronts = rank_fronts(population)
            history.append({"generation": generation, "eligible": sum(p.eligible for p in population),
                            "pareto_size": sum(population[i].eligible for i in fronts[0])})
            if generation == self.generations:
                break

            def tournament():
                left, right = (population[int(i)] for i in rng.integers(len(population), size=2))
                return min((left, right), key=lambda p: (p.rank, -p.crowding))

            children = [individual(self._offspring(tournament().tree, tournament().tree, rng, names))
                        for _ in range(self.population_size)]
            combined = population + children
            survival = []
            for front in rank_fronts(combined):
                ordered = sorted(front, key=lambda i: combined[i].crowding, reverse=True)
                survival.extend(ordered[:self.population_size - len(survival)])
                if len(survival) == self.population_size:
                    break
            population = [combined[i] for i in survival]
        final_front = [population[i] for i in rank_fronts(population)[0] if population[i].eligible]
        # Remove duplicate genotypes from the reported front without changing evolution.
        final_front = list({p.tree: p for p in final_front}.values())
        if not final_front:
            raise ValueError("MOO3 found no eligible strategy with at least two completed trades")
        objectives = np.asarray([p.objectives for p in final_front])
        scores, normalization = modified_sharpe(objectives, self.preferences)
        winner = int(np.argmax(scores))
        self._selected = final_front[winner].tree
        self._diagnostics = {
            "method": "moo3", "source_doi": "10.1007/s10462-025-11390-9",
            "unique_evaluations": len(cache), "minimum_completed_trades": 2,
            "history": history, "selected_index": winner, "training_front_normalization": normalization,
            "pareto_front": [{"rule": p.tree.to_dict(), "paper_total_return_pct": p.objectives[0],
                              "mean_trade_return": p.objectives[1], "trade_return_std": p.objectives[2],
                              "completed_trades": p.trade_count, "modified_sharpe": float(scores[i])}
                             for i, p in enumerate(final_front)],
            "feature_count": len(names), "training_feature_bounds": {k: list(v) for k, v in self._scaler.bounds.items()},
        }
        return self

    def predict(self, bars: pd.DataFrame, extra_features: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if self._selected is None:
            raise RuntimeError("MOO3 must be fitted before prediction")
        features = self._scaler.transform(raw_features(bars, self.threshold, extra_features))
        return evaluate_tree(self._selected, features, len(bars)).astype(np.int8)

    def parameters(self) -> dict:
        return {"method": "moo3", "population_size": self.population_size, "generations": self.generations,
                "seed": self.seed, "threshold": self.threshold, "max_depth": self.max_depth,
                "preferences": list(self.preferences), "execution": asdict(self.execution),
                "entry_rule": self._selected.to_dict() if self._selected is not None else None}

    @property
    def diagnostics(self) -> dict:
        return copy.deepcopy(self._diagnostics)
