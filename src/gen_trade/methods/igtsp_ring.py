"""IGTSP-Ring from the supplied full text, DOI 10.1080/0305215X.2025.2592030.

Implements §3.3.2--5, Eqs.6--16, Algorithms 1/4 and Table 5 defaults.
The paper omits exact indicator rules, bank ranking, weight-bit encoding and
several execution details. Those choices are explicit in docs/IGTSP_RING.md.
Section 3.1's alternative objective is NOT silently mixed into Eq.6.

Labeled EXTENSIONS (off by default; docs/EXTENSIONS.md): mixture-drawdown
fitness, bitwise mutation on every child, duplicate-free children, a
Falkenauer-style group-injection crossover, a correlation-diverse facility-
location bank, two-stage bank ranking under the evolved TP/SL, and an adaptive
migration interval (Mambrini and Sudholt 2015, scheme B).

Review 2026-11-05 extensions: ``fitness="mixture_mtm"`` measures the mixture
on marked-to-market daily sleeve equity instead of Eq.9 monthly trade
averages; ``fitness="blend_excess"`` scores the mixture's excess over an
exposure-matched blend on resampled training months; ``weight_mode`` chooses
evolved ("ga"), equal ("equal") or LP-optimal ("lp", Dinkelbach) group weights.
Results 2026-11-05: ``decision_cadence`` acts weekly or monthly on every bank
strategy (methods/cadence.py), and the blend-excess selection haircut covers
every chromosome evaluated in the final stage.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from ..backtest import ExecutionConfig, PortfolioSignals
from ..data import validate_bars
from ..parallel import process_pool
from .cadence import DECISION_CADENCES, apply_cadence
from .ring_bank import DEFAULT_BANK, INDICATORS, Strategy, strategy_signals
from .ring_fitness import (FITNESS_VARIANTS, INVALID_FITNESS, MonthlyProfiles, PreparedMarket,
                           drawdown_ratio_weights, monthly_trade_profiles, portfolio_excess_fitness, portfolio_fitness)

SOURCE_DOI = "10.1080/0305215X.2025.2592030"
SOURCE_PDF_SHA256 = "7ccf3ab38a5609ffc5b89e72521587ecb6d9b7b42bdf18b2fa299765b8b31d42"
FIDELITY = "full_text_research_implementation"
PAPER_SETTINGS = {"population_size":150, "generations":100, "island_count":10, "groups":17,
                  "migration_rate":.5, "migration_interval":10, "crossover_probability":.6,
                  "mutation_probability":.01, "selection_rate":.5, "inversion_probability":.1,
                  "take_profit_bits":8, "stop_loss_bits":8}
CROSSOVER_MODES = ("paper", "group")
MUTATION_MODES = ("paper", "bitwise")
BANK_SELECTIONS = ("paper", "diverse")
BANK_RANKINGS = ("paper", "two_stage")
MIGRATION_SCHEDULES = ("paper", "adaptive")
WEIGHT_MODES = ("ga", "equal", "lp")


def uniform_weights(chromosome: GroupChromosome) -> GroupChromosome:
    """EXTENSION: canonical equal group weights (all weight bits set)."""
    width = len(chromosome.weight_bits[0])
    return replace(chromosome, weight_bits=tuple((1,) * width for _ in chromosome.weight_bits))


def group_allocations(chromosome: GroupChromosome, group_weights: np.ndarray | None = None) -> np.ndarray:
    """Per-strategy capital: group weight split equally among its members."""
    if group_weights is None:
        return chromosome.allocations()
    weights = np.zeros(len(chromosome.assignments))
    for members, weight in zip(chromosome.groups, group_weights):
        weights[list(members)] = weight / len(members)
    return weights


def decode_bits(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def _valid_bits(bits) -> bool:
    return isinstance(bits, tuple) and 1 <= len(bits) <= 32 and all(type(x) is int and x in (0, 1) for x in bits)


@dataclass(frozen=True)
class GroupChromosome:
    """Four parts: TPSL bits, group labels, bank strategy IDs, group weight bits.

    Strategy ID is its index in assignments; each appears once. All groups must
    be nonempty. Weight codes normalize across groups; there is no cash gene.
    """

    assignments: tuple[int, ...]
    weight_bits: tuple[tuple[int, ...], ...]
    take_profit_bits: tuple[int, ...]
    stop_loss_bits: tuple[int, ...]

    def __post_init__(self):
        k = len(self.weight_bits)
        if (k < 2 or len(self.assignments) < k
                or any(type(x) is not int or not 0 <= x < k for x in self.assignments)
                or set(self.assignments) != set(range(k))):
            raise ValueError("Each strategy must belong to exactly one of >=2 nonempty groups")
        if (any(not _valid_bits(bits) for bits in self.weight_bits)
                or len({len(bits) for bits in self.weight_bits}) != 1
                or not _valid_bits(self.take_profit_bits) or not _valid_bits(self.stop_loss_bits)):
            raise ValueError("TPSL and equal-width group weights must be nonempty binary tuples")

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(i for i, label in enumerate(self.assignments) if label == group)
                     for group in range(len(self.weight_bits)))

    def group_weights(self) -> np.ndarray:
        values = np.array([decode_bits(bits) for bits in self.weight_bits], dtype=float)
        return values / values.sum() if values.sum() else np.full(len(values), 1/len(values))

    def allocations(self) -> np.ndarray:
        weights = np.zeros(len(self.assignments))
        for members, weight in zip(self.groups, self.group_weights()):
            weights[list(members)] = weight / len(members)
        return weights

    def thresholds(self, take_profit_limit=.1, stop_loss_limit=.1) -> tuple[float, float]:
        tp = (decode_bits(self.take_profit_bits) / (2**len(self.take_profit_bits)-1)) * take_profit_limit
        sl = (decode_bits(self.stop_loss_bits) / (2**len(self.stop_loss_bits)-1)) * stop_loss_limit
        return float(tp), float(sl)


def random_chromosome(rng: np.random.Generator, strategies: int, groups: int,
                      take_profit_bits=8, stop_loss_bits=8, weight_bits=8) -> GroupChromosome:
    if not 2 <= groups <= strategies:
        raise ValueError("Chromosomes require at least one strategy per group")
    labels = np.r_[np.arange(groups), rng.integers(groups, size=strategies-groups)]
    rng.shuffle(labels)
    def bits(n):
        return tuple(int(x) for x in rng.integers(2, size=n))
    return GroupChromosome(tuple(int(x) for x in labels), tuple(bits(weight_bits) for _ in range(groups)),
                           bits(take_profit_bits), bits(stop_loss_bits))


def _repair_groups(assignments, groups, rng):
    labels = list(assignments)
    counts = np.bincount(labels, minlength=groups)
    for missing in np.flatnonzero(counts == 0):
        donors = [i for i, label in enumerate(labels) if counts[label] > 1]
        position = int(rng.choice(donors))
        counts[labels[position]] -= 1
        labels[position] = int(missing)
        counts[missing] += 1
    return tuple(labels)


def crossover(first: GroupChromosome, second: GroupChromosome, rng: np.random.Generator) -> GroupChromosome:
    """§3.3.4 one-point crossover on TPSL, grouping and flattened weight parts."""
    shape = lambda c: (len(c.assignments), len(c.weight_bits), len(c.weight_bits[0]),
                       len(c.take_profit_bits), len(c.stop_loss_bits))
    if shape(first) != shape(second):
        raise ValueError("Parents must have the same strategy, group and bit dimensions")
    def splice(a, b):
        cut = int(rng.integers(1, len(a))) if len(a) > 1 else 1
        return a[:cut] + b[cut:]
    tpsl = splice(first.take_profit_bits + first.stop_loss_bits, second.take_profit_bits + second.stop_loss_bits)
    n_tp, width = len(first.take_profit_bits), len(first.weight_bits[0])
    flat = splice(sum(first.weight_bits, ()), sum(second.weight_bits, ()))
    weights = tuple(flat[i:i+width] for i in range(0, len(flat), width))
    labels = _repair_groups(splice(first.assignments, second.assignments), len(weights), rng)
    return GroupChromosome(labels, weights, tpsl[:n_tp], tpsl[n_tp:])


def mutate(chromosome: GroupChromosome, rng: np.random.Generator) -> GroupChromosome:
    """Flip a TPSL bit and weight bit; move a TS without emptying its group."""
    tpsl = list(chromosome.take_profit_bits + chromosome.stop_loss_bits)
    bit = int(rng.integers(len(tpsl)))
    tpsl[bit] ^= 1
    weights = [list(bits) for bits in chromosome.weight_bits]
    group, bit = int(rng.integers(len(weights))), int(rng.integers(len(weights[0])))
    weights[group][bit] ^= 1
    labels = list(chromosome.assignments)
    counts = np.bincount(labels, minlength=len(weights))
    donors = [i for i, label in enumerate(labels) if counts[label] > 1]
    if donors:
        position = int(rng.choice(donors))
        labels[position] = int(rng.choice([g for g in range(len(weights)) if g != labels[position]]))
    n_tp = len(chromosome.take_profit_bits)
    return GroupChromosome(tuple(labels), tuple(tuple(bits) for bits in weights), tuple(tpsl[:n_tp]), tuple(tpsl[n_tp:]))


def group_crossover(first: GroupChromosome, second: GroupChromosome, rng: np.random.Generator) -> GroupChromosome:
    """EXTENSION: inject whole groups of ``second`` (members and weight bits).

    Label-wise n-point crossover mixes two arbitrary labelings of possibly the
    same grouping (m! equivalent chromosomes, Lewis and Pullin). Falkenauer's
    grouping GA instead transmits whole groups. Here a random nonempty, proper
    subset of labels is copied from the second parent; strategies the first
    parent placed in those labels but the second did not are reassigned to a
    random retained label; empty groups are repaired. TP/SL use the paper splice.
    """
    k, n = len(first.weight_bits), len(first.assignments)
    if (len(second.weight_bits), len(second.assignments)) != (k, n):
        raise ValueError("Parents must have the same strategy and group dimensions")
    inject = rng.random(k) < .5
    if inject.all() or not inject.any():
        inject[:] = False
        inject[int(rng.integers(k))] = True
    labels = [-1] * n
    weights = list(first.weight_bits)
    for strategy, label in enumerate(second.assignments):
        if inject[label]:
            labels[strategy] = label
    for label in np.flatnonzero(inject):
        weights[int(label)] = second.weight_bits[int(label)]
    kept = np.flatnonzero(~inject)
    for strategy, label in enumerate(first.assignments):
        if labels[strategy] == -1:
            labels[strategy] = label if not inject[label] else int(rng.choice(kept))
    labels = _repair_groups(tuple(int(x) for x in labels), k, rng)
    cut_source = first.take_profit_bits + first.stop_loss_bits
    other = second.take_profit_bits + second.stop_loss_bits
    cut = int(rng.integers(1, len(cut_source))) if len(cut_source) > 1 else 1
    tpsl = cut_source[:cut] + other[cut:]
    n_tp = len(first.take_profit_bits)
    return GroupChromosome(labels, tuple(weights), tpsl[:n_tp], tpsl[n_tp:])


def bitwise_mutate(chromosome: GroupChromosome, rng: np.random.Generator) -> GroupChromosome:
    """EXTENSION: standard-bit mutation on each part at rate 1/length.

    The paper operator fires for 1% of children and then changes exactly one
    TP/SL bit, one weight bit and one strategy, about six mutations per small
    fit (audit item 9). Here every child mutates: each TP/SL bit and each weight
    bit flips with probability 1/length of its part, and each strategy moves
    with probability 1/n while keeping all groups nonempty.
    """
    tpsl = np.array(chromosome.take_profit_bits + chromosome.stop_loss_bits)
    tpsl ^= (rng.random(len(tpsl)) < 1 / len(tpsl)).astype(tpsl.dtype)
    weights = np.array(chromosome.weight_bits)
    weights ^= (rng.random(weights.shape) < 1 / weights.size).astype(weights.dtype)
    labels = list(chromosome.assignments)
    k = len(chromosome.weight_bits)
    for position in np.flatnonzero(rng.random(len(labels)) < 1 / len(labels)):
        counts = np.bincount(labels, minlength=k)
        if counts[labels[position]] > 1:
            labels[position] = int(rng.choice([g for g in range(k) if g != labels[position]]))
    n_tp = len(chromosome.take_profit_bits)
    return GroupChromosome(tuple(labels), tuple(tuple(int(x) for x in row) for row in weights),
                           tuple(int(x) for x in tpsl[:n_tp]), tuple(int(x) for x in tpsl[n_tp:]))


def invert(chromosome: GroupChromosome, rng: np.random.Generator) -> GroupChromosome:
    """Reverse two selected groups' membership order, keeping position weights."""
    a, b = (int(x) for x in rng.choice(len(chromosome.weight_bits), 2, replace=False))
    labels = tuple(b if label == a else a if label == b else label for label in chromosome.assignments)
    return replace(chromosome, assignments=labels)


@dataclass
class _Evaluator:
    market: PreparedMarket
    signals: np.ndarray
    execution: ExecutionConfig
    take_profit_limit: float
    stop_loss_limit: float
    risk_floor: float
    ratio_cap: float
    profiles: OrderedDict = field(default_factory=OrderedDict)
    variant: str = "paper"
    sign_adjusted: bool = False
    # EXTENSION (review 2026-11-05): daily marked-to-market profiles and the
    # stochastic blend-excess fitness state.
    benchmark: np.ndarray | None = None
    periods_per_year: float = 252.0
    sampler: object | None = None
    mask: np.ndarray | None = None

    @property
    def daily(self) -> bool:
        return self.variant in ("mixture_mtm", "blend_excess")

    @property
    def stochastic(self) -> bool:
        return self.variant == "blend_excess" and self.sampler is not None

    @property
    def _cache_limit(self) -> int:
        return 64 if self.daily else 256

    def set_generation(self, generation):
        """Select the generation's training-month sample (None = full window)."""
        if self.sampler is not None:
            self.mask = self.sampler.mask(generation)

    def _fitness(self, chromosome, profile):
        if self.variant == "blend_excess":
            return portfolio_excess_fitness(chromosome, profile.daily_wealth, profile.held, profile.trade_counts,
                                            self.benchmark, self.mask, periods_per_year=self.periods_per_year)
        wealth = profile.daily_wealth if self.variant == "mixture_mtm" else profile.wealth
        return portfolio_fitness(chromosome, wealth, profile.trade_counts, risk_floor=self.risk_floor,
                                 ratio_cap=self.ratio_cap, variant=self.variant, sign_adjusted=self.sign_adjusted)

    def key(self, chromosome):
        linear = self.execution.instrument is None or self.execution.instrument.scale_invariant
        return (chromosome.take_profit_bits, chromosome.stop_loss_bits) if linear else chromosome

    def profiles_for(self, chromosomes) -> dict:
        """Profiles for these chromosomes' exit policies, keyed like the cache.

        Computed in blocks and returned directly, so the result never depends on
        cache eviction (used to score every evaluated chromosome at the end).
        """
        wanted = {self.key(c):c for c in chromosomes}
        found = {key:self.profiles[key] for key in wanted if key in self.profiles}
        missing = [(key, c) for key, c in wanted.items() if key not in found]
        if self.execution.instrument is None:
            for key, c in missing:
                tp, sl = c.thresholds(self.take_profit_limit, self.stop_loss_limit)
                found[key] = monthly_trade_profiles(self.market, self.signals,
                                                    replace(self.execution, take_profit=tp, stop_loss=sl), daily=self.daily)
            return found
        n = self.signals.shape[1]
        linear = self.execution.instrument.scale_invariant
        for offset in range(0, len(missing), 32):
            block = missing[offset:offset+32]
            thresholds = [c.thresholds(self.take_profit_limit, self.stop_loss_limit) for _, c in block]
            capital = None if linear else np.concatenate([self.execution.initial_cash*c.allocations() for _, c in block])
            batch = monthly_trade_profiles(self.market, np.tile(self.signals, (1, len(block))), self.execution,
                                           initial_cash=capital, take_profit=np.repeat([v[0] for v in thresholds], n),
                                           stop_loss=np.repeat([v[1] for v in thresholds], n), daily=self.daily)
            for i, (key, _) in enumerate(block):
                view = slice(i*n, (i+1)*n)
                found[key] = MonthlyProfiles(batch.wealth[view], batch.trade_counts[view], batch.mean_returns[view],
                                             batch.monthly_trade_counts[view], batch.months,
                                             None if batch.daily_wealth is None else batch.daily_wealth[view],
                                             None if batch.held is None else batch.held[view])
        return found

    def profile(self, chromosome):
        """The (cached) bank profile under this chromosome's exit policy."""
        self.evaluate_many([chromosome])
        linear = self.execution.instrument is None or self.execution.instrument.scale_invariant
        return self.profiles[(chromosome.take_profit_bits, chromosome.stop_loss_bits) if linear else chromosome]

    def evaluate(self, chromosome):
        if self.execution.instrument is not None:
            return self.evaluate_many([chromosome])[0]
        key = chromosome.take_profit_bits, chromosome.stop_loss_bits
        if key not in self.profiles:
            tp, sl = chromosome.thresholds(self.take_profit_limit, self.stop_loss_limit)
            config = replace(self.execution, take_profit=tp, stop_loss=sl)
            self.profiles[key] = monthly_trade_profiles(self.market, self.signals, config, daily=self.daily)
            if len(self.profiles) > self._cache_limit:
                self.profiles.popitem(last=False)
        self.profiles.move_to_end(key)
        return self._fitness(chromosome, self.profiles[key])

    def evaluate_many(self,chromosomes):
        if self.execution.instrument is None:
            return [self.evaluate(c) for c in chromosomes]
        linear=self.execution.instrument.scale_invariant
        key_for=lambda c:(c.take_profit_bits,c.stop_loss_bits) if linear else c
        wanted={key_for(c):c for c in chromosomes}
        profiles={key:self.profiles[key] for key in wanted if key in self.profiles}
        missing=[(key,c) for key,c in wanted.items() if key not in profiles]
        n=self.signals.shape[1]
        for offset in range(0,len(missing),32):
            block=missing[offset:offset+32]
            decisions=np.tile(self.signals,(1,len(block)))
            thresholds=[c.thresholds(self.take_profit_limit,self.stop_loss_limit) for _,c in block]
            tp=np.repeat([v[0] for v in thresholds],n); sl=np.repeat([v[1] for v in thresholds],n)
            capital=None if linear else np.concatenate([self.execution.initial_cash*c.allocations() for _,c in block])
            batch=monthly_trade_profiles(self.market,decisions,self.execution,initial_cash=capital,take_profit=tp,stop_loss=sl,
                                         daily=self.daily)
            for i,(key,_) in enumerate(block):
                view=slice(i*n,(i+1)*n)
                profile=MonthlyProfiles(batch.wealth[view],batch.trade_counts[view],batch.mean_returns[view],
                                        batch.monthly_trade_counts[view],batch.months,
                                        None if batch.daily_wealth is None else batch.daily_wealth[view],
                                        None if batch.held is None else batch.held[view])
                profiles[key]=profile; self.profiles[key]=profile
                if len(self.profiles)>self._cache_limit:self.profiles.popitem(last=False)
        return [self._fitness(c,profiles[key_for(c)]) for c in chromosomes]


@dataclass
class _Island:
    population: list[GroupChromosome]
    rng: np.random.Generator
    cache: dict[GroupChromosome, dict] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    best: list = field(default_factory=list)
    # Every chromosome this island has scored (the cache is cleared each
    # generation under stochastic fitness).
    evaluated: set = field(default_factory=set)

    def remember(self, chromosomes, metrics):
        chromosomes = list(chromosomes)
        self.cache.update(zip(chromosomes, metrics))
        self.evaluated.update(chromosomes)

    def score(self, chromosome, evaluator=None):
        if chromosome not in self.cache:
            if evaluator is None:
                raise ValueError("Migration needs evaluated chromosomes")
            self.cache[chromosome] = evaluator.evaluate(chromosome)
            self.evaluated.add(chromosome)
        return self.cache[chromosome]["fitness"]


def elite_selection(island: _Island, selection_rate: float, evaluator=None) -> list[GroupChromosome]:
    ranked = sorted(island.population, key=lambda c:island.score(c, evaluator), reverse=True)
    return ranked[:max(1, int(np.floor(len(ranked)*selection_rate)))]


def _advance_island(island: _Island, evaluator: _Evaluator, steps: int, settings: dict) -> _Island:
    def score(c):
        return island.score(c, evaluator)
    def precache(candidates):
        missing=list(dict.fromkeys(c for c in candidates if c not in island.cache))
        island.remember(missing,evaluator.evaluate_many(missing))
    crossover_mode = settings.get("crossover_mode", "paper")
    mutation_mode = settings.get("mutation_mode", "paper")
    unique = settings.get("unique_children", False)
    recombine = group_crossover if crossover_mode == "group" else crossover
    equal_weights = settings.get("equal_weights", False)
    stochastic = getattr(evaluator, "stochastic", False)

    def begin(generation):
        # EXTENSION: a fresh training-month sample per generation; every member,
        # elites included, is re-scored on it.
        if stochastic:
            evaluator.set_generation(generation)
            island.cache.clear()
            precache(island.population)

    def breed(parents):
        child = parents[int(island.rng.integers(len(parents)))]
        if island.rng.random() < settings["crossover_probability"]:
            child = recombine(child, parents[int(island.rng.integers(len(parents)))], island.rng)
        if mutation_mode == "bitwise":
            child = bitwise_mutate(child, island.rng)
        elif island.rng.random() < settings["mutation_probability"]:
            child = mutate(child, island.rng)
        if island.rng.random() < settings["inversion_probability"]:
            child = invert(child, island.rng)
        return uniform_weights(child) if equal_weights else child

    if not island.history:
        begin(0)
        precache(island.population)
        island.population.sort(key=score, reverse=True)
        island.history.append({"generation":0, "best_fitness":score(island.population[0])})
        island.best.append(island.population[0])
    for _ in range(steps):
        begin(len(island.history))
        parents = elite_selection(island, settings["selection_rate"], evaluator)
        children = list(parents)
        seen = set(island.population) | set(children)
        while len(children) < len(island.population):
            child = breed(parents)
            if unique:
                # EXTENSION: resample duplicates; a last bitwise mutation is the fallback.
                for _attempt in range(20):
                    if child not in seen:
                        break
                    child = breed(parents)
                if child in seen:
                    child = bitwise_mutate(child, island.rng)
                seen.add(child)
            children.append(child)
        precache(children)
        island.population = sorted(children, key=score, reverse=True)
        island.history.append({"generation":len(island.history), "best_fitness":score(island.population[0])})
        island.best.append(island.population[0])
    return island


_worker_evaluator = None


def _initialize_island_profiles(evaluator: _Evaluator) -> None:
    global _worker_evaluator
    evaluator.signals.setflags(write=False)
    _worker_evaluator = evaluator


def _advance_island_worker(island: _Island, steps: int, settings: dict) -> _Island:
    return _advance_island(island, _worker_evaluator, steps, settings)


def ring_migrate(islands: list[_Island], count: int | None = None, *, migration_rate=.5,
                 evaluator=None) -> list[dict]:
    """Algorithm 1: sequential adjacent BEST-for-BEST exchanges, including wrap.

    A later pair sees earlier exchanges, as in the printed pseudocode. Migrants
    are removed from their source, not cloned over destination worst members.
    """
    if len(islands) < 2 or not np.isfinite(migration_rate) or not 0 <= migration_rate <= 1:
        raise ValueError("Migration needs >=2 islands and a rate in [0,1]")
    if count is not None and (type(count) is not int or not 1 <= count <= min(len(i.population) for i in islands)):
        raise ValueError("Migrant count must fit every island")
    events = []
    for i, left in enumerate(islands):
        right_index = (i+1) % len(islands)
        right = islands[right_index]
        amount = count if count is not None else int(np.floor(migration_rate * min(len(left.population),len(right.population))))
        if amount == 0:
            continue
        if evaluator is not None:
            for island in (left,right):
                missing=list(dict.fromkeys(c for c in island.population if c not in island.cache))
                island.remember(missing,evaluator.evaluate_many(missing))
        left.population.sort(key=lambda c:left.score(c, evaluator), reverse=True)
        right.population.sort(key=lambda c:right.score(c, evaluator), reverse=True)
        a, b = left.population[:amount], right.population[:amount]
        left.population = left.population[amount:] + b
        right.population = right.population[amount:] + a
        events.append({"left_island":i, "right_island":right_index, "count_each_direction":amount})
    return events


def facility_location_bank(monthly_returns: np.ndarray, returns: np.ndarray, risks: np.ndarray,
                           qualified: np.ndarray, size: int, pool_multiple: int = 4) -> np.ndarray:
    """EXTENSION: correlation-diverse bank by greedy facility location.

    Candidates are qualified strategies with positive training return, ranked
    by training return/risk; the best ``pool_multiple*size`` form the pool.
    Greedy maximization of f(A) = sum_j max_{i in A} (1 + corr_ij)/2 over the
    pool is monotone submodular, so greedy selection is within (1 - 1/e) of
    the best ``size``-subset for this coverage objective (Nemhauser, Wolsey and
    Fisher 1978). Remaining slots, if any, fall back to the paper ranking.
    """
    ratio = returns / np.maximum(risks, 1e-12)
    positive = qualified[returns[qualified] > 0]
    order = positive[np.lexsort((positive, -ratio[positive]))]
    pool = order[:max(size, pool_multiple * size)]
    if len(pool) <= size:
        fallback = qualified[np.lexsort((qualified, risks[qualified], -returns[qualified]))]
        return np.asarray(list(pool) + [i for i in fallback if i not in set(pool)][:size - len(pool)])
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(monthly_returns[pool])
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    similarity = (1.0 + correlation) / 2.0
    chosen: list[int] = []
    coverage = np.zeros(len(pool))
    for _ in range(size):
        gains = np.maximum(similarity, coverage[None, :]).sum(axis=1) - coverage.sum()
        gains[chosen] = -np.inf
        best = int(np.argmax(gains))
        chosen.append(best)
        coverage = np.maximum(coverage, similarity[best])
    return pool[np.asarray(chosen)]


class IGTSPRing:
    def __init__(self, population_size=150, generations=100, seed=0, *, island_count=10,
                 groups=17, strategy_count=34, take_profit_bits=8, stop_loss_bits=8, weight_bits=8,
                 take_profit_limit=.1, stop_loss_limit=.1, migration_interval=10, migration_rate=.5,
                 migrants=None, island_workers=1, crossover_probability=.6, mutation_probability=.01,
                 inversion_probability=.1, selection_rate=.5, risk_floor=1e-12, ratio_cap=1e9,
                 bank: tuple[Strategy, ...] | None = None, execution: ExecutionConfig | None = None,
                 fitness: str = "paper", sign_adjusted: bool = False, crossover_mode: str = "paper",
                 mutation_mode: str = "paper", unique_children: bool = False, bank_selection: str = "paper",
                 bank_ranking: str = "paper", migration_schedule: str = "paper", weight_mode: str = "ga",
                 subwindow_fraction: float = 0.5, decision_cadence: str = "daily"):
        for name, value, allowed in (("fitness", fitness, FITNESS_VARIANTS), ("crossover_mode", crossover_mode, CROSSOVER_MODES),
                                     ("weight_mode", weight_mode, WEIGHT_MODES),
                                     ("decision_cadence", decision_cadence, DECISION_CADENCES),
                                     ("mutation_mode", mutation_mode, MUTATION_MODES), ("bank_selection", bank_selection, BANK_SELECTIONS),
                                     ("bank_ranking", bank_ranking, BANK_RANKINGS),
                                     ("migration_schedule", migration_schedule, MIGRATION_SCHEDULES)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        self.fitness, self.sign_adjusted = fitness, bool(sign_adjusted)
        self.bank_selection, self.bank_ranking, self.migration_schedule = bank_selection, bank_ranking, migration_schedule
        if not np.isfinite(subwindow_fraction) or not 0 < subwindow_fraction <= 1:
            raise ValueError("subwindow_fraction must be in (0,1]")
        self.weight_mode, self.subwindow_fraction = weight_mode, float(subwindow_fraction)
        self.decision_cadence = decision_cadence
        for name, value, minimum in (("population_size",population_size,4), ("generations",generations,0),
                ("seed",seed,0), ("island_count",island_count,2), ("groups",groups,2),
                ("strategy_count",strategy_count,2), ("take_profit_bits",take_profit_bits,1),
                ("stop_loss_bits",stop_loss_bits,1), ("weight_bits",weight_bits,1),
                ("migration_interval",migration_interval,1), ("island_workers",island_workers,1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        candidate_bank = tuple(bank) if bank is not None else DEFAULT_BANK
        if (population_size // island_count < 2 or groups > strategy_count or strategy_count > len(candidate_bank)
                or len({s.name for s in candidate_bank}) != len(candidate_bank)):
            raise ValueError("Population needs >=2 per island; selected strategy count must cover nonempty groups and fit the unique bank")
        if max(take_profit_bits, stop_loss_bits, weight_bits) > 32:
            raise ValueError("Bit widths must be at most 32")
        if migrants is not None and (type(migrants) is not int or not 1 <= migrants <= population_size//island_count):
            raise ValueError("migrants must fit the smallest island")
        for name, value in (("crossover_probability",crossover_probability), ("mutation_probability",mutation_probability),
                            ("inversion_probability",inversion_probability), ("migration_rate",migration_rate)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if not np.isfinite(selection_rate) or not 0 < selection_rate <= 1:
            raise ValueError("selection_rate must be in (0,1]")
        if not np.isfinite(take_profit_limit) or take_profit_limit <= 0:
            raise ValueError("take_profit_limit must be positive")
        if not np.isfinite(stop_loss_limit) or not 0 < stop_loss_limit < 1:
            raise ValueError("stop_loss_limit must be in (0,1)")
        if any(not np.isfinite(x) or x <= 0 for x in (risk_floor, ratio_cap)):
            raise ValueError("Fitness guards must be finite and positive")
        self.population_size, self.generations, self.seed = population_size, generations, seed
        self.island_count, self.groups, self.strategy_count = island_count, groups, strategy_count
        self.take_profit_bits, self.stop_loss_bits, self.weight_bits = take_profit_bits, stop_loss_bits, weight_bits
        self.take_profit_limit, self.stop_loss_limit = float(take_profit_limit), float(stop_loss_limit)
        self.migration_interval, self.migration_rate, self.migrants = migration_interval, float(migration_rate), migrants
        self.island_workers, self.risk_floor, self.ratio_cap = island_workers, float(risk_floor), float(ratio_cap)
        self.candidate_bank, self.execution = candidate_bank, execution or ExecutionConfig(fee=0.)
        self._settings = dict(crossover_probability=float(crossover_probability), mutation_probability=float(mutation_probability),
                              inversion_probability=float(inversion_probability), selection_rate=float(selection_rate))
        if crossover_mode != "paper" or mutation_mode != "paper" or unique_children:
            self._settings.update(crossover_mode=crossover_mode, mutation_mode=mutation_mode,
                                  unique_children=bool(unique_children))
        if weight_mode != "ga":
            self._settings["equal_weights"] = True
        self._selected: GroupChromosome | None = None
        self._group_weights: np.ndarray | None = None
        self._snapshot_weights: dict = {}
        self._bank: tuple[Strategy, ...] = ()
        self._diagnostics: dict = {}

    @staticmethod
    def _check_external(extra_features):
        if extra_features is not None and (not isinstance(extra_features, Mapping) or len(extra_features)):
            raise ValueError("IGTSP-Ring uses its training-selected strategy bank; external GP features are unsupported")

    def _rank_bank(self, market, pool_signals, take_profit, stop_loss):
        profile = monthly_trade_profiles(market, pool_signals, replace(self.execution,
                                         take_profit=take_profit, stop_loss=stop_loss))
        returns = profile.wealth[:, -1]-1
        risks = np.max(np.maximum.accumulate(profile.wealth, axis=1)-profile.wealth, axis=1)
        qualified=np.flatnonzero(profile.trade_counts>=2)
        if len(qualified)<self.strategy_count:
            raise ValueError("Too few training-active strategies with two completed trades for the selected bank size")
        if self.bank_selection == "diverse":
            return facility_location_bank(profile.mean_returns, returns, risks, qualified, self.strategy_count)
        ranking=np.lexsort((qualified,risks[qualified],-returns[qualified]))
        return qualified[ranking[:self.strategy_count]]

    def _evolve(self, evaluator, generations, seed_sequence, seed_tpsl=None):
        sizes = [self.population_size//self.island_count + (i < self.population_size%self.island_count)
                 for i in range(self.island_count)]
        islands = []
        for size, seed in zip(sizes, seed_sequence.spawn(self.island_count)):
            rng = np.random.default_rng(seed)
            population = [random_chromosome(rng, self.strategy_count, self.groups, self.take_profit_bits,
                                            self.stop_loss_bits, self.weight_bits) for _ in range(size)]
            if self.weight_mode != "ga":
                population = [uniform_weights(c) for c in population]
            if seed_tpsl is not None:
                # Two-stage extension: half of each island inherits stage-1 TP/SL.
                population = [replace(c, take_profit_bits=seed_tpsl[0], stop_loss_bits=seed_tpsl[1]) if i % 2 == 0 else c
                              for i, c in enumerate(population)]
            islands.append(_Island(population, rng))
        migrations, global_history, intervals = [], [], []
        worker_pool = process_pool(min(self.island_workers,self.island_count), initializer=_initialize_island_profiles,
                                   initargs=(evaluator,)) if self.island_workers > 1 else None
        adaptive = self.migration_schedule == "adaptive"
        # Adaptive schemes start from interval 1 and double while islands stall.
        interval = 1 if adaptive else self.migration_interval
        best_before = -np.inf
        try:
            completed = 0
            while True:
                steps = min(interval, generations-completed)
                if worker_pool:
                    futures = [worker_pool.submit(_advance_island_worker, island, steps, self._settings) for island in islands]
                    islands = [future.result() for future in futures]
                else:
                    islands = [_advance_island(island, evaluator, steps, self._settings) for island in islands]
                completed += steps
                best_now = max(island.score(c, evaluator) for island in islands for c in island.population)
                if adaptive:
                    # Scheme B (Mambrini and Sudholt 2015), applied globally: halve
                    # the interval after an improving epoch, double it otherwise.
                    # No exchange after the last generation, where it cannot matter.
                    if completed < generations:
                        if evaluator.stochastic:
                            evaluator.set_generation(completed)
                        for event in ring_migrate(islands, self.migrants, migration_rate=self.migration_rate, evaluator=evaluator):
                            migrations.append({"generation":completed, **event})
                    intervals.append({"generation":completed, "interval":interval, "improved":bool(best_now > best_before)})
                    interval = max(1, interval//2) if best_now > best_before else min(2*interval, max(generations, 1))
                elif completed and completed % self.migration_interval == 0:
                    if evaluator.stochastic:
                        evaluator.set_generation(completed)
                    for event in ring_migrate(islands, self.migrants, migration_rate=self.migration_rate, evaluator=evaluator):
                        migrations.append({"generation":completed, **event})
                best_before = max(best_before, best_now)
                global_history.append({"generation":completed,
                    "best_fitness":max(island.score(c, evaluator) for island in islands for c in island.population)})
                if completed == generations:
                    break
        finally:
            if worker_pool:
                worker_pool.shutdown(wait=True, cancel_futures=True)
        return islands, sizes, migrations, global_history, intervals

    def fit(self, bars: pd.DataFrame, extra_features=None) -> IGTSPRing:
        self._selected, self._bank, self._diagnostics = None, (), {}
        self._snapshots = []
        self._group_weights, self._snapshot_weights = None, {}
        self._check_external(extra_features)
        frame = validate_bars(bars)
        market = PreparedMarket.from_bars(frame,self.execution.instrument)
        volume_missing=not np.isfinite(frame['volume'].to_numpy()).all()
        candidate_bank=tuple(s for s in self.candidate_bank if not volume_missing or
            not any(rule.startswith(('obv:','mfi14:','adosc:')) for rule in (s.buy_rule,s.sell_rule)))
        if len(candidate_bank)<self.strategy_count:
            raise ValueError("Too few strategies remain after excluding unavailable volume rules")
        pool_signals = apply_cadence(strategy_signals(frame, candidate_bank), frame["timestamp"], self.decision_cadence)
        # The text omits 'best n' ranking: declare return-first and risk/index tie-breaks.
        selected_indices = self._rank_bank(market, pool_signals, self.take_profit_limit, self.stop_loss_limit)
        seeds = np.random.SeedSequence(self.seed)
        excess_state = {}
        if self.fitness == "blend_excess":
            from .robust_fitness import SubWindows, bar_months, benchmark_returns, periods_per_year
            excess_state = dict(benchmark=benchmark_returns(frame, self.execution, market.advanced),
                                periods_per_year=periods_per_year(frame),
                                sampler=SubWindows(bar_months(frame), self.seed, self.subwindow_fraction))
        stages = []
        stage_plan = ([self.generations] if self.bank_ranking == "paper"
                      else [self.generations//2, self.generations-self.generations//2])
        seed_tpsl = None
        for stage, stage_generations in enumerate(stage_plan):
            if stage:
                # Two-stage extension (audit item 11): re-rank the pool under the
                # stage-1 winner's evolved TP/SL, then evolve again on that bank.
                tp, sl = winner.thresholds(self.take_profit_limit, self.stop_loss_limit)
                selected_indices = self._rank_bank(market, pool_signals, tp, sl)
                seed_tpsl = (winner.take_profit_bits, winner.stop_loss_bits)
            evaluator = _Evaluator(market, pool_signals[:, selected_indices], self.execution, self.take_profit_limit,
                                   self.stop_loss_limit, self.risk_floor, self.ratio_cap,
                                   variant=self.fitness, sign_adjusted=self.sign_adjusted, **excess_state)
            stage_seed = seeds if not stage else np.random.SeedSequence([self.seed, stage])
            islands, sizes, migrations, global_history, intervals = self._evolve(evaluator, stage_generations, stage_seed, seed_tpsl)
            candidates = [c for island in islands for c in island.population]
            evaluator.set_generation(None)  # final choice on the full training window
            final_metrics=evaluator.evaluate_many(candidates)
            winning_index=max(range(len(candidates)),key=lambda i:final_metrics[i]['fitness'])
            winner,metrics=candidates[winning_index],final_metrics[winning_index]
            stages.append({"generations":stage_generations, "selected_pool_indices":[int(i) for i in selected_indices],
                           "selected_training":metrics,
                           "take_profit_stop_loss":list(winner.thresholds(self.take_profit_limit, self.stop_loss_limit))})
        self._bank = tuple(candidate_bank[int(i)] for i in selected_indices)
        if not metrics["eligible"]:
            self._bank = ()
            raise ValueError("IGTSP-Ring found no eligible portfolio with two completed sleeve trades")
        self._selected = winner
        # Global best chromosome per generation (final stage) for validation curves.
        for generation in range(min(len(island.best) for island in islands)):
            leader = max(islands, key=lambda island: island.history[generation]["best_fitness"])
            self._snapshots.append(leader.best[generation])
        lp_report = None
        if self.weight_mode == "lp":
            # EXTENSION: memberships and TP/SL evolved with equal weights; group
            # weights then solved exactly on training marked-to-market paths.
            def solve(chromosome):
                tp, sl = chromosome.thresholds(self.take_profit_limit, self.stop_loss_limit)
                daily = monthly_trade_profiles(market, evaluator.signals, replace(self.execution, take_profit=tp, stop_loss=sl),
                                               daily=True).daily_wealth
                paths = np.vstack([daily[list(members)].mean(axis=0) for members in chromosome.groups])
                weights, info = drawdown_ratio_weights(paths, risk_floor=max(self.risk_floor, 1e-9))
                return (weights if weights is not None else np.full(len(paths), 1 / len(paths))), info
            self._group_weights, lp_report = solve(winner)
            last = len(self._snapshots) - 1
            for generation, chromosome in enumerate(self._snapshots):
                if (generation % 5 == 0 or generation == last) and chromosome not in self._snapshot_weights:
                    self._snapshot_weights[chromosome] = solve(chromosome)[0]
        haircut = None
        if self.fitness == "blend_excess":
            from .robust_fitness import excess_paths, period_returns, selection_haircut
            seen = {c for island in islands for c in island.evaluated} | set(candidates)
            order = sorted(seen, key=lambda c:(c.take_profit_bits, c.stop_loss_bits, c.assignments, c.weight_bits))
            by_key = {}
            for chromosome in order:
                by_key.setdefault(evaluator.key(chromosome), []).append(chromosome)
            keys, columns, scored = list(by_key), [], []
            for offset in range(0, len(keys), 32):
                block = keys[offset:offset+32]
                profiles = evaluator.profiles_for([by_key[key][0] for key in block])
                for key in block:
                    for chromosome in by_key[key]:
                        allocations = chromosome.allocations()
                        mixture = allocations @ profiles[key].daily_wealth
                        columns.append((period_returns(mixture), allocations @ profiles[key].held))
                        scored.append(chromosome)
            excess = excess_paths(np.column_stack([c[0] for c in columns]), np.column_stack([c[1] for c in columns]),
                                  evaluator.benchmark)
            haircut = selection_haircut(excess, scored.index(winner), periods_per_year=evaluator.periods_per_year,
                                        seed=self.seed)
            haircut["scope"] = "every chromosome evaluated in the final stage"
        evaluated = {c for island in islands for c in island.evaluated} | {c for island in islands for c in island.cache}
        self._diagnostics = {
            "method":"igtsp_ring", "source_doi":SOURCE_DOI, "source_pdf_sha256":SOURCE_PDF_SHA256,
            "fidelity":FIDELITY, "paper_fitness_verified":True, "fitness":"section_3.3.3_eq6_to_16"
            if self.fitness == "paper" else "extension_mixture_drawdown_capital_entropy",
            "training_rows":len(frame), "training_end":frame["available_at"].iloc[-1].isoformat(),
            "training_months":list(market.months), "selected_training":metrics,
            "population_semantics":"total across islands", "island_population_sizes":sizes,
            "unique_evaluations":len(evaluated), "island_evaluation_count":sum(len(i.cache) for i in islands),
            "unique_exit_policies_evaluated":len({(c.take_profit_bits,c.stop_loss_bits) for c in evaluated}),
            "strategy_pool_size":len(candidate_bank), "selected_strategy_count":self.strategy_count,
            "volume_policy":"exclude volume-dependent buy/sell conditions" if volume_missing else "observed volume used",
            "bank_selection":{"ranking":"monthly terminal return descending; absolute monthly drawdown ascending; stable pool index"
                              if self.bank_selection == "paper" else
                              "extension: facility-location diversity over the top return/risk pool",
                              "training_only":True, "ranking_take_profit":self.take_profit_limit,
                              "ranking_stop_loss":self.stop_loss_limit,
                              "selected_pool_indices":[int(i) for i in selected_indices]},
            "migration_events":migrations, "global_best_history":global_history,
            "island_history":[island.history for island in islands],
        }
        if self.fitness != "paper" or self.bank_selection != "paper" or self.bank_ranking != "paper" \
                or self.migration_schedule != "paper" or self.sign_adjusted or len(self._settings) > 4:
            self._diagnostics["variant"] = {
                "fitness":self.fitness, "sign_adjusted":self.sign_adjusted, "bank_selection":self.bank_selection,
                "bank_ranking":self.bank_ranking, "migration_schedule":self.migration_schedule,
                **{k:v for k,v in self._settings.items() if k in ("crossover_mode","mutation_mode","unique_children")},
                "stages":stages, "adaptive_intervals":intervals}
        if self.weight_mode != "ga" or self.fitness in ("mixture_mtm", "blend_excess"):
            self._diagnostics.setdefault("variant", {}).update(weight_mode=self.weight_mode, lp_weights=lp_report,
                                                               selection_haircut=haircut)
            if excess_state:
                self._diagnostics["variant"]["subwindows"] = excess_state["sampler"].describe()
        return self

    def generation_snapshots(self, every: int = 5) -> list[tuple[int, IGTSPRing]]:
        """Frozen copies holding the final stage's global best per generation."""
        if self._selected is None:
            raise RuntimeError("Fit IGTSP-Ring before requesting snapshots")
        result = []
        for generation, chromosome in enumerate(self._snapshots):
            if generation % every == 0 or generation == len(self._snapshots) - 1:
                model = copy.copy(self)
                model._selected = chromosome
                if self.weight_mode == "lp":
                    model._group_weights = self._snapshot_weights.get(chromosome)
                result.append((generation, model))
        return result

    def predict(self, bars: pd.DataFrame, extra_features=None) -> PortfolioSignals:
        if self._selected is None:
            raise RuntimeError("Fit IGTSP-Ring before predicting")
        self._check_external(extra_features)
        tp, sl = self._selected.thresholds(self.take_profit_limit, self.stop_loss_limit)
        signals = apply_cadence(strategy_signals(bars,self._bank), validate_bars(bars)["timestamp"], self.decision_cadence)
        return PortfolioSignals(tuple(s.name for s in self._bank), signals,
                                tuple(float(x) for x in self.allocations()), 0., tp, sl)

    def allocations(self) -> np.ndarray:
        """Deployed per-strategy capital (LP group weights when weight_mode='lp')."""
        return group_allocations(self._selected, self._group_weights)

    def evaluate_paper(self, bars: pd.DataFrame, *, start: int = 0) -> dict:
        """Describe a frozen model's paper metrics separately from common equity."""
        frame = validate_bars(bars)
        if type(start) is not int or not 0 <= start < len(frame):
            raise ValueError("Evaluation start must identify a nonempty window")
        plan = self.predict(frame)
        market = PreparedMarket.from_bars(frame.iloc[start:].reset_index(drop=True),self.execution.instrument)
        profile = monthly_trade_profiles(market, plan.signals[start:], replace(self.execution,
                                         take_profit=plan.take_profit, stop_loss=plan.stop_loss),
                                         initial_cash=self.execution.initial_cash*self.allocations()
                                         if self.execution.instrument and not self.execution.instrument.scale_invariant else None)
        return {**portfolio_fitness(self._selected, profile.wealth, profile.trade_counts,
                                    risk_floor=self.risk_floor, ratio_cap=self.ratio_cap),
                "months":list(profile.months), "strategy_monthly_mean_returns":profile.mean_returns.tolist(),
                "strategy_monthly_trade_counts":profile.monthly_trade_counts.tolist(),
                "uses_frozen_bank_and_tpsl":True,
                "note":"Paper monthly/component-risk metrics; distinct from marked-to-market equity statistics."}

    def parameters(self) -> dict:
        if self._selected is None:
            raise RuntimeError("Fit IGTSP-Ring before requesting parameters")
        tp, sl = self._selected.thresholds(self.take_profit_limit,self.stop_loss_limit)
        return {"method":"igtsp_ring", "paper":SOURCE_DOI, "source_pdf_sha256":SOURCE_PDF_SHA256,
                "fidelity":FIDELITY, "paper_fitness_verified":True, "published_numerical_results_reproduced":False,
                "population_size":self.population_size, "generations":self.generations, "seed":self.seed,
                "island_count":self.island_count, "island_workers":self.island_workers,
                "groups":self.groups, "strategy_count":self.strategy_count,
                "take_profit_bits":self.take_profit_bits, "stop_loss_bits":self.stop_loss_bits, "weight_bits":self.weight_bits,
                "take_profit_limit":self.take_profit_limit, "stop_loss_limit":self.stop_loss_limit,
                "take_profit":tp, "stop_loss":sl, "migration_interval":self.migration_interval,
                "migration_rate":self.migration_rate, "migrants":self.migrants,
                "risk_floor":self.risk_floor, "ratio_cap":self.ratio_cap, **self._settings,
                "execution":asdict(self.execution), "chromosome":asdict(self._selected),
                "group_weights":(self._selected.group_weights() if self._group_weights is None
                                 else np.asarray(self._group_weights)).tolist(),
                "weight_mode":self.weight_mode, "subwindow_fraction":self.subwindow_fraction,
                "decision_cadence":self.decision_cadence,
                "strategy_groups":[[self._bank[i].name for i in members] for members in self._selected.groups],
                "bank":[asdict(strategy) for strategy in self._bank], "fitness_equations":"6-16",
                "portfolio_policy":"held-out equity uses equal member sleeves within each group; evolved TPSL; no cash gene",
                "adaptations":["explicit_21_indicator_state_cross_rule_catalog", "training_only_return_first_bank_ranking",
                    "DMI_and_ADX_counted_separately", "bit_encoded_group_weights_normalized_no_cash_gene",
                    "nonempty_group_repair", "natural_log_entropy", "empty_month_zero_return_exit_month_attribution",
                    "next_open_costed_fills_close_observed_TPSL", "zero_risk_floor_ratio_cap_minimum_two_trades",
                    "one_based_completed_generation_migration", "selection_rate_floor_with_one_parent_minimum",
                    "exact_linear_all_TSP_aggregation_for_eq8_and_eq13"]}

    @property
    def diagnostics(self) -> dict:
        return copy.deepcopy(self._diagnostics)
