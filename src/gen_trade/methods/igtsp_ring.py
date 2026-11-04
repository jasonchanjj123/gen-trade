"""IGTSP-Ring from the supplied full text, DOI 10.1080/0305215X.2025.2592030.

Implements §3.3.2--5, Eqs.6--16, Algorithms 1/4 and Table 5 defaults.
The paper omits exact indicator rules, bank ranking, weight-bit encoding and
several execution details. Those choices are explicit in docs/IGTSP_RING.md.
Section 3.1's alternative objective is NOT silently mixed into Eq.6.
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
from .ring_bank import DEFAULT_BANK, INDICATORS, Strategy, strategy_signals
from .ring_fitness import INVALID_FITNESS, MonthlyProfiles, PreparedMarket, monthly_trade_profiles, portfolio_fitness

SOURCE_DOI = "10.1080/0305215X.2025.2592030"
SOURCE_PDF_SHA256 = "7ccf3ab38a5609ffc5b89e72521587ecb6d9b7b42bdf18b2fa299765b8b31d42"
FIDELITY = "full_text_research_implementation"
PAPER_SETTINGS = {"population_size":150, "generations":100, "island_count":10, "groups":17,
                  "migration_rate":.5, "migration_interval":10, "crossover_probability":.6,
                  "mutation_probability":.01, "selection_rate":.5, "inversion_probability":.1,
                  "take_profit_bits":8, "stop_loss_bits":8}


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

    def evaluate(self, chromosome):
        if self.execution.instrument is not None:
            return self.evaluate_many([chromosome])[0]
        key = chromosome.take_profit_bits, chromosome.stop_loss_bits
        if key not in self.profiles:
            tp, sl = chromosome.thresholds(self.take_profit_limit, self.stop_loss_limit)
            config = replace(self.execution, take_profit=tp, stop_loss=sl)
            self.profiles[key] = monthly_trade_profiles(self.market, self.signals, config)
            if len(self.profiles) > 256:
                self.profiles.popitem(last=False)
        self.profiles.move_to_end(key)
        profile = self.profiles[key]
        return portfolio_fitness(chromosome, profile.wealth, profile.trade_counts,
                                 risk_floor=self.risk_floor, ratio_cap=self.ratio_cap)

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
            batch=monthly_trade_profiles(self.market,decisions,self.execution,initial_cash=capital,take_profit=tp,stop_loss=sl)
            for i,(key,_) in enumerate(block):
                view=slice(i*n,(i+1)*n)
                profile=MonthlyProfiles(batch.wealth[view],batch.trade_counts[view],batch.mean_returns[view],
                                        batch.monthly_trade_counts[view],batch.months)
                profiles[key]=profile; self.profiles[key]=profile
                if len(self.profiles)>256:self.profiles.popitem(last=False)
        return [portfolio_fitness(c,profiles[key_for(c)].wealth,profiles[key_for(c)].trade_counts,
                                  risk_floor=self.risk_floor,ratio_cap=self.ratio_cap) for c in chromosomes]


@dataclass
class _Island:
    population: list[GroupChromosome]
    rng: np.random.Generator
    cache: dict[GroupChromosome, dict] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def score(self, chromosome, evaluator=None):
        if chromosome not in self.cache:
            if evaluator is None:
                raise ValueError("Migration needs evaluated chromosomes")
            self.cache[chromosome] = evaluator.evaluate(chromosome)
        return self.cache[chromosome]["fitness"]


def elite_selection(island: _Island, selection_rate: float, evaluator=None) -> list[GroupChromosome]:
    ranked = sorted(island.population, key=lambda c:island.score(c, evaluator), reverse=True)
    return ranked[:max(1, int(np.floor(len(ranked)*selection_rate)))]


def _advance_island(island: _Island, evaluator: _Evaluator, steps: int, settings: dict) -> _Island:
    def score(c):
        return island.score(c, evaluator)
    def precache(candidates):
        missing=list(dict.fromkeys(c for c in candidates if c not in island.cache))
        island.cache.update(zip(missing,evaluator.evaluate_many(missing)))
    if not island.history:
        precache(island.population)
        island.population.sort(key=score, reverse=True)
        island.history.append({"generation":0, "best_fitness":score(island.population[0])})
    for _ in range(steps):
        parents = elite_selection(island, settings["selection_rate"], evaluator)
        children = list(parents)
        while len(children) < len(island.population):
            child = parents[int(island.rng.integers(len(parents)))]
            if island.rng.random() < settings["crossover_probability"]:
                child = crossover(child, parents[int(island.rng.integers(len(parents)))], island.rng)
            if island.rng.random() < settings["mutation_probability"]:
                child = mutate(child, island.rng)
            if island.rng.random() < settings["inversion_probability"]:
                child = invert(child, island.rng)
            children.append(child)
        precache(children)
        island.population = sorted(children, key=score, reverse=True)
        island.history.append({"generation":len(island.history), "best_fitness":score(island.population[0])})
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
                island.cache.update(zip(missing,evaluator.evaluate_many(missing)))
        left.population.sort(key=lambda c:left.score(c, evaluator), reverse=True)
        right.population.sort(key=lambda c:right.score(c, evaluator), reverse=True)
        a, b = left.population[:amount], right.population[:amount]
        left.population = left.population[amount:] + b
        right.population = right.population[amount:] + a
        events.append({"left_island":i, "right_island":right_index, "count_each_direction":amount})
    return events


class IGTSPRing:
    def __init__(self, population_size=150, generations=100, seed=0, *, island_count=10,
                 groups=17, strategy_count=34, take_profit_bits=8, stop_loss_bits=8, weight_bits=8,
                 take_profit_limit=.1, stop_loss_limit=.1, migration_interval=10, migration_rate=.5,
                 migrants=None, island_workers=1, crossover_probability=.6, mutation_probability=.01,
                 inversion_probability=.1, selection_rate=.5, risk_floor=1e-12, ratio_cap=1e9,
                 bank: tuple[Strategy, ...] | None = None, execution: ExecutionConfig | None = None):
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
        self._selected: GroupChromosome | None = None
        self._bank: tuple[Strategy, ...] = ()
        self._diagnostics: dict = {}

    @staticmethod
    def _check_external(extra_features):
        if extra_features is not None and (not isinstance(extra_features, Mapping) or len(extra_features)):
            raise ValueError("IGTSP-Ring uses its training-selected strategy bank; external GP features are unsupported")

    def fit(self, bars: pd.DataFrame, extra_features=None) -> IGTSPRing:
        self._selected, self._bank, self._diagnostics = None, (), {}
        self._check_external(extra_features)
        frame = validate_bars(bars)
        market = PreparedMarket.from_bars(frame,self.execution.instrument)
        volume_missing=not np.isfinite(frame['volume'].to_numpy()).all()
        candidate_bank=tuple(s for s in self.candidate_bank if not volume_missing or
            not any(rule.startswith(('obv:','mfi14:','adosc:')) for rule in (s.buy_rule,s.sell_rule)))
        if len(candidate_bank)<self.strategy_count:
            raise ValueError("Too few strategies remain after excluding unavailable volume rules")
        pool_signals = strategy_signals(frame, candidate_bank)
        # The text omits 'best n' ranking: declare return-first and risk/index tie-breaks.
        initial = monthly_trade_profiles(market, pool_signals, replace(self.execution,
                                         take_profit=self.take_profit_limit, stop_loss=self.stop_loss_limit))
        returns = initial.wealth[:, -1]-1
        risks = np.max(np.maximum.accumulate(initial.wealth, axis=1)-initial.wealth, axis=1)
        qualified=np.flatnonzero(initial.trade_counts>=2)
        if len(qualified)<self.strategy_count:
            raise ValueError("Too few training-active strategies with two completed trades for the selected bank size")
        ranking=np.lexsort((qualified,risks[qualified],-returns[qualified]))
        selected_indices=qualified[ranking[:self.strategy_count]]
        self._bank = tuple(candidate_bank[int(i)] for i in selected_indices)
        evaluator = _Evaluator(market, pool_signals[:, selected_indices], self.execution, self.take_profit_limit,
                               self.stop_loss_limit, self.risk_floor, self.ratio_cap)
        sizes = [self.population_size//self.island_count + (i < self.population_size%self.island_count)
                 for i in range(self.island_count)]
        islands = []
        for size, seed in zip(sizes, np.random.SeedSequence(self.seed).spawn(self.island_count)):
            rng = np.random.default_rng(seed)
            population = [random_chromosome(rng, self.strategy_count, self.groups, self.take_profit_bits,
                                            self.stop_loss_bits, self.weight_bits) for _ in range(size)]
            islands.append(_Island(population, rng))
        migrations, global_history = [], []
        worker_pool = process_pool(min(self.island_workers,self.island_count), initializer=_initialize_island_profiles,
                                   initargs=(evaluator,)) if self.island_workers > 1 else None
        try:
            completed = 0
            while True:
                steps = min(self.migration_interval, self.generations-completed)
                if worker_pool:
                    futures = [worker_pool.submit(_advance_island_worker, island, steps, self._settings) for island in islands]
                    islands = [future.result() for future in futures]
                else:
                    islands = [_advance_island(island, evaluator, steps, self._settings) for island in islands]
                completed += steps
                if completed and completed % self.migration_interval == 0:
                    for event in ring_migrate(islands, self.migrants, migration_rate=self.migration_rate, evaluator=evaluator):
                        migrations.append({"generation":completed, **event})
                global_history.append({"generation":completed,
                    "best_fitness":max(island.score(c, evaluator) for island in islands for c in island.population)})
                if completed == self.generations:
                    break
        finally:
            if worker_pool:
                worker_pool.shutdown(wait=True, cancel_futures=True)
        candidates = [c for island in islands for c in island.population]
        final_metrics=evaluator.evaluate_many(candidates)
        winning_index=max(range(len(candidates)),key=lambda i:final_metrics[i]['fitness'])
        winner,metrics=candidates[winning_index],final_metrics[winning_index]
        if not metrics["eligible"]:
            self._bank = ()
            raise ValueError("IGTSP-Ring found no eligible portfolio with two completed sleeve trades")
        self._selected = winner
        evaluated = {c for island in islands for c in island.cache}
        self._diagnostics = {
            "method":"igtsp_ring", "source_doi":SOURCE_DOI, "source_pdf_sha256":SOURCE_PDF_SHA256,
            "fidelity":FIDELITY, "paper_fitness_verified":True, "fitness":"section_3.3.3_eq6_to_16",
            "training_rows":len(frame), "training_end":frame["available_at"].iloc[-1].isoformat(),
            "training_months":list(market.months), "selected_training":metrics,
            "population_semantics":"total across islands", "island_population_sizes":sizes,
            "unique_evaluations":len(evaluated), "island_evaluation_count":sum(len(i.cache) for i in islands),
            "unique_exit_policies_evaluated":len({(c.take_profit_bits,c.stop_loss_bits) for c in evaluated}),
            "strategy_pool_size":len(candidate_bank), "selected_strategy_count":self.strategy_count,
            "volume_policy":"exclude volume-dependent buy/sell conditions" if volume_missing else "observed volume used",
            "bank_selection":{"ranking":"monthly terminal return descending; absolute monthly drawdown ascending; stable pool index",
                              "training_only":True, "ranking_take_profit":self.take_profit_limit,
                              "ranking_stop_loss":self.stop_loss_limit,
                              "selected_pool_indices":[int(i) for i in selected_indices]},
            "migration_events":migrations, "global_best_history":global_history,
            "island_history":[island.history for island in islands],
        }
        return self

    def predict(self, bars: pd.DataFrame, extra_features=None) -> PortfolioSignals:
        if self._selected is None:
            raise RuntimeError("Fit IGTSP-Ring before predicting")
        self._check_external(extra_features)
        tp, sl = self._selected.thresholds(self.take_profit_limit, self.stop_loss_limit)
        return PortfolioSignals(tuple(s.name for s in self._bank), strategy_signals(bars,self._bank),
                                tuple(float(x) for x in self._selected.allocations()), 0., tp, sl)

    def evaluate_paper(self, bars: pd.DataFrame, *, start: int = 0) -> dict:
        """Describe a frozen model's paper metrics separately from common equity."""
        frame = validate_bars(bars)
        if type(start) is not int or not 0 <= start < len(frame):
            raise ValueError("Evaluation start must identify a nonempty window")
        plan = self.predict(frame)
        market = PreparedMarket.from_bars(frame.iloc[start:].reset_index(drop=True),self.execution.instrument)
        profile = monthly_trade_profiles(market, plan.signals[start:], replace(self.execution,
                                         take_profit=plan.take_profit, stop_loss=plan.stop_loss),
                                         initial_cash=self.execution.initial_cash*self._selected.allocations()
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
                "group_weights":self._selected.group_weights().tolist(),
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
