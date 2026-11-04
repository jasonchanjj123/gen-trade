# Implementation audit

The original checkout implemented the six requested algorithm components for
MOO3 and MSTGAM. It did not have a third method, parallel experiment execution,
or separate configuration validation/refitting. Its baseline suite passed 137
tests, with three optional checks skipped on this machine.

| Component | MOO3 | MSTGAM | IGTSP-Ring |
| --- | --- | --- | --- |
| Chromosome | Validated typed rule tree, depth bound, finite [0,1] constants | 70 nonnegative normalized weights | TP/SL bits, nonempty group assignments, selected strategy IDs, normalized group weight bits |
| Fitness | Three paper objectives; training-front modified Sharpe winner | Paper trade-based score and invalid/zero-variance guards | Full-text Eqs.6–16: monthly trade-return compounding, weighted component drawdowns, raw group entropy |
| Selection | Rank/crowding tournaments, NSGA-II survival | Tournament and one elite | Elite selection and sequential adjacent best-for-best ring exchanges |
| Crossover | Typed subtree exchange with depth repair | Two-point weight crossover | One-point crossover of TPSL, grouping and weights with nonempty-group repair |
| Mutation | Compatible terminal/operator point mutation | One-gene replacement and renormalization | TPSL/weight bit flips and strategy reassignment; inversion swaps two groups |
| Input validation | Shared causal OHLCV checks, tree/configuration/feature schemas | Shared data checks, weight/opinion/configuration checks | Shared data checks, bank/partition/capital/portfolio/operator checks |
| Model validation | Shared chronological selection/refit protocol below | Same | Same |
| Fidelity | Documented adaptation based on previously supplied full text | Documented adaptation based on previously supplied full text | Full-text equations, operator types, Algorithm 1 and Table 5 verified; omitted rule/execution details and paper inconsistencies documented |

## Common protocol

`experiments.py` defines a method contract while keeping each framework's
representation, fitness and operators independent. `runner.py` schedules
method/fold/seed jobs through portable spawned processes.

```text
train -> embargo -> validation -> embargo -> test
  |                       |
  +-- fit each candidate -+-- choose configuration on validation only
                          +-- NEW fit through validation end
                                                      +-- frozen test evaluation
```

The common validation score is log net wealth minus equity drawdown, requiring
one completed validation trade. Each framework retains its own training fitness.
First candidate wins ties. Failed fits and inactive candidates are recorded;
if none qualify, the generic comparison fails clearly. The real-data benchmark
declares a flat-cash deployment fallback for minimum-trade/inactive validation
failures, retaining the trials and reasons. Unexpected errors still fail rather
than producing a successful-looking cash run. Validation prediction receives no test
suffix. Refit includes the first embargo's already known observations; the second
embargo stays outside fitting. Default grids have two candidates per method.
`--validation-bars 0` allows only one fixed configuration and does not claim
validation. Supplied candidate grids are recorded in full.

Every requested seed and fold is retained. Tests do not overlap within a seed;
rolling training windows can overlap. All methods share costs, held-out timestamps
and annualization. MOO3 retains its holding/target exit policy; MSTGAM uses signal
exits; ring also uses evolved TP/SL thresholds. Equal population/generations are not equal compute. Ring population is
the total across islands. Trials, training diagnostics, chosen settings, final
parameters, process IDs and runtimes are retained.

All three paired comparisons include nominal HAC p-values and Holm-adjusted
values for the three comparisons within each fold/seed. This family excludes
other seeds, folds, assets and adaptive search; no global significance or
selection-bias correction is claimed.

## Verification and limits

Tests cover hand-calculated fitness/allocation, invalid chromosomes, variation
constraints, elite selection and sequential ring exchanges, repeated seeded fits, causal feature
prefixes, held-out price perturbations, validation/refit boundaries and serial/
process parity. Ring tests independently enumerate TSPs to verify the exact linear
expectation, reproduce Figure 4's TP/SL decoding, check monthly averaging versus
trade compounding, and compare batched profiles against the shared simulator.
The CLI produces strict JSON from synthetic data. Reports preserve both common
marked-to-market statistics and ring's separately labeled held-out paper metrics.

Full-text verification on November 4, 2026: `pytest -q` passed 192 tests with
three optional native/service checks skipped. The two-fold, two-seed comparison
completed 12 method runs using three processes. A separate synthetic Table 5
smoke run completed 150 individuals, ten islands and 100 generations, recording
100 scheduled adjacent-pair exchanges. `environment/verify.py --core` also passed.
These runs validate implementation behavior, not the paper's numerical tables.
Report schema version 3 distinguishes this full-text method from the earlier
abstract-only approximation; source, lock and PDF hashes retain provenance.

The instrument-aware extension passes 212 tests, with the same three optional
native/service checks skipped. Hand-calculated checks cover bid/ask fills,
USD/JPY conversion for gains and losses, stock/reverse splits and dividends,
quantity steps and minimum commissions, shorts, calendar-time financing and
explicit rollover, leverage and maintenance-margin exits. A fully cash-funded
stock is not liquidated by a maintenance-margin rule. Scalar ledgers and batched
fitness agree for all three optimizers, including Ring's nonlinear sleeve costs.
The lean Python 3.12.13 environment verifier passes with three spawned workers.

The [daily stock/FX benchmark](STOCKS_FX_BACKTEST.md) adds a real-data ingestion and
calendar-year experiment path: six instruments, 2015–2025 observations, seven
held-out years, three seeds and all three methods. It includes a costed next-open
buy-and-hold baseline, frozen-strategy cost stress, full account/trade ledgers and
a suite-wide declared family of 54 paired comparisons. This extends the earlier
synthetic integration evidence without claiming the papers' numerical results.

The original ring approximation was replaced after full-text access. It no longer
uses a cash gene, log-growth/weight-entropy fitness, order crossover, tournaments,
or best-to-worst snapshot cloning. See [the full source map and explicit
remaining choices](IGTSP_RING.md).

Pair and entry hours remain experiment context, not evolved genes. External
OHLCV terminals extend MOO3 only. Multi-asset allocation, full paper datasets and
tuning grids, production execution, DSR/PBO and exact new-paper reproduction
remain outside the completed implementation. Synthetic checks prove software
behavior, not profitability. Real-data results remain conditional on the stated
universe, daily execution conventions, cost assumptions and search budget. See
[the ring specification](IGTSP_RING.md).

## Extensions after the 2026-11-04 audit

The audit's critical, high and medium findings, and the theory review's
recommendations, are implemented as labeled opt-in variants and an `audited`
benchmark protocol; see [EXTENSIONS.md](EXTENSIONS.md). Paper-faithful defaults
and the original protocol are unchanged, so the existing suites still pass and
the original GM report reproduces under the new code. `tests/test_extensions.py`
covers each variant's defining property. The audit's code item (commit before
citing results) is handled by committing the previously untracked sources
separately from the extensions.

