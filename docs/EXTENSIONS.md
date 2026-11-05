# Theory and audit extensions

These changes respond to the 2026-11-04 audit and the theory review
("Theory-Based Improvements for the GA Trading Setup"). Each method change is a
**labeled, opt-in variant**. Constructor defaults stay paper-faithful, and the
original benchmark protocol is unchanged. Reports name the variant they used.

```bash
# Paper methods, original protocol (reproduces the existing report)
gen-trade benchmark --output reports/original-paper
# Paper methods, audited protocol: isolates the protocol/cost changes
gen-trade benchmark --protocol audited --output reports/audited-paper
# Extension methods, audited protocol
gen-trade benchmark --protocol audited --method-profile improved --output reports/audited-improved
```

`--method-profile improved` uses `IMPROVED_CANDIDATES` in `experiments.py`.
`--protocol audited` uses `PROTOCOLS['audited']` in `benchmark.py`. Any
option can also be set individually through `--method-configs` (compare/demo)
or the constructors.

## Method variants

| Method | Option | Change | Motivation |
| --- | --- | --- | --- |
| MOO3 | `objectives="robust"` | Eq.3 unit-price total return → compounded trade return `100(∏(1+q)−1)`; mean trade return → lower bound `mean − z·s/√n` (`lcb_z`, default 1) | Audit 1, 5: Eq.3 depends on price level and breaks across splits; two-trade rules win on mean/std ([Lo 2002](https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf)) |
| MOO3 | `min_trades_per_year` | Constrained domination: rules with fewer than `max(2, ⌈rate·years⌉)` training trades are infeasible. Among infeasible rules, a smaller shortfall dominates. mSR scores feasible front members only | Audit 1; [Deb 2000](https://repository.ias.ac.in/9407) |
| MOO3 | `survival="nsga3"` | Das–Dennis reference points, niche-preserving last-front selection | 3-objective NSGA-II failure ([Zheng and Doerr 2024](https://hit.globalimpact.cn/en/publications/runtime-analysis-for-the-nsga-ii-proving-quantifying-and-explaini/)); NSGA-III guarantee ([Wietheger and Doerr 2023](https://arxiv.org/pdf/2211.08202)) |
| MOO3 | `survival="nsga2_sequential"` | Crowding is recomputed after each removal | [Zheng and Doerr 2025](https://researchportal.ip-paris.fr/en/publications/approximation-guarantees-for-the-nondominated-sorting-genetic-alg/) |
| MOO3 | `feature_transform="scale_free"` | MA/EMA/SAR → `x/close−1`, ATR → `ATR/close`, OBV → 20-bar change ÷ 20-bar volume. Families with shared units (price ratios, ATR ratios, RSI, %R, CCI, OSV, TDC, RDC) share min/max bounds | Audit 2: per-feature min/max on price levels saturates out of sample, and two independently scaled MAs are not economically comparable |
| MOO3 | `threshold_volatility_multiple` | DC threshold = k × training daily log-return std (training rows only) | Audit 7: 0.1% and 0.5% thresholds confirm a reversal on 36–47% of stock days |
| MSTGAM | `fitness="tstat"` | `(Σq − Rf·years)/(s·√n)`: the t-statistic of training trades over an annual risk-free rate prorated over the training span | Audit 10: `Σq/σ` grows like √n × t, and the 2.5% Rf is compared against a multi-year sum |
| MSTGAM | `sign_adjusted=True` | Negative numerator: multiply by the denominator instead of dividing | [Israelsen 2005](https://link.springer.com/article/10.1057/palgrave.jam.2240158) |
| MSTGAM | `mutation="lognormal"` | One gene `w ← w·exp(σαZ)`, with α power-law on 1–8 (β = 1.5) | The paper's uniform redraw hands the mutated gene about 1 − ln 2 ≈ 31% of the vote; heavy-tailed steps follow [Doerr et al. 2017](https://arxiv.org/html/1703.03334) |
| Ring | `fitness="mixture_drawdown"` | Return and absolute drawdown of the capital-weighted mixture of member wealth paths, plus entropy of group capital weights | Audit 4: Eq.6 is linear-fractional in the weights, so its optimum is one group (mediant inequality); membership entropy ignores capital |
| Ring | `sign_adjusted=True` | Israelsen refinement of the ratio | As above |
| Ring | `mutation_mode="bitwise"` | Every child mutates: TP/SL and weight bits flip at rate 1/length; strategies move at rate 1/n | Audit 9: the paper operator yields about six mutations per small fit |
| Ring | `unique_children=True` | Children are resampled up to 20 times until new to the island; a final bitwise mutation is the fallback | 69% of child slots repeated an evaluated chromosome in the original benchmark; crossover speed-ups need lasting diversity ([Doerr et al. 2023](https://arxiv.org/abs/2302.12570), [Opris and Antipov 2026](https://arxiv.org/abs/2604.04083)) |
| Ring | `crossover_mode="group"` | Injects a random proper subset of the second parent's whole groups (members and weight bits) | Label encodings are m!-redundant and n-point crossover passes groups out of context ([Lewis and Pullin](https://www.rhydLewis.eu/papers/rgfgaEvolPaper.pdf)); [Falkenauer 1994](https://www.orbel.be/jorbel/index.php/jorbel/article/view/156) |
| Ring | `bank_selection="diverse"` | Greedy facility location on `(1+corr)/2` of monthly returns over the top 4n by return/drawdown with positive return; monotone submodular, so greedy is within 1 − 1/e (Nemhauser, Wolsey and Fisher 1978, cited from memory) | Audit 11: picking the top 34 of 1,764 by return is heavy in-sample selection of correlated rules |
| Ring | `bank_ranking="two_stage"` | Half the generations on the paper-ranked bank; then re-rank under the stage-1 winner's TP/SL and evolve the rest. Half of stage-2 seeds inherit that TP/SL | Audit 11: the bank is ranked at TP = SL = 10%, but the evolved SL median is 1–4% |
| Ring | `migration_schedule="adaptive"` | Interval starts at 1, halves after an improving epoch and doubles otherwise (scheme B, applied globally). No exchange after the last generation | [Mambrini and Sudholt 2015](https://eprints.whiterose.ac.uk/93025/1/ECJ-submission-R1.pdf); the paper schedule's final exchange cannot affect the result |

Under the improved profile, each method keeps its paper validation knob (MOO3
volatility multiple 1 or 2, MSTGAM `min_votes` 3 or 2, Ring 12 or 17 groups)
and switches every variant on. Stochastic sub-window fitness and lexicase
selection, discussed in the review, are not implemented.

## Audited protocol

| Item | Original | Audited | Audit |
| --- | --- | --- | --- |
| Configuration choice | Per seed | One choice per method and fold, from seed-mean validation scores; a candidate needs scores from a strict majority of seeds | Seeds agreed on the config no more often than chance for Ring (31%) and MOO3 (28%) |
| Deployment | New, unvalidated refit through validation end | The validated training-only model, re-created deterministically | 8 |
| Validation activity | ≥1 trade | ≥2 trades | 8 |
| Validation curves | — | Best individual every 5 generations scored on validation | Shows whether more budget helps or overfits |
| FX financing | 2%/yr debit long and short (4% stress) | 0 primary; 2% debit as stress | 6: a symmetric debit ignores carry sign; long USD/JPY earned positive carry |
| Stock spread | 2 bp | max(2 bp, $0.01) full spread ($0.02 stress) | 13: 2 bp is below one tick for low-priced shares such as pre-2020 GRPN |
| FX Sunday bars | Kept as full periods | Merged causally into the next session; annualization recomputed | 13 |
| Code hash | Python + lock | Adds `native/src` and Cargo files | 13 |

## Reporting additions

These are always computed in the summary. Per-fold exposure controls require the
audited protocol (or `exposure_controls_enabled=True`).

- **Exposure controls** (audit 3): capital-weighted time in market; HAC
  alpha/beta on next-open buy-and-hold; paired tests against buy-and-hold and
  against a static cash/buy-and-hold blend with the same mean exposure. Single-
  position methods also get the rank of their net return among 200 random
  placements of the same holding durations, with the same costs.
- **Equal-capital seed ensemble**: each test year splits capital equally across
  the three seeds' frozen models (exact for the scale-invariant cost model).
- **Deflated Sharpe ratio** ([Bailey and López de Prado 2014](https://papers.ssrn.com/abstract=2460551)):
  trials are each instrument's method × seed paths; the Sharpe variance is
  measured across them.
- **PBO by CSCV** ([Bailey et al. 2017](https://escholarship.org/content/qt4w1110bb/qt4w1110bb.pdf)):
  8 contiguous blocks of the stitched held-out returns, 70 splits.
- **Hansen SPA** ([Hansen 2005](https://scispace.com/pdf/a-test-for-superior-predictive-ability-1fbgz05jht.pdf),
  `arch.bootstrap.SPA`): H0 is that no method beats buy-and-hold, tested per
  seed and on the ensembles.

None of these corrects the genetic search inside a fit. They address
selection among the reported paths and confounding by market exposure.

## Verification

`tests/test_extensions.py` checks the defining property of each variant: the
constrained-domination order, sequential versus one-shot crowding, NSGA-III niche
filling, compounded and lower-bound objectives, scale invariance of the
transformed features, training-only volatility thresholds, t-statistic and
sign-adjusted scores, the mediant bound and hedging benefit for Ring fitness,
group-crossover transmission, facility-location coverage, duplicate-free
evolution, pooled majority selection, training-only validated deployment,
exposure/alpha/blend identities, random-timing duration replay, PSR/DSR/PBO/SPA
behavior, the tick floor, the cost scenarios and the causal Sunday merge.
Paper-default suites are unchanged.

## Review 2026-11-05 extensions

These follow `claude/review-2026-11-05.md` in the gen-trader project. The
paper and improved profiles are unchanged: small fits of every method under
both profiles reproduce the previous code exactly. Everything below is opt-in.

### Tests on the quantity that measures timing

| Change | Where | Why |
| --- | --- | --- |
| SPA on excess over the exposure-matched blend and on beta-hedged returns (zero benchmark), per seed and for the ensembles | `statistics.spa_excess`, `benchmark.excess_tests` | SPA against buy-and-hold rejects a 10%-exposure strategy in any rising market regardless of timing |
| Romano–Wolf StepM on blend excess: which methods are positive at 5% familywise error | `statistics.stepm_excess` | SPA only says whether any method beats the null |
| Rademacher Anti-Serum over every method-seed path's blend excess | `statistics.rademacher_antiserum` | Uniform lower bound on true Sharpe for all tested paths ([Paleologo 2024](https://bsic.it/wp-content/uploads/2024/10/backtesting-episod-2.pdf); constants from secondary summaries) |
| Random timing for Ring: each active sleeve replays its own durations at random times with its allocation; idle capital stays in cash | `runner.random_timing` | Ring had no chance-timing control; for one full sleeve it reduces exactly to the single-position test |
| Pooled timing evidence: instrument × year cells, seeds averaged, with a naive t and a year-clustered t | `benchmark.pooled_controls` | Per-instrument tests have little power (IR 0.3: 12% on 7 years) |
| `gen-trade reanalyze SRC --output DST [--data-directory D]` recomputes a finished benchmark's summary with the current statistics; with data it adds Ring random timing post hoc after checking that the rebuilt test windows reproduce the stored buy-and-hold equity | `benchmark.reanalyze` | Corrected tests on existing reports without refitting |

### Fitness that targets the tested quantity

`methods/robust_fitness.py`: each candidate is scored by the annualized
information ratio of its daily net return in excess of a static cash/buy-and-
hold blend with its own mean exposure, `d_t = r_t − e·b_t`. Static exposure
earns exactly zero, so the score rewards only being invested on better days.
Each generation scores a fresh random half of the training months (stochastic
sub-window fitness; [Gonçalves and Silva 2012](https://gpbib.cs.ucl.ac.uk/gp-html/goncalves_2012_EuroGP.html),
[Geiger et al. 2025](https://arxiv.org/abs/2502.18093v1)); elites are re-scored
on every sample and the final pick uses the full window. Each fit also reports
a Rademacher haircut over every evaluated candidate's training excess
(`selection_haircut`), so the search's own data snooping is visible per fold.

| Method | Option | Change |
| --- | --- | --- |
| MOO3 | `objectives="blend_excess"`, `complexity_penalty` | Scalar score = sub-window excess IR minus a per-node penalty; all three objective slots carry it, so Pareto rank is score order, survival is NSGA-II truncation and the pick is the best full-window score |
| MOO3 | `objectives="single"` | Scalar baseline: lower bound of annualized log growth, (Σ log(1+q) − z·√n·s)/years, under the same trade floor (s is the trade-return std) |
| MSTGAM | `fitness="blend_excess"`, `min_trades_per_year` | Weight search on sub-window excess IR; DC threshold calibration keeps the t-statistic; deployed weights are the final population's best on the full window |
| Ring | `fitness="blend_excess"` | Excess IR of the capital mixture's marked-to-market daily return |
| All | `subwindow_fraction` (default 0.5) | Share of training months per generation |

### Ring mixture and weights

| Option | Change | Why |
| --- | --- | --- |
| `fitness="mixture_mtm"` | Mixture return/drawdown on marked-to-market daily sleeve equity | Eq.9 averages trades closing in a month (three sequential +2% trades book +2%, not +6.1%) and never marks open positions |
| `weight_mode="equal"` | Evolve memberships and TP/SL with equal group weights | 1/N baseline |
| `weight_mode="lp"` | Evolve with equal weights, then solve group weights exactly: max (M_T − 1)/maxDD(M) over the simplex by Dinkelbach iteration on an LP (running-peak drawdown formulation) | With memberships fixed the weight problem is convex-fractional; the GA's weights were near random |

### Protocol, profile and lockbox

- `--protocol audited2`: the audited protocol, with configurations chosen on
  the validation excess IR instead of log wealth minus drawdown.
- `--method-profile robust`: blend-excess fitness for all three methods (MOO3
  complexity penalty 0.01 per node, trade floors of 3 per year), Ring equal
  weights. Validation knobs are unchanged.
- `--method-configs FILE`: any candidate set as profile `custom`, for example
  to compare Ring weight modes.
- `gen-trade freeze-profiles --first-test-year 2026 [--symbols ...]` writes
  `lockbox.json`. The benchmark then refuses lockbox symbols and test years
  unless `--unlock-lockbox` is given with a profile whose candidates still hash
  to the frozen value; the unlock is recorded in the report settings.

```bash
# Corrected tests on the existing reports (no refitting)
gen-trade reanalyze reports/audited-improved-v2 --output reports/audited-improved-v2-reanalysis \
  --data-directory data/stocks-fx-2015-2025
# New profile and protocol
gen-trade benchmark --protocol audited2 --method-profile robust --output reports/audited2-robust
# Isolate the protocol change
gen-trade benchmark --protocol audited2 --method-profile improved --output reports/audited2-improved
```

`tests/test_review_extensions.py` checks each defining property: equity and
exposure recording identical to full records, SPA/StepM/RAS behavior, the
single-sleeve reduction and idle-cash rule of portfolio random timing,
deterministic month sub-windows, zero excess for static exposure, MSTGAM
determinism and trade floor, MOO3 scalar objectives and the complexity
penalty, daily profiles equal to compounded trades, LP weights at the global
ratio maximum, Ring variants deploying their weights, the validation metric
and protocol header, lockbox refusal and unlock, stitched excess identities
and year-clustered pooling.

## Results 2026-11-05 extensions: turnover

The audited2 runs (`claude/results-audited2-2026-11-05.md` in the gen-trader
project) split each method's shortfall against the exposure-matched blend into
timing (method versus random placements of the same trades) and the structure
and cost of trading (random placements versus the blend). Timing was at chance;
the trading component was −0.7% to −5.3% a year and highly significant. These
changes test fewer, slower decisions directly. All defaults keep earlier
behavior: small fits under the paper, improved and robust profiles reproduce
the previous commit exactly.

| Option | Where | Change |
| --- | --- | --- |
| `decision_cadence="weekly"\|"monthly"` | all three methods (`methods/cadence.py`) | Act only on the first bar of each calendar week or month (UTC open time), on the most recent nonzero signal raised since the previous decision; other bars emit no action. Applied in training, calibration and prediction; causal (uses bars up to the decision bar only). Profit targets, stops and holding deadlines still run daily |
| `vote_margin` | MSTGAM | A buy or sell needs a weighted lead of at least this share of the vote over the other side; otherwise hold. 0 keeps the paper rule. Replaces `min_votes` as the validation knob: min_votes 3 and 2 gave identical weights in 111–115 of 126 fits |
| Ring selection haircut | `IGTSPRing` (blend_excess) | Now over every chromosome evaluated in the final stage (exit-policy profiles computed in blocks), not only the final population; `unique_evaluations` counts all of them under stochastic fitness |

Profiles (validation knobs in brackets):

- `robust_weekly`: robust with weekly decisions; trade floors 3 a year; MSTGAM
  [vote_margin 0.1, 0.25]; MOO3 [volatility multiple 1, 2]; Ring [12, 17 groups].
- `robust_monthly`: the same with monthly decisions and trade floors of 1 a year.

```bash
gen-trade benchmark --protocol audited2 --method-profile robust_weekly --output reports/audited2-robust-weekly
gen-trade benchmark --protocol audited2 --method-profile robust_monthly --output reports/audited2-robust-monthly
```

Random timing is cadence-aligned for weekly and monthly methods (follow-up
commit): random entries happen only after the method's decision bars, trades
that exited after a decision bar keep their length in decision periods, and the
others keep their length in bars (`statistics.random_aligned_signals`).
Finished reports can be corrected with `gen-trade reanalyze SRC --output DST
--data-directory D --aligned-timing`. Ring sleeves with more trades than free
decision points get no control (undefined).

Known limits: the Ring haircut covers the GA search inside the final 34-rule bank, not the bank's
selection from the 1,764-rule pool; monthly decisions leave few validation-year
trades, so expect more configurations to fail the two-trade validation rule.
`lockbox.json` was re-frozen to include both profiles (lockbox untouched).

## Lockbox universe and pre-registered hypotheses

Every profile so far was tuned and compared on the same six instruments and
test years, so the monthly MSTGAM result (+5.3%/yr over the blend, t 2.8,
concentrated in GRPN and GM) is a hypothesis, not a finding. `lockbox.json`
now holds out a second universe, fixed by rule before any of its data was
fetched (`market_data.LOCKBOX_*`):

- Stocks: the 30 DJIA constituents in effect on 2015-01-02, minus eight with a
  spin-off, an entity-replacing merger or a ticker change in 2015–2025 (MMM,
  T, DD, GE, IBM, MRK, PFE, UTX), because the pipeline models only splits and
  cash dividends: 22 stocks.
- FX: the remaining G10 currencies against USD (AUD/USD, NZD/USD, USD/CAD,
  USD/CHF, USD/NOK, USD/SEK); crosses are excluded because the engine needs
  USD as base or quote.

Two hypotheses are declared in `docs/preregistered-hypotheses.json` and
frozen into `lockbox.json` with a hash; an unlock refuses edited hypotheses,
a changed profile or different run settings:

| ID | Run | Test | Level |
| --- | --- | --- | --- |
| H1 | `robust_monthly`, audited2 | MSTGAM's pooled excess over the exposure-matched blend > 0; one-sided t over yearly cross-instrument means | 0.025 (Bonferroni, 2) |
| H2 | `robust`, audited2 | Ring's selected training excess IR is positively rank-correlated with its test-year excess over instrument-year cells | 0.025 |

An unlocked run writes the decisions to `summary.json` (`preregistered_tests`)
and to a "Pre-registered lockbox tests" section of `report.md`, with the median
cell and leave-one-instrument-out t as non-decisive diagnostics.

```bash
gen-trade fetch-data --universe lockbox --directory data/lockbox-2015-2025
gen-trade benchmark --data-directory data/lockbox-2015-2025 --protocol audited2 \
  --method-profile robust_monthly --output reports/lockbox-robust-monthly --unlock-lockbox
gen-trade benchmark --data-directory data/lockbox-2015-2025 --protocol audited2 \
  --method-profile robust --output reports/lockbox-robust --unlock-lockbox
```

Run each once with the declared settings (population 64, 20 generations,
seeds 11/22/33, test years 2019–2025); 28 instruments take roughly five times
as long as the six-instrument runs. Fetching runs only the data-quality checks
(sessions, crossed quotes); look at no returns before the two runs finish.
