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
