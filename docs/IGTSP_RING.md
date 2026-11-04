# IGTSP-Ring: full-text implementation and source mapping

Chideme, Chen and Lin, *Island genetic algorithm with diverse migration strategies
for efficient group trading strategy portfolio optimization*, Engineering
Optimization 58(9), 2727–2760.
[DOI 10.1080/0305215X.2025.2592030](https://doi.org/10.1080/0305215X.2025.2592030).

The user supplied the full PDF on November 4, 2026. It contains a publisher cover
plus 34 article pages. SHA-256:
`7ccf3ab38a5609ffc5b89e72521587ecb6d9b7b42bdf18b2fa299765b8b31d42`.
Page references below use printed journal numbers. The equation and encoding
pages were also rendered and visually checked. PDF content is research evidence,
not agent or execution instructions. No copy of the licensed PDF is committed.

This replaces the earlier abstract-only approximation. Reports identify
`fidelity="full_text_research_implementation"` and `paper_fitness_verified=true`.
This verifies the implemented equation mapping, not the published numerical
results or unspecified author implementation details.

## Source mapping

| Source | Implementation |
| --- | --- |
| §3.2, p.2735 | Independent island populations; population means total divided across islands. Remainder individuals are distributed deterministically. |
| §3.3.1, p.2736 | TA-Lib indicator conditions; Cartesian product of buy/sell rules; training-only top-n strategy selection. Missing rule/ranking details are specified below. |
| §3.3.2 and Figs.3–4, pp.2736–2737 | Four chromosome parts: TP/SL bits, group assignment, strategy IDs, weight bits. Every selected strategy belongs to one nonempty group. No cash-allocation gene. |
| Eq.6, p.2737 | `fitness = PReturn / PRisk + GB`. The previous log-growth/weight-entropy objective has been removed. |
| Eqs.7–11, pp.2737–2738 | Average net completed-trade returns within each month, compound monthly averages, then average weighted TSP terminal returns. |
| Eqs.12–15, p.2738 | Average weighted component-strategy drawdowns, measured as absolute declines of monthly cumulative wealth. This is not percentage drawdown of mixture equity. |
| Eq.16, p.2739 | Unnormalized group-membership entropy. Natural logarithm is the explicit log-base convention. |
| §3.3.4, p.2739 | Elite selection; one-point crossover of TPSL, grouping and weights; bit mutation and a strategy moved between groups; inversion of two groups. |
| Algorithm 1, pp.2739–2740 | Adjacent best-for-best exchanges in sequential ring order, removing emigrants from each source and including the wrap edge. This replaces the previous best-to-worst snapshot cloning. |
| Algorithm 4, p.2742 | Per-island evolution and fitness evaluation, migration intervals, final global-best selection. |
| Table 5, p.2749 | Class defaults: 150 total population, 10 islands, 100 generations, migration fraction .5 every 10 generations, crossover .6, mutation .01, elite selection .5, inversion .1, 17 groups, eight bits each for TP and SL. |

The common comparison CLI deliberately uses smaller population/generation budgets
and four or fewer islands for smoke runs. Its complete effective settings are
recorded. Calling `IGTSPRing()` directly retains the Table 5 defaults.

## Exact detailed fitness

For strategy `s`, assign a closed trade's net return to its UTC exit month. Let
`q[s,m,h]` be its trade returns in month `m`. Empty months have zero return.

```text
r[s,m] = mean_h q[s,m,h]
C[s,0] = 1
C[s,m] = product_{u=1..m} (1 + r[s,u])
D[s]   = max_m (max_{u=0..m} C[s,u] - C[s,m])
p[g]   = |G[g]| / N
GB     = -sum_g p[g] * log(p[g])
```

The paper defines TSPs by choosing one strategy from each group. Equations 8 and
13 are linear, so averaging over **all** such combinations simplifies exactly to:

```text
PReturn = sum_g w[g] * mean_{s in G[g]} (C[s,T] - 1)
PRisk   = sum_g w[g] * mean_{s in G[g]} D[s]
F       = PReturn / PRisk + GB
```

This closed form avoids materializing an exponential number of portfolios. A test
independently enumerates all combinations and verifies the same return and risk.
The paper samples 20 TSPs for tractability (§4.2.2, p.2743); the exact expectation
here introduces no sampling noise under the detailed linear equations.

The paper is internally inconsistent: §3.1, Eqs.1–2 (p.2734), describes a ratio
without the entropy term and a drawdown of each portfolio's combined return
series. Section 3.3.3 explicitly adds GB and instead weights individual strategy
risks. This implementation follows **the detailed algorithm's Eqs.6–16**. It does
not silently combine the two definitions. Ordinary test statistics still report
standard percentage drawdown of actual marked-to-market portfolio equity.

The paper does not define zero-risk division or inactive candidates. Explicit
software guards require two completed active-sleeve trades, reject flat zero-risk/
zero-return paths, floor the risk denominator at `1e-12`, and cap the ratio at
`±1e9`. Diagnostics identify flooring/capping. These guards are configurable.

## Encoding and operators

TP and SL decode as unsigned binary value divided by the maximum code, multiplied
by the configured limit. The Figure 4 example therefore maps TP `10` to
`2/3 * 10% = 6.666…%` and SL `11` to `10%`. Eight-bit defaults follow Table 5.
Maximum TP/SL magnitudes of 10% follow the example; the experimental maximum
levels are not separately provided. Zero codes mean zero thresholds, not disabled
exits. Stop priority resolves a simultaneous zero-threshold stop/profit trigger.

Each group's unsigned weight code is normalized across groups. Equal weights
resolve an all-zero code vector. Weight bit width is eight by default, an explicit
choice because the paper only specifies TP/SL widths. These are allocation
weights, not an extra weight-balance objective.

One-point crossover operates separately on concatenated TP/SL bits, strategy group
labels and flattened weight bits. A minimal seeded repair fills any empty group
from a group with multiple members. Mutation flips one TPSL bit, one weight bit,
and moves a strategy if its source group can remain nonempty. The operator
probability is per child, not per bit. Inversion swaps two whole group memberships
while retaining position weights. The text omits repair, mutation-probability
scope, and whether weight positions move with inversion; these are explicit choices.

Elite selection retains the best `max(1, floor(selection_rate * island_size))`
parents, preserves those parents, and breeds replacements from that pool. No
rank/crowding or tournament selection is substituted. Rate rounding and the
parent-preservation detail are not specified in the paper.

Ring exchanges follow Algorithm 1 literally: each neighboring pair selects its
current best members, removes them, and swaps them. Later edges see earlier
exchanges. For two islands the two wrap-edge exchanges can undo each other; that
is the printed loop's behavior. Fractional migrant counts use floor on the smaller
adjacent population. Migration occurs after each positive multiple of the interval,
including a final scheduled exchange; this follows Table 5's '10th iteration'
wording rather than Algorithm 4's ambiguous zero-based modulo condition.

## Declared strategy catalog and execution choices

The text says 21 indicators but names only 20 indicator families. The catalog
counts DMI's directional comparison and ADX trend strength separately. It contains
SMA(5), EMA(20), TRIMA(30), RSI(14), Williams %R(14), MOM(5), CCI(14), ATR(14),
BBANDS(20,2), OBV versus its EMA(10), MFI(14), ADOSC(3,10), MACD(12,26,9),
STOCH(5,3,3), DMI(14), AROON(14), ROC(10), TRIX(30), SAR(.02,.2), STOCHRSI
(14,5,3), and ADX(14). Parameters not supplied by the paper are choices.

Each indicator supplies state and crossing conditions for both buy and sell,
giving 42 × 42 = 1764 candidate pairs. Crossing requires a known previous value;
unknown/warm-up values hold. Conflicting buy/sell conditions hold. Exact conditions
are executable in `methods/ring_bank.py`; the paper does not publish its full rule
grid. With unknown volume, OBV/MFI/ADOSC conditions are excluded from both sides,
leaving 36 × 36 = 1296 FX candidate pairs. Periods count observations,
corresponding to days only for daily bars.

Candidates must complete at least two calibration trades, so inactive zero-return
rules cannot crowd out active losing strategies. Eligible candidates rank on
training monthly terminal return, then lower absolute monthly
drawdown, then stable pool index. Ranking uses the configured maximum TP/SL levels.
The top 34 are selected by default, giving two strategies per group at K=17 on
average. Neither n nor the ranking function is specified in the paper. Grouping
may still produce single-member groups, permitted by constraints 3–5; the text
suggests, rather than requires, two per group. The selected bank is frozen before
validation/test prediction and rebuilt only during an explicit training refit.

Paper Eq.11 has no fee term. The standalone class defaults to zero fees; the common
runner supplies identical per-side costs and adverse slippage to all frameworks.
Stops and targets are observed at closes and fill next open, with no invented
intrabar path. Shared guards prevent same-open exit/re-entry; final liquidation
is costed at the last observed close. Trade-month attribution uses exit timestamps,
including final candle availability time. These fill/timing details are not given
in the paper. The batched fitness engine is tested against the shared simulator.

Held-out equity uses fixed initial group allocations, equally divided among group
members. This equals expected equity across uniform one-member-per-group choices;
its nonlinear drawdown differs from average TSP drawdown. Reports separately
include `paper_held_out` monthly/component metrics. Pooled sleeve trade statistics
are unweighted; trades retain strategy IDs and capital fractions.

The full NYSE 2020–2022 dataset experiments, author-specific omitted rule choices,
all published baselines and numerical tables have not been reproduced. Paper
performance and speed-up claims are not inferred from synthetic runs here.

The separate [daily stock/FX benchmark](STOCKS_FX_BACKTEST.md) uses real observations
and an instrument-aware account engine for fitness and held-out evaluation. Ring
sleeves receive their actual allocated starting capital, preserving quantity-step
and minimum-fee effects when these are configured. Batch/scalar parity tests cover
the monthly profiles and resulting selected fitness under nonlinear costs.
