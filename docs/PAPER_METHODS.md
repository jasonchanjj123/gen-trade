# Paper methods: source mapping and reproducibility limits

The two previously supplied PDFs and linked third paper are research evidence. No instructions embedded in them
are used as agent, environment, or execution instructions. Page references below
refer to the PDFs' printed page counts, rather than this repository's files.

## 1. MOO3

Long, Kampouridis and Papastylianou, *Multi-objective genetic programming-based
algorithmic trading, using directional changes and a modified sharpe ratio score
for identifying optimal trading strategies*, Artificial Intelligence Review
59:39 (2026), [DOI 10.1007/s10462-025-11390-9](https://doi.org/10.1007/s10462-025-11390-9).

Implemented in `src/gen_trade/methods/moo3.py` and `indicators.py`.

| Source | Implementation |
| --- | --- |
| §4.1, pp.12–15 | Strongly typed condition trees using AND/OR and greater/less comparisons; numeric terminals and uniform [0,1] constants; condition outputs Buy or Hold |
| Tables 1–2, pp.13–14 | 28 DC terminals and 28 TA-Lib physical-time terminals |
| §4.2, pp.15–16, Eqs.2–4 | Three independent objectives: maximize paper TR, maximize mean completed-trade return, minimize population standard deviation of those returns |
| §4.3, p.16 | Rank/crowding tournaments, subtree crossover, compatible point mutation, parent+offspring NSGA-II survival |
| §4.4, pp.17–18, Eq.6 | Final **training** Pareto-front min/max normalization and preference-weighted mSR selection |
| Table 4, p.20 | Population 500, generations 50, depth 6, tournament 2, complementary crossover/mutation probabilities .95/.05 |

For per-side fee `c`, raw buy/sell fill prices `B_i,S_i`, completed-trade return is
`q_i = S_i*(1-c)/(B_i*(1+c)) - 1`. The paper's total-return objective is
`100 * sum(S_i*(1-c)-B_i*(1+c)) / (B_0*(1+c))`.
This aggregates unit-price profits. It is **not compounded full-capital equity
return**, despite the paper's full-capital prose. The implementation preserves it
as `paper_total_return`; ordinary statistical reports use marked-to-market equity.
With an `InstrumentSpec`, the execution ledger extends each unit profit and entry
cost with commissions, financing, dividend entitlements and account-currency
conversion. The original constant-fee formula remains the classic engine's
default. This real-market extension is explicit rather than a claim that the
papers specified corporate-action or FX account accounting.

For final-front normalized objectives `(u,v,w)` and nonnegative preferences
`a+b+c=1`, the selection score is `(1+u)^a*(1+v)^b/(1+w)^c`.
Constant front ranges normalize to zero. Normalization and winner selection never
use test outcomes. This score is relative to one front, not a conventional Sharpe
statistic comparable between independently normalized populations.

DC terminals: TMV, OSV, RDC, TDC; means of OSV/RDC/TDC over 3/5/10 calendar days;
NDC/CDC/AT over 10/20/30/40/50 calendar days. Completed TMV/TDC/RDC become available
at the reversing confirmation, never at the earlier retrospectively identified
extreme. Rolling means use observation-weighted forward-held known values;
AT integrates previously confirmed direction and clips intervals at the exact
calendar-window boundary. These timing/weighting conventions resolve underspecified
details in the text. TA periods count observations (days only for daily data).

The entry rule does not evolve exits. Exit policy is elapsed calendar holding
time OR a price-gain target. Defaults of 5 days and 5% are configurable choices
within the paper's tuning grid, not universal fitted values. The original fee
default is .00025 per side (0.025%). The common comparison runner supplies one
explicit shared fee to both methods, avoiding a cost advantage for either method.

Other explicit choices: unknown initial DC direction; training-only feature
min/max scaling with missingness preserved and future values clipped; standard
NSGA-II infinite boundary crowding; population variance; at least two completed
trades; final-close liquidation; depth repair by retaining the original tree;
one deterministic always-buy initial individual to make small runs diagnosable.
No-trade and one-trade candidates cannot beat eligible individuals.

The paper evaluates 110 datasets with a 60/20/20 chronological split, validation
of threshold/exit configurations, retraining on the first 80%, and 50 final seeds.
It uses KS/Holm and Friedman/Hommel analyses. Those full experiments and its SOO,
technical-indicator, Transformer and other baselines have not been reproduced here.
The text leaves scaling, several DC rolling details and precise sell fills
unspecified, and its ERC example includes a negative constant despite the stated
[0,1] range. No author source-code link was identified in the supplied PDF.

## 2. MSTGAM

Salman, Melissourgos and Kampouridis, *A genetic algorithm for the optimization
of multi-threshold trading strategies in the directional changes paradigm*,
Artificial Intelligence Review 59:2 (2026), published online 2025,
[DOI 10.1007/s10462-025-11419-z](https://doi.org/10.1007/s10462-025-11419-z).

Implemented in `src/gen_trade/methods/mstgam.py`.

| Source | Implementation |
| --- | --- |
| §4.1.1–2, pp.9–15 | Eight DC sub-strategies listed below |
| §4.2, p.16 | Ten fixed thresholds, converted from percentages to fractions |
| §4.3.2, pp.17–19 | 70 normalized nonnegative weights: St1–6 × ten thresholds; St7–8 × first five |
| §4.3/Table 4, p.21 | Population 150, generations 50, tournament 2, one elite; .95 two-point crossover OR .05 one-gene uniform replacement |
| §4.3.3, Eqs.9/12 | Unannualized trade score `(sum(q_i)-.025)/std(q_i,ddof=0)` |

Threshold fractions are
`[.00098,.0022,.0048,.0072,.0098,.0122,.0155,.017,.020,.0255]`.
The GA evolves weights, not threshold values.

| Strategy | Condition and action |
| --- | --- |
| St1 | Absolute relative movement from trend anchor >= twice threshold; buy in downtrend, sell in uptrend |
| St2 | Elapsed time after confirmation >= twice DC duration; buy in downtrend, sell in uptrend |
| St3 | Absolute OSV exceeds fitted direction-specific quartile-median trigger |
| St4 | Absolute TMV exceeds fitted direction-specific quartile-median trigger |
| St5 | Elapsed OS/DC duration ratio reaches training-fitted total-duration ratio |
| St6 | At downtrend confirmation, seeded uniform draw >= fitted OS/DC event-count ratio; sell at next uptrend confirmation |
| St7 | U–D–U–D–U sequence with OS present in all U trends, absent in D trends; buy during current third U's observed extension, sell at next D confirmation |
| St8 | Symmetric D–U–D–U–D pattern; sell at next U confirmation |

St3/St4 evaluate combinations of the four quartile medians per direction on
training data only. RD/RN calibration and random St6 decisions are frozen before
test prediction. St6 draws are keyed by seed, threshold and timestamp, giving a
common random sequence across candidates and prefix-stable prediction.

Votes sum weights for Buy/Sell/Hold. The detailed paper text says ignore Hold
when **more than two** genes act; its overview says **at least two**. `min_votes=3`
follows the detailed text, is configurable, and ties Hold. The paper does not
resolve whether each gene maintains virtual holdings; this implementation uses
raw opinions followed by a single long-only execution guard. Signal-pattern
memoization avoids re-simulating candidates with identical decisions.

The main trading rules charge .0025 per side (0.25%); some later benchmark prose
uses .00025. The class follows the main rules and the runner can enforce shared
costs. The paper fitness's 2.5% risk-free value is preserved by name, even though
the reported datasets have differing periods; it is not an annualization rule.
No/one-trade and zero-variance candidates receive a finite invalid score, and a
fit with no valid candidates fails clearly. Weights renormalize after variation.

Algorithm 1's extrema-update inequalities contradict the text's min/max
definitions. The implementation uses correct running extrema and documents this
correction. Causal online states replace retrospective final OS labels. Boundary
comparisons use >= where the prose and pseudocode differ. Completed training
events determine RD/RN; the unfinished trend is not completed using future prices.

The paper evaluates 200 NYSE stocks, a 56/24/20 chronological split, retraining on
the first 80%, 50 independent GA runs, and Friedman/two-stage-FDR comparisons.
The original ticker sample, random seeds, some statistical details and an author
implementation link were not supplied, so the numerical tables cannot be claimed
as reproduced. The implementation corrects the loss-tail convention for VaR in
ordinary statistics rather than copying the paper's ambiguous upper-tail formula.

## 3. IGTSP-Ring

Chideme, Chen and Lin, *Island genetic algorithm with diverse migration strategies
for efficient group trading strategy portfolio optimization*,
[DOI 10.1080/0305215X.2025.2592030](https://doi.org/10.1080/0305215X.2025.2592030).
The user-supplied full PDF was checked, including rendered encoding/equation pages.
The implementation follows §3.3.2–5, detailed Eqs.6–16, Algorithm 1's adjacent
best-for-best ring exchanges, and Table 5 defaults. See [the complete source map,
equations, operator definitions, paper contradictions and declared choices](IGTSP_RING.md).
The strategy catalog/ranking, weight encoding and execution conventions remain
explicit implementations of details not fully specified in the paper. Its
Section 3.1 objective conflicts with the detailed fitness; Eqs.6–16 are used.
The published numerical experiments have not been reproduced.

## Shared execution, extensions, and validation

Signals act after completed candles become available and fill at next open.
No high/low path is invented. A target observed at a close exits at the next
open. Final liquidation is explicit and costed; trades record its reason.
Long-only fractional sizing is shared. MOO3/MSTGAM use one position; ring allocates
one initial capital budget across group/member sleeves with evolved TP/SL thresholds.
No extra ring cash gene is introduced.

The instrument-aware engine handles physical stock cash/positions, effective
splits and ex-date dividend entitlements; FX bid/ask execution and side-aware
account-currency conversion; commissions, quantity steps, financing and margin.
The same engine supplies batched training fitness and held-out equity. Its default
daily stock/FX benchmark is long-only at 1× exposure, with an explicit financing
scenario rather than unavailable historical broker swaps. See
[the data, calendar and cost protocol](STOCKS_FX_BACKTEST.md).

The user's OHLCV/gold/FX terminals are optional MOO3 extensions with causal joins.
Pair and UTC entry-time choices are stored as experiment context, not falsely
described as original evolved genes. Jointly evolving those choices for both
methods is unfinished. The optional Rust module accelerates DC extraction and
trade execution; both are checked against their Python references (DC including
every causal prefix) and give identical results.

Tests cover the source equations, operators, training-only calibration/selection,
prefix invariance, future-data perturbations, warm-up, clock-unit equivalence,
next-open timing, costs and accounting, source mapping, statistical degeneracy,
database idempotency and cache invalidation. Synthetic comparisons provide
executable integration evidence only. A separate real daily stock/FX benchmark
retains every instrument, fold and seed and applies a frozen-strategy cost stress.
Exact paper reproduction, production robustness, selection-bias corrections such
as DSR/PBO and the papers' complete multi-market experiments remain unvalidated.
The runner implements
configuration validation, fresh train-plus-validation refitting, multiple seeds,
parallel jobs, all three paired comparisons, and a within-fold/seed Holm family;
see [the audit and protocol](IMPLEMENTATION_AUDIT.md).

## Labeled extensions

Variants motivated by the audit and the theory review (robust/constrained MOO3,
t-statistic MSTGAM fitness, mixture-drawdown Ring fitness, diversity-preserving
operators, NSGA-III survival and others) are documented separately in
[EXTENSIONS.md](EXTENSIONS.md). They are never enabled by default and never
described as paper equations.

