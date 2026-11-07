# Daily stock and FX benchmark

The user-selected universe is AAPL, GM, GRPN, EUR/USD, GBP/USD and USD/JPY, with
historical observations from 2015-01-01 through 2025-12-31. These are six
independent instrument experiments. The benchmark does not allocate a common
portfolio across the six instruments.

## Reproduce

```bash
bash scripts/setup-environment.sh --market-data
source .venv/bin/activate
gen-trade fetch-data --directory data/stocks-fx-2015-2025
gen-trade benchmark --population 64 --generations 20 --seeds 11 22 33 \
  --workers 3 --bootstrap-reps 500 --output reports/stocks-fx-2015-2025
python scripts/verify-benchmark.py reports/stocks-fx-2015-2025
```

`--protocol audited` switches to pooled seed selection, deployment of the
validated model, at least two validation trades, exposure controls, zero-carry
primary FX financing (2% debit stress), a one-cent stock spread floor and merged
FX Sunday stubs. `--method-profile improved` enables the labeled method variants.
Both are documented in [EXTENSIONS.md](EXTENSIONS.md). The commands above keep the
original protocol.

The `market-data` extra adds exchange calendars and plotting to the core
application. Downloads use Python's standard HTTPS client and verified TLS.
Neither exchange credentials nor live orders are involved. Raw data and generated
reports are ignored by Git. No historical prices are invented or interpolated.

The benchmark resumes completed instruments only when their code, lockfile, data
and experiment-setting fingerprints agree. A mismatched checkpoint fails clearly.
Use a new output directory for a changed experiment. `--no-resume` explicitly
recomputes an existing output. Per-asset JSON is written atomically.

## Historical data and availability

Equity prices and dividend/split events come from Yahoo Finance chart responses.
[Yahoo's adjustment definition](https://help.yahoo.com/kb/SLN28256.html) distinguishes
split/dividend adjustments. The adapter undoes the provider's retrospective split
adjustment to recover contemporaneous execution prices and volumes. Later split
events are used only to undo that vendor representation; they are not indicator
features. A regression test verifies that an appended future split revision
reconstructs identical earlier observations.

Indicator OHLC uses raw execution prices times cumulative **already effective**
split factors, creating a causal split-neutral series. Dividends remain separate
cash events; adjusted close is not used to inject future dividend adjustments.
Positions change quantity and basis on the effective split date. Long positions
receive and shorts owe dividends when held before the ex-date opening. Entitlements
are credited on the ex-date, with no withholding tax or later payment-date delay.

NYSE regular-session openings and closes, including early closes and daylight
saving changes, come from
[exchange_calendars](https://github.com/gerrymanoim/exchange_calendars).
All three equities have 2,766 observations and no missing expected NYSE sessions
in the requested interval. Provider revisions and delisted-stock universes are
not reconstructed. This fixed universe is not a survivorship-free sample.

FX uses Dukascopy's historical bid and ask daily candles, as described in its
[historical data export](https://www.dukascopy.com/api/data/get/historical-data-export).
The adapter reads the public yearly compressed files, retaining both price sides
and raw file hashes. Indicator OHLC is the observed bid series. Executions buy on
ask and sell on bid. Dukascopy feed activity is retained separately but is not
presented as consolidated spot-FX volume; volume-dependent buy/sell conditions are
excluded from the ring's training strategy pool when volume is unknown.

The feed includes observed partial Sunday sessions: EUR/USD and GBP/USD each have
3,439 retained observations and USD/JPY has 3,440. Inactive zero/flat placeholders
are excluded. One crossed EUR/USD quote candle (2019-03-10) is quarantined with its
date recorded in the manifest; the original response remains cached. No spread
is repaired or replaced with an invented quote. Daily FX timestamps are the
provider's UTC candle-bucket labels; opening prices are the bucket's first quoted
price, not a guarantee of an executable quote at midnight on a partial/holiday
session. Finer execution-time or intrabar claims require finer data.

Two retained Dukascopy candles are partial feed records: EUR/USD 2016-06-08 and
USD/JPY 2017-07-12, each with about 1% or less of normal feed activity. Both lie
in training years only. The data-quality audit flags them, and the ECB reference
fix confirms the EUR/USD one. They are kept so that the dataset matches earlier
reports. See [data sources and quality](DATA_QUALITY.md).

Every dataset includes timezone-aware bar start and full-candle availability time.
Signals see only completed observations and execute at the following bar's open.
FX daily availability is conservatively set to the next UTC day. Stops and targets
are observed at closes, then fill the next open. No favorable intrabar ordering is
assumed. Final liquidation uses the last observed closing quote and charges costs.

## Calendar protocol and research budget

For each test year from 2019 through 2025:

1. Fit candidates on the preceding three training years.
2. Leave five observations between training and validation.
3. Select configurations on the following validation year only.
4. Refit through the end of validation, leaving five observations before testing.
5. Evaluate the untouched complete test year, starting flat.

For example, the 2019 test trains on 2015–2017 and validates on 2018. The 2025 test
trains on 2021–2023 and validates on 2024. Embargo observations are removed from the
ends of training and validation. Refit may use the first gap's already known
observations; it never uses the test gap or test observations for fitting.

Calendar splits align stock and FX test years despite the extra observed FX Sunday
bars. All methods use identical per-instrument timestamps and declared cost models.
Annualization uses observed bars per calendar year over the supplied history and
is recorded per instrument; 252 is not imposed on a feed containing Sunday bars.

The declared first historical benchmark uses population 64, 20 generations and
seeds 11, 22, 33 for all three frameworks. It evaluates two configurations per
method and then performs a fresh refit. Across six instruments and seven test
years, this produces 378 method runs. This is a research search budget, not a
replication of each paper's complete population, tuning grids or 50-seed studies.

Validation score is log net wealth minus maximum equity drawdown, requiring one
closed validation trade. If every configuration is inactive or fails the declared
minimum-trade eligibility rules, that method deploys cash for the test year. Its
trials and reasons remain in the report. Folds and seeds are never dropped or
retuned using test performance. Unexpected input/implementation errors still fail.

## Account and cost model

`InstrumentSpec` declares asset class, account/base/quote currencies, quantity step,
leverage, short permission, commissions, modeled spread and financing. The same
prepared execution engine is used in training fitness, validation, test evaluation
and buy-and-hold baselines. Batched execution avoids changing semantics to speed up
fitness evaluations.

The benchmark starts each fold at USD 100,000, uses long-only positions at 1×
exposure and continuous fractional quantities. Idle cash earns zero. These
settings retain the papers' long-only comparison while adding market accounting.
The engine also supports entry quantity steps, minimum commissions, short positions,
borrow/debit interest, leverage and maintenance-margin exits. Entry-step rounding
does not discard split-generated fractional share entitlements.

| Assumption | Primary scenario | Frozen-strategy stress |
| --- | --- | --- |
| Stock full spread | Modeled 2 bps | 4 bps |
| Stock adverse slippage | 1 bp per side | 2 bps per side |
| Stock commission | USD .005/share, no minimum | USD .01/share |
| FX spread | Historical bid/ask | Same historical bid/ask |
| FX adverse slippage | .1 bp per side | .2 bp per side |
| FX commission | .35 bp of notional per side | .7 bp per side |
| FX financing | 2% annualized debit on held notional | 4% annualized debit |

These are explicit cost scenarios, not claims about a broker's current or
historical fee schedule. Financing integrates elapsed calendar time, including
weekends. If supplied, per-bar long/short rollover amounts in quote currency
replace assumed annual financing. The benchmark has no historical broker swap
series. Currency P&L is converted to account USD at the appropriate side; USD/JPY
uses inverse contemporaneous bid/ask conversion, with different sides for gains
and losses. Crosses whose quote currency is not the account or base currency
require explicit contemporary conversion quotes.

A cash-stock position has no maintenance-margin exit at 1× exposure. Leveraged,
short and FX contract positions can trigger margin exits, filled next open with
no same-open re-entry. Commissions, financing, dividends, quantity, realized
account P&L, currency, and exit reasons are retained in each trade ledger.

Ring sleeves receive their actual allocated initial capital. They are not each
funded with the entire account and then rescaled when minimum fees or quantity
steps are nonlinear. Fitness profiles account for sleeve sizing in that case.
Scale-invariant percentage/per-unit cost models can share equivalent profiles.

## Reports and interpretation

The first complete run finished on November 4, 2026 with software version 0.4.0
and source/lock fingerprint
`d7a958f19ddccbb455e2d0186aca2f6aa4d9064403d601620a071a4a20817396`.
All 378 declared method/year/seed cases completed. Forty-three MOO3 cases used
the declared cash fallback; no MSTGAM or Ring case did. These cash cases are
included in the outcomes below.

Median cumulative net returns across seeds, chaining the seven held-out years
2019–2025 (not annual returns):

| Instrument | MOO3 | MSTGAM | IGTSP-Ring | Buy-and-hold |
| --- | ---: | ---: | ---: | ---: |
| AAPL | 4.08% | 233.09% | 313.73% | 754.28% |
| GM | 0.00% | -14.12% | 52.31% | 164.92% |
| GRPN | -51.25% | -88.18% | -55.15% | -73.98% |
| EUR/USD | -0.01% | -1.98% | 0.25% | -11.40% |
| GBP/USD | 1.42% | -13.41% | -9.91% | -8.45% |
| USD/JPY | 1.20% | 4.36% | 6.22% | 18.75% |

AAPL and GM buy-and-hold exceeded all methods' median returns. GRPN produced
substantial losses. Ring's median EUR/USD result fell from +0.25% to -4.43% under
cost stress; USD/JPY fell from +6.22% to -0.09%. No paired method comparison
remained significant at the 5% level after the declared 54-test Holm adjustment.
These results do not establish a consistent profitable method.

The complete artifact check passed across 1,008 primary/stress paths, including
baselines, and 115,063 trade-ledger rows. The largest account P&L reconciliation
error was less than USD 0.000001. The application suite passed 212 tests with
three optional native/service checks skipped. The rendered six-panel equity
figure was inspected.

`reports/stocks-fx-2015-2025/` contains:

- One JSON per instrument: every trial, frozen parameter set, data/code hashes,
  held-out equity observation, trade ledger, paired comparison and cost stress.
- `summary.json`: all asset/seed/method results and a declared family of 54 paired
  HAC tests (six assets × three seeds × three method pairs), with Holm adjustment.
- `report.md` and `equity.png`: median outcomes and the range of held-out equity
  across all declared seeds, alongside costed next-open buy-and-hold.
- `progress.json` and `progress.log`: live run state (jobs done/total, ETA, current
  asset, recent jobs, final status) and a line per event; `gen-trade status` reads
  them. They are monitoring files, not results, and take no part in verification.
- `verification.json`: completed by the verification command after checking
  current source/data fingerprints, every declared seed/year/method, chronological
  boundaries, held-out timestamps and cash/cost ledger conservation.

Each fold starts flat and independently capitalized. The summary chains normalized
fold returns across nonoverlapping years. This is equivalent to proportional
capital scaling under the benchmark's scale-invariant cost model. Dollar ledger
totals are separately labeled as sums over independently capitalized folds.
Results with nonlinear lot/minimum-fee settings need their stated capital basis;
no pooled multi-asset account is inferred.

Ordinary return, annualized volatility/Sharpe, drawdown, loss tails and dependent-data
intervals use marked-to-market equity. Ring's paper monthly-return/component-risk
metrics are separately labeled. Seed medians/ranges are descriptive, not independent
market samples. Holm adjustment covers only the declared tests, not evolutionary
search, later research choices or unreported experiments. Market impact, partial
fills, intrabar order paths, taxes, actual broker swap histories and point-in-time
vendor revisions remain outside this daily research benchmark.
