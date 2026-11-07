# Audit: trading setup and evaluation protocol (2026-11-07)

Scope: the daily stock/FX benchmark (`benchmark.py`, `runner.py`,
`experiments.py`, `backtest.py`, `market_execution.py`, `market_data.py`,
`instruments.py`, `methods/`), its documentation and the finished reports under
`reports/`. Checked against the required setup fields (universe, decision
schedule, admissible information, score-to-trade mapping, risk constraints,
cost model, version) and evaluation requirements (walk-forward, windows, step,
retraining, label horizon, feature lookback, test dates, purging, embargo).
The lockbox data were not opened.

**Verdict.** The protocol was leak-free by construction and well documented,
but it did not meet the requirements as stated: there was no liquidity
filter, no setup version, and the label horizon, maximum feature lookback and
purging rule were implicit. Protocol `v1` ([SETUP_V1.md](SETUP_V1.md)) closes
these gaps. Four items stay open by design (findings 6–9); they need a decision
or data the project does not have.

## Requirement coverage

| Requirement | Before | v1 |
| --- | --- | --- |
| Universe: asset selection | Met (fixed list; rule-based lockbox) | Met; declared |
| Universe: liquidity filters | **Missing** | Enforced point-in-time screen |
| Schedule: signal time, execution time, delay | Met in code; spread across docs | Declared in one place |
| Information: feature lookbacks | **Not declared** (88-bar maximum found by measurement) | Declared; test checks it against TA-Lib |
| Information: publication lags | Met (`available_at`, next-open rule) | Declared |
| Mapping: entry, exit, sizing, no-trade | Met per method; spread across docs | Declared |
| Risk: leverage, concentration, exposure | Met implicitly (1×, long-only, single instrument) | Declared; no portfolio limits (finding 7) |
| Cost model | Met | Declared, including stress |
| Version | **Missing** | `setup_version: v1` + SHA-256 in every report |
| Walk-forward, windows, step, retraining, test dates | Met (calendar 3y/1y/1y, annual) | Declared with fold calendar |
| Label horizon | **Not declared** | Declared per method |
| Purging | Met implicitly (truncation) | Declared; tests cited |
| Embargo in trading days | **Partial**: 5 observations, which is 4 sessions in some FX folds under `original` | Counted in sessions; v1 refuses fewer than 5 |

## Findings

**1. No liquidity filter (high; fixed in v1).** Nothing checked whether the
instruments could absorb the orders the cost model assumes free of impact.
Measured on the 63 sessions before each test year:

| GRPN test year | Median price | Median dollar volume | $100k order share |
| --- | ---: | ---: | ---: |
| 2019 | $3.17 | $22.0M | 0.45% |
| 2020 | $2.85 | $11.3M | 0.89% |
| 2023 | $7.66 | $6.3M | 1.59% |

AAPL ($6–13B a day) and GM ($0.3–1B) pass easily. This matters for the
research conclusions: the monthly MSTGAM result that became pre-registered
hypothesis H1 (+5.3%/yr over the blend) was "concentrated in GRPN and GM". It
should be re-checked on the instrument-years that pass the screen. The lockbox
universe (DJIA stocks, G10 FX) is unaffected.

**2. No setup version or single declaration (medium; fixed).** The setup was
spread across five documents and constants in four modules. Nothing tied a
report to the exact rules it ran under, beyond the code hash. v1 builds one
declaration from the code's constants, validates the required fields, hashes
it, stamps it into every report, and keeps a committed copy that a test
compares with the code.

**3. Label horizon and maximum feature lookback undeclared (medium; fixed).**
Held-out trades in the audited2 runs:

| Method | Median hold (bars) | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| MOO3 | 4 | 4–5 | 5 |
| MSTGAM | 2–22 | 12–85 | 77–173 |
| IGTSP-Ring | 5–13 | 31–64 | 259 |

The longest finite indicator warm-up is TRIX(30) in the Ring bank at 88 bars
(MOO3: 50). Recursive indicators carry state from each fold's training start.

**4. Purging was implicit (low; declared).** No leak was found. `fit` receives
only training rows, validation is scored on validation rows only, and the
simulator liquidates open positions at each window's last close, so no label
crosses a boundary. Because MSTGAM and Ring horizons are unbounded, the
5-session embargo is *not* what prevents leakage for them; truncation is. v1
states this and cites the tests that prove it.

**5. Embargo counted in observations (low; fixed).** Under the `original`
protocol a raw FX Sunday stub counts as one of the 5 embargo bars, leaving 4
trading sessions in some folds. Stocks and the audited protocols (stubs
merged) were already at exactly 5. v1 counts sessions per fold and refuses a
shorter gap.

**6. FX fills assume zero latency at the 00:00 UTC rollover (medium; open).**
A daily candle completes at 00:00 UTC, and the signal fills at the next
candle's first quote at the same instant. Rollover is the thinnest, widest-spread
moment of the FX day. The observed first bid/ask is charged, but whether it was
executable is unknown. Options: fill at a later fixed time (needs hourly data),
or add a rollover-spread stress scenario.

**7. No account-level risk limits (low for research; high before any live use;
open).** There is no drawdown stop, volatility target, kill switch or
cross-instrument allocation. Each instrument is a separate 1× long-only
account. This is a deliberate research simplification, now declared.

**8. Cost model gaps (medium; open).** No market impact (acceptable only under
the participation cap, now enforced). A fixed proportional stock spread is used
instead of historical quotes. FX carry is zero in the primary scenario, although
long USD/JPY earned positive carry. Historical broker swap data would close the
last gap.

**9. Unbounded holding periods (design; open).** MSTGAM and Ring can hold
positions for a full year. This makes the label horizon unbounded and lets
truncation shape the last trades of each window. A declared maximum holding
period (for example 63 sessions) in a v2 would bound the horizon, allow a
classical fixed-width purge, and make fitness less sensitive to window edges.
This changes the methods, so it needs a decision.

**10. Annualization uses whole-sample bar counts (information).** Bars per
year are counted over 2015–2025. This only scales reported statistics and never
enters fitting or selection.

**11. Default CLI protocol is still `original` (decision).** Keeping it
reproduces the published first report. Making `v1` the default would make
every new run compliant by default.

## Verification

- 9 new tests in `tests/test_setup.py`: required fields and stale hashes; the
  declaration matches the code's constants and the committed
  `docs/setup-v1.json`; declared lookbacks equal TA-Lib's actual warm-ups; the
  liquidity screen ignores rows after its cutoff and applies each threshold;
  session counting with raw and merged FX Sunday stubs; seeds stay tied to the
  calendar when years are dropped; an end-to-end v1 and `original` benchmark
  with an illiquid year and an always-illiquid stock, verified by
  `scripts/verify-benchmark.py`, which also rejects a tampered fold audit.
- Full suite: all existing tests still pass (the earlier protocols are
  unchanged).
- Real data: the v1 fold audit on `data/stocks-fx-2015-2025` drops GRPN 2019,
  2020 and 2023 and finds every embargo at exactly 5 sessions.
- Not run: a full v1 benchmark (hours of compute). Run
  `gen-trade benchmark --protocol v1 --method-profile <profile>` to get results.

## Enterprise improvements

Ordered by what most reduces the risk of a wrong decision.

**Now: research integrity**

1. Run v1 for the profiles that produced the current conclusions, and compare
   with audited2 to see which conclusions survive the liquidity screen.
2. Add CI (none exists; `.github/` is absent): `pytest`, a linter
   (`ruff`) and a type checker on every pull request, plus a required check
   on `docs/setup-*.json` changes so a setup change needs explicit review.
3. Make lockbox unlocks auditable: require a signed or reviewed commit of
   `lockbox.json`, log every unlock (who, when, code hash, setup hash) to an
   append-only store, and keep the lockbox data in separate storage that
   research runs cannot read.
4. Record a run manifest for every benchmark (git commit, setup hash, data
   hashes, environment lock, command line, host) in the existing PostgreSQL
   store, so any number in a report traces to an immutable run.

**Next: data and realism**

5. Point-in-time, survivorship-free equity data (with delisted names,
   spin-offs and mergers) from a licensed vendor. The current pipeline handles
   only splits and cash dividends, which is why the lockbox had to exclude
   eight DJIA names.
6. Intraday or quote data for execution realism: fills at a fixed time after
   the open or rollover, measured spreads, and a square-root market-impact
   model calibrated on participation.
7. Historical swap/carry for FX and borrow costs for any future short side.
8. Versioned datasets (object storage or DVC) with schema checks at ingestion
   (pandera is already in the research extra). *Partly done 2026-11-07:*
   ingestion contracts, anomaly screens, ECB reconciliation, rebuild-from-raw
   lineage and a benchmark quality gate ([DATA_QUALITY.md](DATA_QUALITY.md));
   versioned raw storage remains.

**Later: governance and production readiness**

9. Model risk management in the style of SR 11-7: an independent validator
   signs off on each setup version; model cards per method profile; periodic
   re-validation; a register of known limitations (findings 6–9).
10. Portfolio and risk layer before any live use: cross-instrument allocation,
    exposure and concentration limits, drawdown stop, volatility targeting,
    pre-trade checks and a kill switch.
11. Paper trading with the same engine and setup version, with daily
    reconciliation of paper fills against backtest assumptions, and
    monitoring of feature and performance drift.
12. Configuration as typed schemas (pydantic is already a research
    dependency) for setups and method profiles, replacing nested dicts.
