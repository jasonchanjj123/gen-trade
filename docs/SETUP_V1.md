# Trading setup and evaluation protocol — `setup_version: v1`

This is the declared trading setup and evaluation protocol for the daily stock/FX
benchmark. The machine-readable copy is [`setup-v1.json`](setup-v1.json); it is
generated from the code (`src/gen_trade/setup.py`), and a test fails if the two
diverge. The audit that produced v1 is in [SETUP_AUDIT.md](SETUP_AUDIT.md).

```bash
gen-trade setup                                   # print the v1 declaration (JSON)
gen-trade setup --method-profile robust_monthly   # the same, for another method profile
gen-trade benchmark --protocol v1 --method-profile robust --output reports/v1-robust
python scripts/verify-benchmark.py reports/v1-robust
```

`--protocol v1` is the audited2 protocol plus three things: the liquidity screen
below is **enforced**, every embargo is counted and checked in trading sessions,
and the full declaration and its SHA-256 are stamped into `summary.json`, every
per-asset report and `report.md`. Earlier protocols (`original`, `audited`,
`audited2`) are unchanged, so earlier reports still reproduce. They record the
same liquidity screen as a flag only.

## Trading setup

| Requirement | v1 specification | Code |
| --- | --- | --- |
| **Trading universe** — asset selection | A fixed list chosen in advance. Default: AAPL, GM, GRPN (XNYS) and EUR/USD, GBP/USD, USD/JPY (Dukascopy bid/ask). Lockbox: the rule-based DJIA-2015 and G10 universe, held out (`lockbox.json`). One independent USD account per instrument; no cross-asset portfolio. Not survivorship-free. | `market_data.UNIVERSES` |
| **Trading universe** — liquidity filters | Per instrument and test year, on the **63 sessions before the validation end** (all known before the test embargo). Stocks: median raw close ≥ $5; median close × volume ≥ $10M; fold capital ($100k) ≤ 1% of that dollar volume. FX: both currencies in the G10 with USD on one side, and observed bid/ask quotes. A failing instrument-year is not traded or evaluated; its seeds are not reused by other years. | `setup.liquidity_screen`, `LIQUIDITY_V1` |
| **Decision schedule** — signal calculation time | At the bar's `available_at`: the XNYS close for stocks (early closes respected); 00:00 UTC after the candle for FX. Signals use only bars with `available_at` at or before it. | `backtest.run_backtest`, `market_execution` |
| **Decision schedule** — execution time | Open of the next bar, only if that open is at or after the signal bar's `available_at`. Stocks: next 09:30 ET open plus spread and slippage. FX: buy at the next candle's first ask, sell at its first bid. Targets, stops and holding deadlines are checked at closes and filled at the next open; no intrabar fills. | same |
| **Decision schedule** — delay | 1 bar. Stocks: one overnight (about 17.5 hours). FX: none after the candle completes (the 00:00 UTC first quote is assumed executable; see audit finding 6). Weekly/monthly profiles act only on the first bar of each ISO week or month. Retraining: once per test year, before its first session. | `methods/cadence.py` |
| **Admissible information** — feature lookbacks | Longest finite warm-up: **88 bars** (Ring, TRIX(30)); MOO3 50 bars and 50 calendar days; MSTGAM has no fixed window. EMA, Wilder RSI/ATR/ADX, OBV, Parabolic SAR and directional-change state are recursive: they start at the fold's training start and never use a bar after the decision bar. Cross-market features use a backward as-of join on `available_at` with a finite maximum age; nothing is backfilled. | `setup.FEATURE_LOOKBACKS`, `data.align_features` |
| **Admissible information** — data publication lags | Stock OHLCV: known at the session close. FX OHLC: known at 00:00 UTC after the candle. Splits: applied on the effective date (later split factors only undo the vendor's back-adjustment and are never features). Dividends: ex-date cash entitlement; adjusted closes are not used. Scalers, thresholds and the Ring rule bank fit on training rows only; configuration choice uses validation rows only; test rows are never used to fit or select. | `market_data`, `experiments.run_trials` |
| **Score-to-trade mapping** — entry | Signal alphabet: +1 buy if flat, −1 sell if long, 0 no action. MOO3: evolved Boolean rule tree true at the close. MSTGAM: buy side wins the weighted vote of 70 directional-change genes (hurdle `min_votes` or `vote_margin`). Ring: each sleeve's TA-Lib buy rule. | `methods/*` |
| **Score-to-trade mapping** — exit | MOO3: 5-calendar-day holding deadline or +5% target observed at a close (no sell rule). MSTGAM: sell vote. Ring: sleeve's sell rule, or evolved take-profit/stop-loss in [0, 10%] observed at a close. All exits fill at the next open; positions open at a window's last close are liquidated there with costs. | `runner.MOO3_EXIT` |
| **Score-to-trade mapping** — sizing | All allocated capital in fractional units at the fill price after costs. MOO3/MSTGAM: one position with the fold's whole capital. Ring: capital split across groups by evolved weights (equal in robust profiles) and equally among members, fixed for the year. $100,000 per fold; every fold starts flat. | `backtest.run_portfolio_backtest` |
| **Score-to-trade mapping** — no-trade state | Signal 0 keeps the current state (flat stays in cash at 0%). Warm-up features never trigger. Non-decision bars under weekly/monthly cadence emit nothing. Conflicting Ring rules hold. No configuration with ≥ 2 validation trades → cash for the whole test year. Liquidity-screen failure → instrument-year not traded. | `experiments._cash_report` |
| **Risk constraints** — leverage | 1× cash account; no borrowing. | `instruments.benchmark_spec` |
| **Risk constraints** — concentration | 100% in one instrument per account by design; Ring diversifies only across rules on that instrument. | — |
| **Risk constraints** — exposure limits | Long only; gross exposure ≤ 100% of fold equity; no pyramiding; Ring sleeves fixed at the fold start with no rebalancing or borrowing between sleeves; order ≤ 1% of median dollar volume (stocks). No account drawdown stop, volatility target or kill switch (audit finding 7). | — |
| **Cost model** — commissions | Stocks $0.005/share, no minimum. FX 0.35 bp of notional per side. | `instruments.benchmark_spec` |
| **Cost model** — spreads | Stocks: full spread max(2 bp, $0.01). FX: observed historical bid/ask. | same |
| **Cost model** — other costs | Slippage 1 bp per side (stocks), 0.1 bp (FX). FX financing 0 in the primary scenario (carry sign not modeled). Dividends credited ex-date, no withholding. USD/JPY P&L converted side-aware at contemporaneous bid/ask. No market impact (relies on the 1% participation cap), taxes or idle-cash interest. **Stress:** stock spread max(4 bp, $0.02), $0.01/share, 2 bp slippage; FX 0.7 bp commission, 0.2 bp slippage, 2%/yr financing debit — on the same frozen strategies. One engine applies these in training fitness, validation, test and the buy-and-hold baseline. | `setup.BENCHMARK_SLIPPAGE_BPS` |
| **Version** | `setup_version: v1`, SHA-256 of the declaration in every report. | `setup.SETUP_VERSION` |

## Evaluation

| Item | v1 |
| --- | --- |
| Scheme | Rolling walk-forward on calendar years: train → embargo → validation → embargo → test |
| Training window | 3 calendar years before the validation year, minus the embargo (about 750 sessions) |
| Validation window | The calendar year before the test year, minus the embargo |
| Test window | One complete calendar year; fresh capital, starting flat |
| Step size | 1 calendar year |
| Retraining frequency | Annual: a fresh fit per method, seed and test year. The deployed model is the validated training-only model (pooled seed-mean selection, ≥ 2 validation trades, blend-excess information ratio) |
| Label horizon | Execution delay 1 bar, then: MOO3 ≤ 5 calendar days; MSTGAM and Ring unbounded (until a sell, target or stop) and truncated at the window end; robust profiles' daily marked-to-market fitness 1 bar |
| Maximum feature lookback | 88 bars finite warm-up; recursive state from the training start. Every training window must exceed it (checked per fold) |
| Test dates | 2019-01-01 to 2025-12-31 (seven folds). From 2026, and the lockbox universe, are held out |
| Label purging | **Truncation at every window boundary.** `fit` receives only rows before the training end; validation scores use only validation rows; any position open at a window's last close is liquidated there. No training or validation label ends after its window |
| Embargo | **5 trading sessions** between training and validation and between validation and test. FX Sunday stubs are merged first, so 5 bars are 5 weekday sessions. Checked per fold; a shorter gap stops a v1 run |
| Selection | 2 candidates per method; configuration chosen once per method and fold from seed-mean validation scores; test results never select anything |
| Search budget | Population 64, 20 generations, seeds 11, 22, 33 (declared per run) |

### Fold calendar

| Test year | Training | Validation | Test |
| --- | --- | --- | --- |
| 2019 | 2015–2017 (− 5 sessions) | 2018 (− 5 sessions) | 2019 |
| 2020 | 2016–2018 | 2019 | 2020 |
| 2021 | 2017–2019 | 2020 | 2021 |
| 2022 | 2018–2020 | 2021 | 2022 |
| 2023 | 2019–2021 | 2022 | 2023 |
| 2024 | 2020–2022 | 2023 | 2024 |
| 2025 | 2021–2023 | 2024 | 2025 |

Exact first and last sessions per instrument are recorded in each report's
`fold_audit`.

### Why purging is done by truncation

A fixed-width purge removes the last *h* training bars, where *h* is the longest
label. MSTGAM and Ring hold positions until a sell signal, target or stop. In the
audited2 runs their held-out holding periods had a 95th percentile of 12–85
bars and a maximum of a full year, so a fixed purge would delete a whole year
of training data. v1 instead clips every label at its window boundary: the
simulator liquidates open positions at the window's last close, and `fit` never
receives later rows. This removes exactly the overlapping part of each label.
The 5-session embargo is an extra buffer: it covers the 1-bar execution delay
and MOO3's 5-day maximum hold. Tests that prove the boundary:

- `tests/test_runner.py::test_future_price_changes_cannot_change_training_inputs_or_frozen_parameters`
- `tests/test_runner.py::test_validation_selection_and_refit_never_see_held_out_prices`
- `tests/test_backtest.py::test_terminal_liquidation_uses_final_close_and_records_boundary_convention`

## What a v1 run enforces and records

- **Before fitting:** the declaration is built and validated (every required
  field present, hash current). Each instrument's folds are audited: dates,
  session counts, embargo sessions, warm-up and the liquidity screen. A short
  embargo or warm-up stops the run. Failing instrument-years are dropped. An
  instrument with no passing year is excluded and listed in
  `summary.json` (`excluded_instruments`).
- **Seeds stay tied to the calendar:** a dropped year does not shift the seeds
  of the remaining years.
- **In the outputs:** `summary.json` holds `setup` and `liquidity_screen`;
  each asset report holds `setup_sha256` and `fold_audit`; `report.md` has
  "Setup v1" and "Liquidity screen" sections.
- **On verification:** `scripts/verify-benchmark.py` re-validates the
  declaration, recomputes every fold's liquidity screen, embargo sessions and
  warm-up from the dataset, and requires exactly the traded years.

On the current default dataset, v1 drops **GRPN 2019 and 2020** (median price
$3.17 and $2.85) and **GRPN 2023** (median dollar volume $6.3M; a $100k order
would be 1.6% of it). Every other instrument-year passes, and every embargo is
exactly 5 trading sessions.

## Changing the setup

Do not edit v1 in place. A change to trading or evaluation rules (universe,
screen thresholds, timing, information, mapping, risk, costs, windows, purging
or embargo) is a new version: add `PROTOCOLS['v2']`, set `SETUP_VERSION`
accordingly, and commit a new `docs/setup-v2.json`.
`tests/test_setup.py::test_committed_v1_declaration_matches_the_code` fails on any
drift. If only wording changed, regenerate with
`gen-trade setup --output docs/setup-v1.json`.
