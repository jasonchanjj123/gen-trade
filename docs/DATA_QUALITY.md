# Data sources and quality

This page has two parts: the audit of the data sources and quality controls
(2026-11-07), and how to use the data-quality controls added after it
(`gen_trade/data_quality.py`, `gen-trade audit-data`).

## Audit, 2026-11-07

Scope: the default dataset `data/stocks-fx-2015-2025` (AAPL, GM, GRPN from
Yahoo Finance; EUR/USD, GBP/USD, USD/JPY from Dukascopy bid/ask), the
acquisition code (`market_data.py`), the data contract (`data.py`) and how
benchmarks consume the data. The lockbox data were not opened; only file
presence was checked.

**Verdict.** The data content is sound. The controls around it were not
enough for an enterprise setting: no completeness check for FX, no anomaly
screening, no independent source, no lineage from dataset files back to the
code that built them, and no quality record in benchmark reports. Two defective
FX candles had been accepted without notice. The controls below close those
gaps. All error-level checks pass on the current dataset.

### Already in place

- Raw responses cached with SHA-256 sidecars; dataset files hashed in
  `manifest.json` and checked on every load.
- Strict XNYS session coverage for stocks, with no interpolation; missing
  OHLCV rejected.
- Causal bar contract (`validate_bars`): timezone-aware, strictly ordered,
  availability after the bar opens, positive finite OHLC with high/low
  containment.
- Contemporaneous stock prices reconstructed from split-adjusted vendor data,
  with a regression test against later vendor revisions; dividends kept as
  cash events.
- FX: separate bid and ask, crossed-quote quarantine (one EUR/USD candle,
  2019-03-10), empty and flat placeholders excluded.

### Findings

| # | Finding | Impact | Now |
| --- | --- | --- | --- |
| 1 | **Two partial Dukascopy candles were accepted.** EUR/USD 2016-06-08 holds feed activity of 54 against ~200,000 on a normal day and a 1.4-pip range; it opens 43 pips above the prior close. The ECB fix that day lies 17.7 bp outside the candle. USD/JPY 2017-07-12 has ~1% of normal activity and opens mid-move. | Both sit in the 2015–2017 training years of the 2019 fold (and its 2015–2018 refit): wrong indicator inputs on two days. Test years are unaffected. | Flagged (`completeness.partial_bars`, and the ECB check for EUR/USD). Not removed: dropping them changes the dataset hash and comparability with earlier reports, so it is a research decision. |
| 2 | **No completeness check for FX.** Stocks were checked against the XNYS calendar; FX had none, so a missing week would have passed. | Silent gaps distort indicators and annualization. | Weekday coverage (Jan 1 and Dec 25 excluded as closed) and ECB business-day coverage. Current data: only 2016-01-01 and 2021-01-01 are absent, both closed days. |
| 3 | **Interrupted downloads could not resume on a later day.** The Yahoo URL embeds the retrieval day, so a cached response failed the cache check ("Raw cache integrity mismatch"). | The lockbox fetch in `data/lockbox-2015-2025` is stuck in this state: 24 stock files and part of the FX years, no manifest. | Fixed: a cached response to the same query that reaches the requested end is reused. |
| 4 | **Fragile retries.** Two immediate attempts, no backoff; a missing file (404) was retried. | Throttling by the vendor fails a fetch. | Four attempts with exponential backoff and jitter, `Retry-After` honoured, no retry on permanent errors, empty responses rejected. |
| 5 | **Vendor metadata not enforced.** The source currency was recorded but not checked; listing time zone, instrument type and bar size were not checked. | A non-USD or wrong listing would be priced in the USD account silently. | The Yahoo response must be a USD daily equity listed in New York, for the requested symbol. |
| 6 | **Single source, no reconciliation.** Every price came from one vendor per asset class. | Vendor errors are invisible. | ECB reference rates reconcile FX (an independent price verification). Yahoo's dividend events are reconciled against its own adjusted closes. A reference file from any licensed vendor can be supplied. |
| 7 | **No lineage from dataset files to code.** Nothing recorded which normalizing code built a dataset file, so files could silently drift from the current code. | Results could rest on derived data the current code would not produce. | New fetches record `lineage.normalizer_sha256`. The audit rebuilds every file from its raw cache with the current code and requires an exact match. The current dataset reproduces bit for bit. |
| 8 | **No anomaly screening or review trail.** | Bad prints and genuine events were indistinguishable and unrecorded. | Robust outlier, spike-and-reversal, wick, stale-price and spread screens, plus a reviewed-exceptions register. |
| 9 | **Columns beyond OHLCV were unchecked.** Execution prices, bid/ask sides and corporate actions had no data-layer contract. | A corrupted side would reach the execution engine. | Contracts for every price set, ask ≥ bid, signal/execution consistency, corporate-action validity, session dates. |
| 10 | **Benchmarks recorded no quality status.** | A report could not show what data checks it relied on. | Benchmarks read `quality.json`, refuse failed error checks, archive the audit as `data-quality.json` and summarize it. `verify-benchmark.py` checks the archive. |
| 11 | **Unclear manifest errors.** A malformed manifest raised `KeyError`; file names were not confined to the dataset directory. | Hard-to-diagnose failures. | Manifest structure and schema version validated, with plain-file-name and missing-file errors. |

### Reconciliation results

| Check | Result |
| --- | --- |
| ECB reference fixes inside the daily bid-low/ask-high range | 2,816 business days per pair, all matched to a bar. GBP/USD max deviation 0.02 bp, USD/JPY 0.5 bp; EUR/USD one day outside (finding 1). |
| Yahoo dividends against adjusted-close steps | AAPL 44 and GM 35 events in the window, maximum relative error 0.03%; no unexplained adjustment steps. GRPN pays none. |
| Rebuild from the raw cache | All six instruments identical, including column types. |
| XNYS sessions | All 2,766 sessions present for each stock, none unexpected. |

### Warnings to review

The audit flags these observations. None is a contract violation. Each
needs a person to confirm it is genuine and then record it in the exceptions
register:

- GBP/USD 2016-06-24 return outlier: the EU referendum result.
- GM 2015-08-24 and 2020-03-18 wicks: the 24 August 2015 opening crash and
  the March 2020 sell-off.
- GM 2018-05-31 and 2025-10-21 return outliers; GM 2025-07-22 spike and
  reversal.
- GRPN: eight return outliers (2015-11-04, 2016-07-28, 2016-10-27,
  2017-02-15, 2020-02-19, 2024-03-15, 2025-03-12, 2025-05-08), dates
  consistent with earnings releases of a volatile small cap; three identical
  closes ending 2019-03-14.
- EUR/USD 2017-01-02 and 2023-01-02; GBP/USD and USD/JPY 2023-01-02: wide
  opening spreads on New Year holiday Mondays.
- EUR/USD 2016-06-08 and USD/JPY 2017-07-12: partial candles (finding 1).

### Observations

- Friday closing spreads are about ten times wider than on other days
  (EUR/USD median 3.9 bp against 0.34 bp). The UTC daily candle closes after
  the weekend shutdown. Weekly-open (Sunday) spreads are 4–9 bp. Fills happen
  at opens. Under the audited protocols the merged Monday bar opens at the
  Sunday stub's quotes, so entries after a Friday signal pay the weekly-open
  spread. That is conservative. Friday marks lower mark-to-market equity by
  roughly 2 bp until Monday.
- Dukascopy's daily candles are bucketed on UTC days. Execution at the "open"
  means the first quote of that UTC day, which may sit in thin liquidity.

### Remaining gaps (need data the project does not have)

- **Licensed, point-in-time equity data.** Yahoo's chart endpoint is
  unofficial and undocumented, with no service level and terms that restrict
  use. The universe is not survivorship-free. An enterprise setup should
  license point-in-time history with delisted names and full corporate
  actions. Feed its closes to `--reference` until then.
- **No automated second source for stocks.** No free official daily equity
  source exists. Stooq now requires a browser challenge, which this project
  does not bypass.
- **Intraday quotes** for execution realism; **historical FX swaps**.
- **Versioned raw storage.** `data/` is ignored by Git. Keep raw caches and
  manifests in versioned object storage (or DVC) so an audit can be repeated
  years later.

## Using the controls

```bash
gen-trade audit-data data/stocks-fx-2015-2025 --ecb   # writes quality.json, prints a scorecard
gen-trade audit-data data/markets.parquet             # any compare-format file; prints only
gen-trade audit-data DATASET --strict                 # exit 3 while warnings are unreviewed (for CI)
```

`fetch-data` runs the audit after every fetch, with the ECB check when FX pairs
are present (`--no-audit`, `--no-ecb` turn these off). Exit status 3 means a
failed error check (or, with `--strict`, an unreviewed warning). The ECB file
is downloaded once into `raw/ECB/` with a SHA-256 sidecar like other raw data.

### Severities

- **error**: results built on the data would be wrong. Benchmarks refuse the
  dataset unless `--allow-data-quality-failures` is passed; that override is
  recorded in `summary.json` and `report.md`.
- **warning**: an observation for a person to review. Most are genuine market
  events. A reviewed exception records the decision.
- **info**: describes the data without judging it.

The audit never edits, fills or removes an observation.

### Checks

| Check | Dimension | Severity | Verifies |
| --- | --- | --- | --- |
| `contract.dataset` | validity | error | manifest structure, file presence and hashes |
| `contract.bars` | validity | error | the causal bar contract (`validate_bars`) |
| `contract.schema` | validity | error | columns required by the asset class are present and numeric |
| `contract.price_sets` | validity | error | every OHLC set (signal, execution, bid, ask) positive with high/low containment |
| `contract.bid_ask` | validity | error | ask ≥ bid at the open and the close |
| `contract.signal_prices` | consistency | error | signal OHLCV = execution × already-effective splits (stocks), or the bid (FX) |
| `contract.corporate_actions` | validity | error | split ratios positive; dividends finite, nonnegative; none for FX |
| `contract.session_dates` | consistency | error | `session_date` is the bar date in the market time zone |
| `completeness.sessions` | completeness | error (stocks), warning (FX) | every XNYS session; every FX weekday except Jan 1 and Dec 25 |
| `completeness.partial_bars` | completeness | warning | activity ≥ 5% of the trailing median on full sessions |
| `completeness.reference_coverage` | completeness | warning | every reference date inside the data range has a bar |
| `timeliness.window` | timeliness | warning | data starts and ends within 7 days of the requested window |
| `accuracy.return_outliers` | accuracy | warning | robust z-score of close-to-close log returns ≤ 10 |
| `accuracy.spike_reversals` | accuracy | warning | no move beyond z 6 reversed by more than half on the next bar |
| `accuracy.wicks` | accuracy | warning | wicks ≤ 6 trailing median ranges |
| `accuracy.stale_prices` | accuracy | warning | no 3+ identical closes in a row; no zero-range full sessions |
| `accuracy.intraday_crossed` | accuracy | warning | ask high/low ≥ bid high/low (no crossed quote inside the bar) |
| `accuracy.spreads` | accuracy | warning | opening spread ≤ 10× its trailing median (or ≤ 3 bp); records spreads by weekday |
| `consistency.split_dates` | consistency | error | no split-sized gap in split-neutral prices on a split date (vendor history adjusted) |
| `consistency.split_like_jumps` | consistency | warning | no overnight move of 1.9× or more without a split event |
| `consistency.dividends` | consistency | warning | each dividend ≤ 10% of the prior close |
| `reconciliation.vendor_dividends` | accuracy | error | Yahoo dividend events match the steps in its adjusted close |
| `reconciliation.reference_prices` | accuracy | warning | reference closes within 25 bp; reference fixes inside the bar range (2 bp tolerance) |
| `lineage.raw_integrity` | lineage | error | raw files match their sidecars and the manifest hashes |
| `lineage.raw_availability` | lineage | warning | the raw responses behind each file are cached |
| `lineage.rebuild` | lineage | error | rebuilding from raw with current code reproduces the file exactly |
| `lineage.normalizer` | lineage | info | the normalizing code is unchanged since the fetch |

Robust scales use the median and median absolute deviation of the 60 earlier
full sessions (at least 20), so each screen uses only data before the row it
judges. Thresholds are fields of `QualityThresholds`. They were calibrated on
the default dataset so that flags stay few enough to review.

### Reviewed exceptions

`data-quality-exceptions.json` in the repository root (or `--exceptions PATH`)
records decisions about warnings. Keep it under version control:

```json
{"exceptions": [
  {"symbol": "GBP/USD", "check": "accuracy.return_outliers", "dates": ["2016-06-24"],
   "reason": "EU referendum result", "reviewed_by": "J. Analyst", "reviewed_on": "2026-11-07"}
]}
```

A warning whose flagged dates are all covered becomes `reviewed`. Warnings
only: errors are fixed in the data, not waived. The report stores the
register's hash.

### Reference prices from another vendor

`--reference FILE` (CSV or Parquet) reconciles any independent source. It
needs the columns `symbol`, `session_date`, `value` and `kind`, plus an
optional `source`:

- `close`: an as-traded official closing price (stocks), compared with
  `execution_close` within 25 bp;
- `fix`: a price observed during the session (e.g. a benchmark fixing), which
  must lie inside the bar's range.

### Benchmarks

`gen-trade benchmark` reads `quality.json` from the data directory:

- no file: the run is recorded as `not audited`;
- a file for a different manifest: refused as stale;
- failed error checks on the run's instruments: refused, unless overridden.

The audit used is copied to `data-quality.json` in the output directory and
summarized in `summary.json` (`data_quality`) and `report.md`.
`scripts/verify-benchmark.py` checks the copy's hash, and that it still
describes the current manifest.
