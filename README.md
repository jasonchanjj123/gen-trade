# gen-trade

Three independent evolutionary trading sub-frameworks, with a causal
simulator, optional Rust acceleration, chronological validation, parallel experiments, and statistical
reports. These are **documented adaptations**, not reproductions of the authors'
published numerical results. No live trading is implemented.

| Method | Representation and search | Training objective |
| --- | --- | --- |
| **MOO3** — [Long et al.](https://doi.org/10.1007/s10462-025-11390-9) | Typed trading-rule trees; NSGA-II; 28 DC + 28 technical indicators | Maximize paper total return and mean trade return; minimize trade-return risk; select the final training Pareto front by modified Sharpe |
| **MSTGAM** — [Salman et al.](https://doi.org/10.1007/s10462-025-11419-z) | 70 normalized weights over eight DC strategies and ten thresholds | Maximize the paper's trade-based Sharpe score |
| **IGTSP-Ring** — [Chideme et al.](https://doi.org/10.1080/0305215X.2025.2592030) | Training-selected TA-Lib rule bank; TP/SL and weight bits; grouped strategies; island GA with adjacent best-for-best ring exchanges | Detailed paper Eqs.6–16: monthly return / weighted component risk + group entropy |

Read [the source mapping, equations, and adaptation notes](docs/PAPER_METHODS.md)
before using these methods. See the [implementation audit](docs/IMPLEMENTATION_AUDIT.md)
and [IGTSP-Ring research and specification](docs/IGTSP_RING.md).
IGTSP-Ring was checked against the user-supplied full PDF, including its equations,
operators and Table 5 defaults. Omitted rule choices and conflicting risk definitions
are documented. Published numerical results have not been reproduced. The papers use different fees and return
statistics; their scores must not be compared as ordinary annualized Sharpe.

## Complete development environment

The recommended experiment environment is Python 3.12.13 with
NumPy/pandas/SciPy/TA-Lib and spawned process workers:

```bash
bash scripts/setup-environment.sh --core     # add --native for the Rust backend (about 5-10x faster runs)
source scripts/activate.sh                   # bash or zsh
bash scripts/doctor.sh                       # what is installed, what is missing, how to fix it
gen-trade demo --workers 3 --cross-market-features --output reports/three-method-comparison.json
pytest -q
```

Setup is safe to rerun: it reuses installed pieces, remembers the last profile,
logs to `.tools/logs/`, and verifies before reporting success.

The repository also offers the optional full research environment:
Freqtrade/DEAP/Optuna, statistics and plotting tools, Polars/Arrow/DuckDB,
Rust 1.90.0 with tested PyO3/Rayon integration, and PostgreSQL 17 / Redis 7.4
service configuration. See [environment setup](docs/ENVIRONMENT.md) for prerequisites,
version pins, persistent storage, and verification details.

```bash
cd gen-trade
bash scripts/setup-environment.sh --full
source scripts/activate.sh
bash scripts/services.sh up -d --wait --wait-timeout 60
python environment/verify.py --services
gen-trade demo --cross-market-features --output reports/synthetic-comparison.json
GEN_TRADE_TEST_SERVICES=1 pytest tests/ -q
```

The demo generates invented BTC, gold, and FX OHLCV. Its JSON report records
train/validation selections, fresh refits, held-out fold statistics, paired dependent-data
inference, a next-open buy-and-hold baseline, costs, seeds, source/data hashes,
and limitations. **Synthetic results validate software, not profitability.**

## Core-only installation

For a smaller installation without the complete research tools, Python 3.12 and
`uv` are required. The optional native build requires Rust 1.90. Use the setup
script above to install and verify the full stack, including the Rayon probe.

```bash
uv sync --locked --extra dev --extra storage
uv run --no-sync pytest tests/ -q
# Optional native backend; run maturin from native/ to use its own metadata.
cd native
uv run --project .. --no-sync maturin build --release --locked --out ../dist
cd ..
uv pip install --python .venv/bin/python --no-deps dist/gen_trade_native-*.whl
```

`uv sync` removes packages not declared in the project, including a manually
installed native wheel. Reinstall that wheel after syncing when the Rust backend
is wanted. Python fallback always remains available. `scripts/setup-environment.sh
--native` handles this order (and reuses an existing rustup).

The Rust module runs the trade simulator (`execute_batch`), which is about 90% of a
benchmark's work in Python, and DC extraction. Its results are identical to the
Python reference: `tests/test_native_execution.py`, `tests/test_dc.py` and
`environment/verify.py` compare them exactly. On an M4 Mac, one asset/year/seed of
`--protocol audited2 --method-profile robust_monthly` took 4.7 s instead of 46 s,
and four assets × two years × two seeds with `--workers 8` took 30 s instead of
179 s, with identical reports. `python scripts/benchmark_execution.py` times single
calls; `python scripts/benchmark_dc.py` times DC, where the gain is small (about
1.4×). Set `GEN_TRADE_EXECUTION_BACKEND=python` to run the Python reference;
reports record the engine as `runtime.execution_backend`.

## Supply real data

The daily stock/FX benchmark uses AAPL, GM, GRPN, EUR/USD, GBP/USD and USD/JPY
from 2015–2025. It tests the seven complete years 2019–2025 with calendar-based
training, validation and refitting. It downloads and hashes real observations,
handles stock splits/dividends and FX bid/ask accounting, and exports complete
trade/equity ledgers under primary and doubled-cost scenarios.

```bash
bash scripts/setup-environment.sh --market-data
source scripts/activate.sh
gen-trade fetch-data --directory data/stocks-fx-2015-2025
gen-trade benchmark --population 64 --generations 20 --seeds 11 22 33 \
  --workers 3 --output reports/stocks-fx-2015-2025
python scripts/verify-benchmark.py reports/stocks-fx-2015-2025
```

### Data quality

`fetch-data` audits every dataset it builds, and `gen-trade audit-data` re-audits
one. The audit covers data contracts, session completeness, anomaly screens,
reconciliation (FX against ECB reference rates; Yahoo dividends against its own
adjusted closes; an optional licensed reference file) and lineage (every file is
rebuilt from its raw cache with the current code). It writes `quality.json`.
Benchmarks refuse a dataset with failed error checks and archive the audit they
used. Warnings are reviewed and recorded in `data-quality-exceptions.json`. See
[data sources and quality](docs/DATA_QUALITY.md), including the 2026-11-07 audit.

```bash
gen-trade audit-data data/stocks-fx-2015-2025 --ecb
```

### Following a long run

`benchmark`, `demo` and `compare` show live progress: a redrawn line on a terminal
(jobs done, ETA, last finished fold/seed/method), or one line per finished job when
output is redirected. `--progress off` keeps only the per-asset messages. A benchmark
also keeps `progress.json` and `progress.log` in its output directory:

```bash
gen-trade status                                   # every run under reports/
gen-trade status reports/stocks-fx-2015-2025 --watch   # refresh until it stops
tail -f reports/stocks-fx-2015-2025/progress.log
```

A job is one method fit for one test year and seed (audited protocols count the
validation and deployment phases separately). Status says `stopped` when the process
disappeared without finishing; rerun the same command to resume, since finished
assets are verified and reused. A sleeping Mac pauses the run: prefix long runs with
`caffeinate -i` to keep it awake.

See the [stock/FX protocol, costs and data limitations](docs/STOCKS_FX_BACKTEST.md).
The benchmark runs six separate instrument accounts, long-only at 1× exposure;
it does not allocate a common portfolio across stocks and currencies. Its declared
research budget differs from the papers' full experimental budgets.

For other feeds, the generic comparison command accepts your data:

CSV or Parquet must contain `symbol`, timezone-aware `timestamp` (bar open),
`available_at` (when full candle data became known), and `open,high,low,close,volume`.
Rows per symbol must be strictly chronological, unique, and nonoverlapping.
Prices must be finite and positive; missing volume is allowed. Delayed/revised or
overlapping feeds need an ingestion adapter before use. No timestamps, market
calendars, or missing prices are inferred silently.

```bash
gen-trade compare data/markets.parquet --pair BTC/USDT \
  --periods-per-year 105120 --train-bars 50000 --validation-bars 10000 --test-bars 10000 \
  --embargo-bars 1500 --population 100 --generations 20 \
  --workers 3 --seeds 11 22 33 --method-configs examples/experiment-config.json \
  --cross-market-features --fee 0.001 --slippage-bps 2 \
  --output reports/real-data-comparison.json
```

`105120` is an explicit **365 × 24 × 12** convention for five-minute 24/7 data;
choose the convention appropriate to your instrument, calendar, and observations.
Population/generation defaults are small for a smoke run. All three classes
retain their paper population/generation defaults; ring also retains Table 5's
10 islands, 17 groups, TP/SL bit widths and operator/migration rates. Equal
population/generations are not equal total compute: calibration, candidate grids
and bank backtests differ. Ring population is the **total across islands**.
Diagnostics retain evaluation counts and runtimes.

The default protocol is train → embargo → validation → embargo → test. Two
candidate configurations per method fit on training only; validation selects
by log net wealth minus drawdown. A new model fits through validation end before
frozen test evaluation. All requested seeds and all three pairwise comparisons
are retained. Test results never select configurations, seeds or a framework.
This is rolling chronological validation, not full purged combinatorial CV or
exact replication of each paper's tuning. Select embargo length to cover the
strategy's information/holding horizon.

`--workers 3` runs independent method/fold/seed jobs concurrently. For parallel
islands inside the ring framework, use `--workers 1 --island-workers 4`. Numerical
libraries use one thread per process worker; nested pools are rejected.
`--validation-bars 0` permits only one fixed configuration per method. Parallel
API calls from scripts need an `if __name__ == "__main__":` guard; notebooks can
invoke the CLI or use `workers=1`.

## Chromosomes and external markets

`ChromosomeContext` declares the trading pair, allowed UTC entry hours, and typed
`FeatureGene` references to OHLCV sources with lag, rolling window, and maximum
feature age. Cross-market features use backward as-of joins on availability time,
preserve warm-up/missing values, and never use future observations.

MOO3 can evolve rules referencing those external terminals. MSTGAM deliberately
keeps its published 70-weight representation. Ring selects its strategy bank on
training only, then evolves groups, TP/SL bits and group-capital bits. The selected
bank and genes stay frozen during prediction. Pair/session choices are explicit
experiment context in this version; they are not automatically evolved genes.
Extending the representations to jointly evolve pairs and entry sessions remains
separate research work, as does multi-asset portfolio execution.

## Database and cache

```bash
bash scripts/services.sh up -d --wait --wait-timeout 60
source scripts/activate.sh
GEN_TRADE_TEST_SERVICES=1 pytest tests/test_storage.py -q
gen-trade demo --persist --experiment-id demo-001 --output reports/demo-001.json
```

`DATABASE_URL` selects PostgreSQL. Experiment IDs are immutable: retries of the
same JSON report succeed; a different report under that ID fails. `FitnessCache`
provides optional Redis TTL caching with keys containing chromosome, dataset,
code, execution, fold, and seed context. Cache failure means recompute. The
optimizers currently use fit-local memoization; the Redis helper is available
to an orchestration layer and is not silently enabled inside their fitness loops.

## Trading setup v1

The declared trading setup and evaluation protocol is `setup_version: v1`
([docs/SETUP_V1.md](docs/SETUP_V1.md); audit in [docs/SETUP_AUDIT.md](docs/SETUP_AUDIT.md)).
It covers universe and liquidity filters, decision schedule, admissible
information, score-to-trade mapping, risk constraints, costs, and the
walk-forward windows, label horizon, feature lookback, purging and embargo.
`--protocol v1` enforces it and stamps it into every report:

```bash
gen-trade setup                                   # the v1 declaration as JSON
gen-trade benchmark --protocol v1 --method-profile robust --output reports/v1-robust
python scripts/verify-benchmark.py reports/v1-robust
```

## Theory and audit extensions

Labeled, opt-in method variants and an audited benchmark protocol are documented
in [docs/EXTENSIONS.md](docs/EXTENSIONS.md). Defaults stay paper-faithful:

```bash
gen-trade benchmark --protocol audited --method-profile improved --output reports/audited-improved
# Review 2026-11-05: blend-excess fitness and validation, tests against the exposure-matched blend
gen-trade benchmark --protocol audited2 --method-profile robust --output reports/audited2-robust
# Turnover: weekly or monthly decisions (docs/EXTENSIONS.md, last section)
gen-trade benchmark --protocol audited2 --method-profile robust_weekly --output reports/audited2-robust-weekly
gen-trade reanalyze reports/audited-improved --output reports/audited-improved-reanalysis --data-directory data/stocks-fx-2015-2025
```

`lockbox.json` holds out test years from 2026 and a rule-based second universe (22
DJIA-2015 stocks, 6 USD pairs; `gen-trade fetch-data --universe lockbox`); the
benchmark refuses them unless `--unlock-lockbox` runs a profile that still matches
its frozen hash. Unlocked runs evaluate the hypotheses pre-registered in
`docs/preregistered-hypotheses.json` (docs/EXTENSIONS.md, last section).

## Statistical scope

Reports include net equity returns, annualization under an explicit sampling
convention, drawdown and duration, trade/tail statistics, stationary-block
bootstrap confidence intervals, and HAC mean-return inference. Paired comparisons
require identical held-out timestamps. Undefined/degenerate values stay `null`.
Nominal p-values are labeled unadjusted; Holm-adjusted HAC values cover the three
paired comparisons within each fold/seed. One run does not account for evolutionary
selection, many strategy trials, many assets, or dependent folds. A Holm helper is
provided, but does not replace a declared multiple-testing protocol. Deflated
Sharpe, PBO and Hansen SPA are reported in the benchmark summary over the
retained method × seed paths (see [docs/EXTENSIONS.md](docs/EXTENSIONS.md)); they
do not correct the genetic search inside each fit. Broad baseline replication and
the papers' 50-seed experiments are not claimed. The stock/FX benchmark separately defines a suite of 54 paired tests
across its six assets, three seeds and three method pairs, and applies Holm to
that family. Its stress scenario reuses the same frozen strategies with doubled
assumed costs; historical broker swap data are unavailable.
