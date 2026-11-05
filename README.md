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
bash scripts/setup-environment.sh --core
source .venv/bin/activate
python environment/verify.py --core
gen-trade demo --workers 3 --cross-market-features --output reports/three-method-comparison.json
pytest -q
```

The repository also offers the optional full research environment:
Freqtrade/DEAP/Optuna, statistics and plotting tools, Polars/Arrow/DuckDB,
Rust 1.90.0 with tested PyO3/Rayon integration, and PostgreSQL 17 / Redis 7.4
service configuration. See [environment setup](docs/ENVIRONMENT.md) for prerequisites,
version pins, persistent storage, and verification details.

```bash
cd gen-trade
bash scripts/setup-environment.sh
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
is wanted. Python fallback always remains available. The prepared cloud install
script handles this order. Native and Python DC outputs are checked for parity.

## Supply real data

The daily stock/FX benchmark uses AAPL, GM, GRPN, EUR/USD, GBP/USD and USD/JPY
from 2015–2025. It tests the seven complete years 2019–2025 with calendar-based
training, validation and refitting. It downloads and hashes real observations,
handles stock splits/dividends and FX bid/ask accounting, and exports complete
trade/equity ledgers under primary and doubled-cost scenarios.

```bash
bash scripts/setup-environment.sh --market-data
source .venv/bin/activate
gen-trade fetch-data --directory data/stocks-fx-2015-2025
gen-trade benchmark --population 64 --generations 20 --seeds 11 22 33 \
  --workers 3 --output reports/stocks-fx-2015-2025
python scripts/verify-benchmark.py reports/stocks-fx-2015-2025
```

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

`lockbox.json` holds out test years from 2026 (and any listed instruments); the
benchmark refuses them unless `--unlock-lockbox` runs a profile that still matches
its frozen hash (`gen-trade freeze-profiles`).

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
