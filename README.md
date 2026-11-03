# gen-trade

Research implementations of two directional-change trading papers, with a causal
simulator, optional Rust acceleration, chronological evaluation, and statistical
reports. These are **documented adaptations**, not reproductions of the authors'
published numerical results. No live trading is implemented.

| Method | Representation and search | Training objective |
| --- | --- | --- |
| **MOO3** — [Long et al.](https://doi.org/10.1007/s10462-025-11390-9) | Typed trading-rule trees; NSGA-II; 28 DC + 28 technical indicators | Maximize paper total return and mean trade return; minimize trade-return risk; select the final training Pareto front by modified Sharpe |
| **MSTGAM** — [Salman et al.](https://doi.org/10.1007/s10462-025-11419-z) | 70 normalized weights over eight DC strategies and ten thresholds | Maximize the paper's trade-based Sharpe score |

Read [the source mapping, equations, and adaptation notes](docs/PAPER_METHODS.md)
before using either method. The papers use different fees and different return
statistics; their scores must not be compared as ordinary annualized Sharpe.

## Complete development environment

The repository includes the full research environment: Python 3.12.14,
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
training-only selections, held-out fold statistics, paired dependent-data
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

CSV or Parquet must contain `symbol`, timezone-aware `timestamp` (bar open),
`available_at` (when full candle data became known), and `open,high,low,close,volume`.
Rows per symbol must be strictly chronological, unique, and nonoverlapping.
Prices must be finite and positive; missing volume is allowed. Delayed/revised or
overlapping feeds need an ingestion adapter before use. No timestamps, market
calendars, or missing prices are inferred silently.

```bash
gen-trade compare data/markets.parquet --pair BTC/USDT \
  --periods-per-year 105120 --train-bars 50000 --test-bars 10000 \
  --embargo-bars 1500 --population 100 --generations 20 \
  --cross-market-features --fee 0.001 --slippage-bps 2 \
  --output reports/real-data-comparison.json
```

`105120` is an explicit **365 × 24 × 12** convention for five-minute 24/7 data;
choose the convention appropriate to your instrument, calendar, and observations.
Population/generation defaults are small for a smoke run. The method classes
retain the paper population/generation defaults. Equal population and generations
give equal evolutionary settings, not equal total compute: MSTGAM also calibrates
its sub-strategies before evolving weights. Diagnostics retain unique evaluation
counts. Select embargo length to cover the strategy's information/holding horizon.
The runner provides chronological rolling folds, not a full purged combinatorial
cross-validation implementation or the papers' parameter-tuning protocol.

## Chromosomes and external markets

`ChromosomeContext` declares the trading pair, allowed UTC entry hours, and typed
`FeatureGene` references to OHLCV sources with lag, rolling window, and maximum
feature age. Cross-market features use backward as-of joins on availability time,
preserve warm-up/missing values, and never use future observations.

MOO3 can evolve rules referencing those external terminals. MSTGAM deliberately
keeps its published 70-weight representation. Pair/session choices are explicit
experiment context in this version; they are not automatically evolved genes.
Extending both representations to jointly evolve pairs and entry sessions remains
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

## Statistical scope

Reports include net equity returns, annualization under an explicit sampling
convention, drawdown and duration, trade/tail statistics, stationary-block
bootstrap confidence intervals, and HAC mean-return inference. Paired comparisons
require identical held-out timestamps. Undefined/degenerate values stay `null`.
Nominal p-values are labeled unadjusted; one run does not account for evolutionary
selection, many strategy trials, many assets, or dependent folds. A Holm helper is
provided, but does not replace a declared multiple-testing protocol. Deflated
Sharpe, PBO, broad baseline replication, real-data stress tests, and the papers'
50-seed experiments are not claimed.
