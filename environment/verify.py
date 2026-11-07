"""Verify the reproducible research stack; optionally check local services.

All data generated here is synthetic. No exchange credentials, live market data,
or trading orders are used. Connection errors are reported without credential
values or raw driver messages.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid


PACKAGES = {
    "freqtrade": "freqtrade", "deap": "deap", "optuna": "optuna",
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "statsmodels": "statsmodels", "arch": "arch", "polars": "polars",
    "pyarrow": "pyarrow", "duckdb": "duckdb", "numba": "numba",
    "scikit-learn": "sklearn", "sqlalchemy": "sqlalchemy",
    "pydantic": "pydantic", "pandera": "pandera", "matplotlib": "matplotlib",
    "seaborn": "seaborn", "plotly": "plotly", "jupyterlab": "jupyterlab",
    "psycopg": "psycopg", "redis": "redis", "maturin": "maturin",
    "TA-Lib":"talib", "threadpoolctl":"threadpoolctl",
}
CORE_PACKAGES = {"deap", "numpy", "pandas", "scipy", "statsmodels", "arch", "pyarrow", "TA-Lib", "threadpoolctl"}


class CheckFailed(RuntimeError):
    """Only predefined, nonsecret messages belong in this exception."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def verify_toolchains(core: bool = False) -> dict:
    pinned = (Path(__file__).resolve().parents[1] / ".python-version").read_text().strip()
    require(sys.version.split()[0] == pinned, f"Pinned Python {pinned} is required")
    if core:
        return {"python":sys.version.split()[0], "rust":"optional; core profile uses the Python DC backend if native is absent"}
    result = subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True)
    rust_version = result.stdout.split()[1]
    require(rust_version == "1.90.0", "Rust 1.90.0 is required")
    return {"python": sys.version.split()[0], "rust": rust_version}


def verify_packages(core: bool = False) -> dict:
    installed = {}
    for distribution, module in PACKAGES.items():
        if core and distribution not in CORE_PACKAGES:
            continue
        importlib.import_module(module)
        installed[distribution] = version(distribution)
    return installed


def verify_core_computation() -> dict:
    import numpy as np
    import pandas as pd
    from gen_trade.data import ChromosomeContext
    from gen_trade.runner import compare_methods, synthetic_markets

    markets = synthetic_markets(120, seed=21)
    with TemporaryDirectory(prefix="gen-trade-core-") as directory:
        path = Path(directory) / "bars.parquet"
        markets["BTC/USDT"].to_parquet(path, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(path), markets["BTC/USDT"])
    report = compare_methods(markets, ChromosomeContext("BTC/USDT"), train_bars=65,
                             validation_bars=0, test_bars=25, population_size=8,
                             generations=1, workers=3, max_folds=1, bootstrap_reps=100)
    methods = report["folds"][0]["methods"]
    require(set(methods) == {"moo3", "mstgam", "igtsp_ring"}, "Three-framework execution failed")
    require(all(np.isfinite(m["statistics"]["net_total_return"]) for m in methods.values()),
            "Nonfinite held-out statistics")
    checks = ["Parquet round-trip", "three spawned-process framework fits and next-open backtests",
              "finite net equity statistics", "stationary bootstrap/HAC", "all three paired comparisons"]
    from gen_trade import dc
    backend = "rust" if dc._native_dc_arrays is not None else "python (Rust backend not installed)"
    if dc._native_dc_arrays is not None:
        verify_dc_parity()
        checks.append("Python/Rust DC parity")
    return {"data":"synthetic dependency checks only", "methods":list(methods), "dc_backend":backend,
            "worker_processes":len({m["worker_pid"] for m in methods.values()}), "checks":checks}


def verify_dc_parity() -> None:
    import numpy as np
    from gen_trade.dc import DCProfile, directional_changes

    prices = np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, 512)))
    for threshold in (0.005, 0.01, 0.02):
        python_profile = directional_changes(prices, threshold, backend="python")
        rust_profile = directional_changes(prices, threshold, backend="rust")
        for field in fields(DCProfile):
            if field.name != "threshold":
                np.testing.assert_array_equal(getattr(python_profile, field.name),
                                              getattr(rust_profile, field.name))


def verify_computation() -> dict:
    import duckdb
    import gen_trade_env_native
    import numba
    import numpy as np
    import polars as pl
    import scipy.stats as stats
    import statsmodels.api as sm
    from arch.bootstrap import StationaryBootstrap

    rng = np.random.default_rng(42)
    values = rng.normal(0, 0.01, 128)
    frame = pl.DataFrame({"symbol": ["SYNTHETIC"] * len(values), "value": values})
    require(frame.to_arrow().num_rows == len(values), "Arrow conversion failed")
    with TemporaryDirectory(prefix="gen-trade-verify-") as directory:
        path = Path(directory) / "synthetic.parquet"
        frame.write_parquet(path)
        restored = pl.read_parquet(path)
        require(restored.equals(frame), "Polars Parquet round-trip failed")
        with duckdb.connect() as db:
            count, total = db.execute(
                "SELECT count(*), sum(value) FROM read_parquet(?)", [str(path)]
            ).fetchone()
        require(count == len(values) and np.isclose(total, values.sum()),
                "DuckDB Parquet query failed")

    @numba.njit(cache=False)
    def sum_squares(array):
        return np.sum(array * array)

    expected = float(np.dot(values, values))
    require(bool(np.isclose(sum_squares(values), expected, rtol=1e-12)), "Numba JIT failed")
    require(bool(np.isclose(gen_trade_env_native.sum_squares(values.tolist()), expected, rtol=1e-12)),
            "PyO3/Rayon numerical check failed")
    verify_dc_parity()

    bootstrap = StationaryBootstrap(8, values, seed=42)
    interval = bootstrap.conf_int(np.mean, reps=100, method="percentile")
    require(interval.shape == (2, 1) and np.isfinite(interval).all(), "Block bootstrap failed")
    fit = sm.OLS(values, np.ones((len(values), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    require(bool(np.isfinite(fit.bse).all()), "HAC statistics failed")
    require(bool(np.isfinite(stats.jarque_bera(values).pvalue)), "SciPy statistics failed")
    return {
        "data": "synthetic dependency checks only",
        "rows": len(values),
        "checks": ["Polars/Arrow/Parquet/DuckDB", "Numba JIT", "Python/Rust DC parity",
                   "PyO3/Rayon numerical parity", "stationary bootstrap", "HAC", "SciPy"],
    }


def verify_services() -> dict:
    import psycopg
    import redis

    database_url = os.getenv("DATABASE_URL", "postgresql://gen_trade@127.0.0.1:55432/gen_trade")
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:56379/0")
    with psycopg.connect(database_url, connect_timeout=5) as db:
        server_version = int(db.execute("SHOW server_version_num").fetchone()[0])
        require(server_version // 10000 == 17, "PostgreSQL major version 17 is required")
        db.execute("CREATE TEMP TABLE environment_verify (value integer NOT NULL)")
        db.execute("INSERT INTO environment_verify VALUES (%s)", (42,))
        require(db.execute("SELECT value FROM environment_verify").fetchone()[0] == 42,
                "PostgreSQL transactional query failed")
        db.rollback()

    cache = redis.Redis.from_url(redis_url, decode_responses=True,
                                 socket_connect_timeout=5, socket_timeout=5)
    redis_version = cache.info("server")["redis_version"]
    require(redis_version.startswith("7.4."), "Redis 7.4 is required")
    key = "gen-trade:environment-verify:" + uuid.uuid4().hex
    try:
        require(bool(cache.set(key, "verified", ex=30)), "Redis cache write failed")
        require(cache.get(key) == "verified" and 0 < cache.ttl(key) <= 30,
                "Redis cache read/TTL failed")
    finally:
        cache.delete(key)
    return {"postgresql_version_num": server_version, "redis_version": redis_version,
            "checks": ["PostgreSQL temporary table/write/read/rollback", "Redis write/read/TTL/delete"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--services", action="store_true", help="Also verify configured PostgreSQL/Redis")
    parser.add_argument("--core", action="store_true", help="Verify the lean three-framework stack; Rust/research extras optional")
    args = parser.parse_args(argv)
    report = {"status": "checking"}
    stage = "toolchains"
    try:
        report[stage] = verify_toolchains(core=args.core)
        stage = "packages"
        report[stage] = verify_packages(core=args.core)
        stage = "computation"
        report[stage] = verify_core_computation() if args.core else verify_computation()
        if args.services:
            stage = "services"
            report[stage] = verify_services()
        else:
            report["services"] = "not requested; run with --services to check running services"
        report["status"] = "passed"
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        # A dependency/driver error may embed a DSN. Never emit raw error text.
        report.update(status="failed", failed_stage=stage, error_type=type(error).__name__)
        if isinstance(error, CheckFailed):
            report["reason"] = str(error)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
