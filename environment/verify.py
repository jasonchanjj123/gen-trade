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
}


class CheckFailed(RuntimeError):
    """Only predefined, nonsecret messages belong in this exception."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def verify_toolchains() -> dict:
    require(sys.version_info[:2] == (3, 12), "Python 3.12 is required")
    result = subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True)
    rust_version = result.stdout.split()[1]
    require(rust_version == "1.90.0", "Rust 1.90.0 is required")
    return {"python": sys.version.split()[0], "rust": rust_version}


def verify_packages() -> dict:
    installed = {}
    for distribution, module in PACKAGES.items():
        importlib.import_module(module)
        installed[distribution] = version(distribution)
    return installed


def verify_computation() -> dict:
    import duckdb
    import gen_trade_env_native
    import numba
    import numpy as np
    import polars as pl
    import scipy.stats as stats
    import statsmodels.api as sm
    from arch.bootstrap import StationaryBootstrap
    from gen_trade.dc import DCProfile, directional_changes

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
    prices = np.exp(np.cumsum(values))
    python_profile = directional_changes(prices, 0.01, backend="python")
    rust_profile = directional_changes(prices, 0.01, backend="rust")
    for field in fields(DCProfile):
        if field.name != "threshold":
            np.testing.assert_array_equal(getattr(python_profile, field.name),
                                          getattr(rust_profile, field.name))

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
    args = parser.parse_args(argv)
    report = {"status": "checking"}
    stage = "toolchains"
    try:
        report[stage] = verify_toolchains()
        stage = "packages"
        report[stage] = verify_packages()
        stage = "computation"
        report[stage] = verify_computation()
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
