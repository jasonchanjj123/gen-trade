"""Time Python-vs-Rust execute_batch calls, including conversions; not a trading benchmark."""
import json
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd

from gen_trade.backtest import ExecutionConfig
from gen_trade.instruments import benchmark_spec
from gen_trade.market_execution import MarketArrays, execute_batch


def main():
    seed, bars, repeats = 42, 750, 3
    rng = np.random.default_rng(seed)
    opens = pd.date_range("2020-01-02", periods=bars, freq="B", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, bars)))
    open_ = np.concatenate(([100.0], close[:-1]))
    frame = pd.DataFrame({"timestamp":opens, "available_at":opens + pd.Timedelta(hours=7), "open":open_,
                          "high":np.maximum(open_, close) * 1.01, "low":np.minimum(open_, close) * 0.99,
                          "close":close, "volume":1000.0})
    spec = benchmark_spec("AAPL")
    config = ExecutionConfig(initial_cash=100_000, fee=0.0, slippage_bps=1.0, instrument=spec)
    market = MarketArrays.from_bars(frame, spec)
    results = []
    for strategies in (1, 32, 512):
        signals = rng.choice([-1, 0, 1], size=(bars, strategies), p=[0.05, 0.9, 0.05])
        outputs = {engine:execute_batch(market, signals, config, backend=engine) for engine in ("python", "rust")}
        np.testing.assert_array_equal(outputs["python"].terminal_equity, outputs["rust"].terminal_equity)
        seconds = {}
        for engine in outputs:
            times = []
            for _ in range(repeats):
                start = perf_counter()
                execute_batch(market, signals, config, backend=engine)
                times.append(perf_counter() - start)
            seconds[engine] = median(times)
        results.append({"strategies":strategies, "median_seconds":seconds,
                        "python_over_rust_ratio":seconds["python"] / seconds["rust"]})
    print(json.dumps({"scope":"execute_batch on daily stock bars, including Python/native conversions",
                      "seed":seed, "bars":bars, "repeats":repeats, "results":results,
                      "limitation":"Machine-specific measurement, not total benchmark speed or trading performance."},
                     indent=2))


if __name__ == "__main__":
    main()
