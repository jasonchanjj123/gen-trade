"""Measure full Python-vs-Rust DC calls, including conversions; not a trading benchmark."""
from dataclasses import fields
import json
from statistics import median
from time import perf_counter

import numpy as np

from gen_trade.dc import DCProfile, directional_changes


def main():
    seed, observations, repeats = 42, 100_000, 5
    values = np.exp(np.cumsum(np.random.default_rng(seed).normal(0, .001, observations)))
    outputs = {backend:directional_changes(values,.002,backend=backend) for backend in ("python","rust")}
    for field in fields(DCProfile):
        if field.name != "threshold":
            np.testing.assert_array_equal(getattr(outputs["python"],field.name),getattr(outputs["rust"],field.name))
    seconds = {}
    for backend in outputs:
        times = []
        for _ in range(repeats):
            start = perf_counter()
            directional_changes(values,.002,backend=backend)
            times.append(perf_counter()-start)
        seconds[backend] = median(times)
    print(json.dumps({"scope":"single DC profile including Python/native conversions",
                      "seed":seed,"observations":observations,"repeats":repeats,
                      "median_seconds":seconds,"python_over_rust_ratio":seconds["python"]/seconds["rust"],
                      "limitation":"Machine-specific measurement, not total optimizer speed or trading performance."},indent=2))


if __name__ == "__main__":
    main()
