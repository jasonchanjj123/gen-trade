"""Verify a completed benchmark's provenance, chronology and account ledgers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from gen_trade.benchmark import _code_hash
from gen_trade.experiments import METHOD_NAMES
from gen_trade.instruments import benchmark_spec
from gen_trade.market_data import load_dataset


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(report_directory, data_directory):
    directory = Path(report_directory)
    summary = json.loads((directory / "summary.json").read_text())
    markets, manifest = load_dataset(data_directory)
    settings = summary["settings"]
    source = _code_hash()
    require(summary["source_sha256"] == source, "Current source/lock differs from benchmark")
    require(summary["dataset_manifest"] == manifest, "Dataset manifest differs from benchmark")
    for name in ("report.md", "equity.png"):
        require((directory / name).stat().st_size > 0, f"Missing/empty {name}")
    paths = trades = methods = fallbacks = 0
    maximum_error = 0.0
    family_keys = set()
    for symbol in summary["universe"]:
        asset = json.loads((directory / (symbol.replace("/", "_") + ".json")).read_text())
        signature = hashlib.sha256(json.dumps({
            "settings": settings, "data": manifest["instruments"][symbol]["sha256"],
            "source": source, "instrument": asdict(benchmark_spec(symbol)),
        }, sort_keys=True).encode()).hexdigest()
        require(asset["benchmark_signature"] == signature, f"{symbol}: checkpoint signature differs")
        require(asset["data_provenance"] == manifest["instruments"][symbol], f"{symbol}: data provenance differs")
        require(asset["protocol"]["instrument"] == asdict(benchmark_spec(symbol)), f"{symbol}: primary cost profile differs")
        seen = set()
        bars = markets[symbol]
        for fold in asset["folds"]:
            bounds = fold["bounds"]
            year = int(bars.iloc[bounds["test_start"]]["session_date"][:4])
            key = (fold["base_seed"], year)
            require(key not in seen, f"{symbol}: duplicate seed/year")
            seen.add(key)
            require(bounds["validation_start"] - bounds["train_stop"] == settings["embargo_bars"], "Training embargo differs")
            require(bounds["test_start"] - bounds["validation_stop"] == settings["embargo_bars"], "Test embargo differs")
            require(pd.Timestamp(fold["train_last_known_at"]) < pd.Timestamp(fold["validation_first_open"]), "Training reaches validation")
            require(pd.Timestamp(fold["refit_last_known_at"]) < pd.Timestamp(fold["test_first_open"]), "Refit reaches test")
            expected_times = [t.isoformat() for t in bars.iloc[bounds["test_start"]:bounds["test_stop"]]["available_at"]]
            cases = [fold["baseline_backtest"], fold["baseline_cost_stress"]["backtest"]]
            require(set(fold["methods"]) == set(METHOD_NAMES), f"{symbol}: missing method")
            for name, method in fold["methods"].items():
                methods += 1
                selection = method["selection"]
                require(selection["test_used_for_selection"] is False, "Test-based selection reported")
                require(len(selection["trials"]) == len(settings["candidate_configurations"][name]), "Candidate trial omitted")
                stress = method["cost_stress"]
                require(stress["frozen_model"] is True, "Stress model was not frozen")
                require(stress["execution"]["instrument"] == asdict(benchmark_spec(symbol, stress=True)), "Stress cost profile differs")
                require(stress["execution"]["slippage_bps"] == 2 * asset["protocol"]["slippage_bps"], "Stress slippage differs")
                if method.get("status") == "cash_fallback_no_eligible_configuration":
                    fallbacks += 1
                    require(not method["backtest"]["trades"], "Cash fallback contains trades")
                else:
                    require(selection["refit_rows"] == bounds["validation_stop"] - bounds["train_start"], "Refit length differs")
                cases.extend([method["backtest"], method["cost_stress"]["backtest"]])
            for case in cases:
                paths += 1
                trades += len(case["trades"])
                require(case["timestamps"] == expected_times, f"{symbol}: held-out timestamps differ")
                require(len(case["equity"]) == len(expected_times) + 1, "Equity path length differs")
                require(all(math.isfinite(value) and value > 0 for value in case["equity"]), "Nonpositive/nonfinite equity")
                delta = case["equity"][-1] - case["initial_cash"]
                error = abs(delta - math.fsum(t["cash_pnl"] for t in case["trades"]))
                maximum_error = max(maximum_error, error)
                require(error < 1e-6, f"{symbol}: trade P&L does not reconcile")
                for field, total in (("fees", "commissions"), ("financing", "financing"), ("dividend_income", "dividends")):
                    require(math.isclose(math.fsum(t[field] for t in case["trades"]), case["ledger"][total], abs_tol=1e-6), f"{symbol}: {total} does not reconcile")
                for trade in case["trades"]:
                    require(trade["entry_time"] <= trade["exit_time"], "Trade exits before entry")
                    require(trade["account_currency"] == "USD", "Unexpected account currency")
        expected = {(seed, year) for seed in settings["seeds"]
                    for year in range(settings["first_test_year"], settings["last_test_year"] + 1)}
        require(seen == expected, f"{symbol}: a requested seed/year is absent")
        family_keys.update((symbol, seed, first, second) for seed in settings["seeds"]
                           for index, first in enumerate(METHOD_NAMES) for second in METHOD_NAMES[index + 1:])
    actual_keys = {(row["symbol"], row["seed"], row["first"], row["second"]) for row in summary["paired_comparisons"]}
    require(actual_keys == family_keys and len(actual_keys) == len(summary["paired_comparisons"]), "Paired-test family differs")
    require(all(row["suite_family_size"] == len(family_keys) for row in summary["paired_comparisons"]), "Multiple-test family size differs")
    require(summary["method_runs"] == methods, "Method-run count differs")
    require(summary["cash_fallback_method_folds"] == fallbacks, "Cash-fallback count differs")
    expected_rows = {(symbol, seed, name) for symbol in summary["universe"]
                     for seed in settings["seeds"] for name in (*METHOD_NAMES, "buy_and_hold")}
    rows = {(row["symbol"], row["seed"], row["method"]) for row in summary["runs"]}
    require(rows == expected_rows and len(rows) == len(summary["runs"]), "Summary omits/repeats a declared result")
    return {"status": "passed", "source_sha256": source, "instruments": len(summary["universe"]),
            "method_runs": methods, "primary_and_stress_paths": paths, "trade_ledger_rows": trades,
            "paired_test_family_size": len(family_keys), "cash_fallback_method_folds": fallbacks,
            "maximum_cash_reconciliation_error_usd": maximum_error,
            "checks": ["current code/lock and dataset hashes", "all declared instruments/seeds/years/methods",
                       "train/validation/refit/test chronology and embargoes", "identical held-out timestamps",
                       "cash and cost ledger conservation", "summary and paired-test family completeness"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory", nargs="?", default="reports/stocks-fx-2015-2025")
    parser.add_argument("--data-directory", default="data/stocks-fx-2015-2025")
    args = parser.parse_args()
    result = verify(args.report_directory, args.data_directory)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (Path(args.report_directory) / "verification.json").write_text(output)
    print(output, end="")
