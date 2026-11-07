"""Progress reporting observes runs without changing them."""
from __future__ import annotations

import io
import json
import os
import socket

import pytest

from gen_trade.cli import main
from gen_trade.data import ChromosomeContext
from gen_trade.progress import (LOG_FILE, SCHEMA, STATUS_FILE, ProgressTracker, format_duration, read_status,
                                render_overview, render_status)
from gen_trade.runner import compare_methods, synthetic_markets

RUNTIME_FIELDS = ("worker_pid", "elapsed_seconds")


def _strip_runtime(report):
    for fold in report["folds"]:
        for method in fold["methods"].values():
            for field in RUNTIME_FIELDS:
                method.pop(field, None)
            for trial in method.get("selection", {}).get("trials", []):
                trial.get("training", {}).pop("runtime_seconds", None)
    return report


def _comparable(report):
    return {(fold["index"], fold["base_seed"], name): {key: method[key] for key in ("parameters", "selection", "statistics")}
            for fold in report["folds"] for name, method in fold["methods"].items()}


def test_format_duration():
    assert format_duration(None) == "?"
    assert format_duration(5.4) == "5s"
    assert format_duration(125) == "2m05s"
    assert format_duration(3 * 3600 + 61) == "3h01m"


def test_tracker_counts_eta_and_status_files(tmp_path):
    stream, out = io.StringIO(), io.StringIO()
    tracker = ProgressTracker(mode="log", stream=stream, out=out, command="gen-trade benchmark")
    for asset in ("AAPL", "GM", "GRPN"):
        tracker.plan(asset, 4)
    tracker.attach(tmp_path, ["AAPL", "GM", "GRPN"])
    tracker.resume("AAPL")
    tracker.begin("GM", 4)
    for index in range(2):
        tracker.job_done({"asset": "GM", "test_start": "2021-01-04", "seed": 11, "method": "moo3"}, 1.5)
    tracker.note("Running GM")
    data = json.loads((tmp_path / STATUS_FILE).read_text())
    assert data["schema"] == SCHEMA and data["status"] == "running"
    assert data["jobs"] == {**data["jobs"], "done": 2, "resumed": 4, "total": 12}
    assert data["completed"] == ["AAPL"] and data["requested"] == ["AAPL", "GM", "GRPN"]
    assert data["assets"]["GM"] == {"status": "running", "jobs_done": 2, "jobs_total": 4}
    assert data["eta_seconds"] is not None and data["eta_seconds"] >= 0
    assert "[ 6/12  50%] GM 2021-01-04 seed 11 moo3 · 1.5s" in stream.getvalue()
    assert out.getvalue() == "Running GM\n"
    tracker.finish("completed")
    data = json.loads((tmp_path / STATUS_FILE).read_text())
    assert data["status"] == "completed" and data["current"] is None
    log = (tmp_path / LOG_FILE).read_text()
    assert "=== run started" in log and "Running GM" in log and "Progress: completed" in log


def test_off_mode_prints_only_ordinary_messages():
    stream, out = io.StringIO(), io.StringIO()
    tracker = ProgressTracker(mode="off", stream=stream, out=out)
    tracker.begin("BTC/USDT", 1)
    tracker.job_done({"asset": "BTC/USDT", "method": "moo3"}, 0.1)
    tracker.note("hello")
    tracker.finish()
    assert stream.getvalue() == "" and out.getvalue() == "hello\n"


def test_bar_mode_redraws_one_line():
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    stream = Terminal()
    tracker = ProgressTracker(mode="auto", stream=stream, out=io.StringIO())
    assert tracker.mode == "bar"
    tracker.begin("EUR/USD", 2)
    tracker.job_done({"asset": "EUR/USD", "method": "mstgam", "seed": 1}, 2.0)
    tracker.finish()
    text = stream.getvalue()
    assert "\r\033[K" in text and "1/2 jobs" in text and "Progress: completed" in text


@pytest.mark.parametrize("workers", [1, 3])
def test_progress_sees_every_job_and_leaves_reports_unchanged(workers):
    markets = synthetic_markets(140, seed=5)
    options = dict(train_bars=60, validation_bars=15, test_bars=25, max_folds=1, population_size=6,
                   generations=1, seeds=(3, 4), bootstrap_reps=100, workers=workers)
    tracker = ProgressTracker(mode="off", stream=io.StringIO(), out=io.StringIO())
    observed = compare_methods(markets, ChromosomeContext("BTC/USDT"), progress=tracker, **options)
    plain = compare_methods(markets, ChromosomeContext("BTC/USDT"), **options)
    tracker.finish()
    assert tracker.done == tracker.total == 6 and tracker.failed == 0
    assert {(job["seed"], job["method"]) for job in tracker.recent} == {
        (seed, method) for seed in (3, 4) for method in ("moo3", "mstgam", "igtsp_ring")}
    assert _comparable(_strip_runtime(observed)) == _comparable(_strip_runtime(plain))


def test_pooled_selection_counts_both_phases():
    tracker = ProgressTracker(mode="off", stream=io.StringIO(), out=io.StringIO())
    compare_methods(synthetic_markets(140, seed=5), ChromosomeContext("BTC/USDT"), train_bars=60, validation_bars=15,
                    test_bars=25, max_folds=1, population_size=6, generations=1, seeds=(3, 4), bootstrap_reps=100,
                    workers=1, selection="pooled", deploy="validated", progress=tracker)
    assert tracker.done == tracker.total == 12
    assert {job["phase"] for job in tracker.recent} == {"validate", "deploy"} or tracker.recent[-1]["phase"] == "deploy"


def test_status_detects_stopped_runs_and_reads_old_format(tmp_path):
    stopped = tmp_path / "stopped"
    stopped.mkdir()
    (stopped / STATUS_FILE).write_text(json.dumps({
        "schema": SCHEMA, "status": "running", "pid": 2 ** 22 + 12345, "host": socket.gethostname(),
        "jobs": {"done": 3, "resumed": 0, "total": 10}, "elapsed_seconds": 30, "eta_seconds": 70,
        "assets": {"AAPL": {"status": "running", "jobs_done": 3, "jobs_total": 10}}, "recent": [],
        "command": "gen-trade benchmark --output x"}))
    data = read_status(stopped)
    assert data["state"] == "stopped"
    assert "Rerun the same command to resume" in render_status(stopped, data)
    old = tmp_path / "old"
    old.mkdir()
    (old / STATUS_FILE).write_text(json.dumps({"completed": ["AAPL"], "requested": ["AAPL", "GM"],
                                               "elapsed_seconds": 12.0}))
    assert read_status(old)["state"] == "incomplete"
    (old / "summary.json").write_text("{}")
    assert read_status(old)["state"] == "completed"
    overview = render_overview([stopped, old])
    assert "stopped" in overview and "1/2 assets" in overview


def test_live_run_is_running(tmp_path):
    tracker = ProgressTracker(mode="off", stream=io.StringIO(), out=io.StringIO(), directory=tmp_path)
    tracker.plan("AAPL", 2)
    assert read_status(tmp_path)["state"] == "running"
    assert json.loads((tmp_path / STATUS_FILE).read_text())["pid"] == os.getpid()
    tracker.finish("interrupted", error="KeyboardInterrupt: ")
    assert read_status(tmp_path)["state"] == "interrupted"


def test_status_command(tmp_path, capsys):
    run = tmp_path / "run"
    tracker = ProgressTracker(mode="off", stream=io.StringIO(), out=io.StringIO(), directory=run)
    tracker.begin("AAPL", 1)
    tracker.job_done({"asset": "AAPL", "seed": 1, "method": "moo3"}, 1.0)
    tracker.finish()
    assert main(["status", str(run)]) == 0
    assert "completed" in capsys.readouterr().out
    ProgressTracker(mode="off", stream=io.StringIO(), out=io.StringIO(), directory=tmp_path / "second").finish()
    assert main(["status", "--root", str(tmp_path)]) == 0
    assert "STATUS" in capsys.readouterr().out
    assert main(["status", str(tmp_path / "missing")]) == 2
