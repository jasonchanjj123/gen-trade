"""Report what is installed in this checkout and how to fix gaps.

Standard library only, so it also runs when .venv is missing or broken:
    bash scripts/doctor.sh            # or: python3 environment/doctor.py [--json]
It reads state and never installs, starts or changes anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(os.environ.get("GEN_TRADE_TOOLS_DIR", ROOT / ".tools"))
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
EXTRAS = {  # extra -> modules that prove it is installed
    "core": ["numpy", "pandas", "scipy", "statsmodels", "arch", "deap", "talib", "pyarrow", "threadpoolctl"],
    "dev": ["pytest", "hypothesis", "maturin"],
    "storage": ["psycopg", "redis"],
    "market-data": ["exchange_calendars", "matplotlib"],
    "research": ["freqtrade", "optuna", "polars", "duckdb", "numba", "sklearn", "jupyterlab"],
}
PROFILE_EXTRAS = {"core": ["dev", "storage"], "market-data": ["dev", "storage", "market-data"], "full": None}
OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"


def run(command, timeout=30, **kwargs):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
        return result.returncode, (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, type(error).__name__


def check_uv(state):
    if not shutil.which("uv"):
        return FAIL, "not installed", "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    code, text = run(["uv", "--version"])
    if code != 0:
        return FAIL, "installed but does not run", "Reinstall uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    return OK, text.split()[1], None


def check_venv(state):
    pin = (ROOT / ".python-version").read_text().strip()
    if not VENV_PYTHON.exists():
        return FAIL, "no .venv", "Create it: bash scripts/setup-environment.sh --market-data"
    code, text = run([str(VENV_PYTHON), "-c", "import platform; print(platform.python_version())"])
    if code != 0:
        return FAIL, ".venv python does not start", "Recreate it: rm -rf .venv && bash scripts/setup-environment.sh"
    if text != pin:
        return FAIL, f"{text} at .venv, but the pin is {pin}", "Recreate it: rm -rf .venv && bash scripts/setup-environment.sh"
    state["venv_ok"] = True
    return OK, f"{text} at .venv (pin {pin})", None


def check_setup_record(state):
    path = TOOLS / "environment.json"
    if not path.exists():
        return WARN, "no record of a completed setup in this checkout", \
            "Run: bash scripts/setup-environment.sh --market-data (or --core / --full)"
    try:
        record = json.loads(path.read_text())
    except ValueError:
        return WARN, "unreadable .tools/environment.json", "Rerun bash scripts/setup-environment.sh"
    state["record"] = record
    when = record.get("completed_at", "?").replace("T", " ").replace("Z", " UTC")
    return OK, f"profile {record.get('profile')}, Rust backend {record.get('native')} (set up {when})", None


def check_lock(state):
    record = state.get("record")
    if not state.get("venv_ok") or not record or not shutil.which("uv"):
        return INFO, "skipped (needs uv, .venv and a setup record)", None
    extras = PROFILE_EXTRAS.get(record.get("profile"))
    flags = ["--all-extras"] if extras is None else [arg for extra in extras for arg in ("--extra", extra)]
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    environment.update(UV_PROJECT_ENVIRONMENT=str(ROOT / ".venv"),
                       UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", str(TOOLS / "cache" / "uv")))
    code, text = run(["uv", "sync", "--locked", "--dry-run", "--offline", *flags], cwd=ROOT, env=environment, timeout=60)
    if code != 0:
        first = next((line for line in text.splitlines() if line.strip()), text)
        return WARN, f"could not compare with uv.lock: {first[:120]}", "Rerun bash scripts/setup-environment.sh"
    changes = [line.strip() for line in text.splitlines()
               if line.strip().startswith(("+", "-", "~")) and "gen-trade-native" not in line
               and "gen-trade-env-native" not in line]
    if changes:
        shown = ", ".join(changes[:4]) + (" …" if len(changes) > 4 else "")
        return WARN, f".venv differs from uv.lock: {shown}", "Rerun bash scripts/setup-environment.sh to resync"
    return OK, ".venv matches uv.lock for this profile", None


def check_extras(state):
    if not state.get("venv_ok"):
        return INFO, "skipped (no working .venv)", None
    probe = ("import importlib.util, json, sys; extras=json.loads(sys.argv[1]); "
             "print(json.dumps({k:[m for m in v if importlib.util.find_spec(m) is None] for k,v in extras.items()}))")
    code, text = run([str(VENV_PYTHON), "-c", probe, json.dumps(EXTRAS)])
    if code != 0:
        return FAIL, "could not inspect packages", "Rerun bash scripts/setup-environment.sh"
    missing = json.loads(text)
    present = [name for name, modules in missing.items() if not modules]
    partial = {name: modules for name, modules in missing.items() if modules and len(modules) < len(EXTRAS[name])}
    absent = [name for name, modules in missing.items() if len(modules) == len(EXTRAS[name])]
    state["extras"] = present
    if missing["core"]:
        return FAIL, f"core packages missing: {', '.join(missing['core'])}", "Rerun bash scripts/setup-environment.sh"
    text = "installed: " + ", ".join(present)
    if absent:
        text += " · not installed: " + ", ".join(absent)
    if partial:
        text += " · incomplete: " + ", ".join(f"{k} (missing {', '.join(v)})" for k, v in partial.items())
        return WARN, text, "Rerun bash scripts/setup-environment.sh with the profile you need"
    return OK, text, None


def check_package(state):
    if not state.get("venv_ok"):
        return INFO, "skipped (no working .venv)", None
    probe = ("import gen_trade, json; from gen_trade import market_execution as e; "
             "print(json.dumps({'version':gen_trade.__version__,'native':e._native_execute_batch_arrays is not None,"
             "'path':gen_trade.__file__}))")
    code, text = run([str(VENV_PYTHON), "-c", probe], cwd="/")
    if code != 0:
        return FAIL, "gen_trade does not import from .venv", "Rerun bash scripts/setup-environment.sh"
    info = json.loads(text)
    if not Path(info["path"]).resolve().is_relative_to(ROOT):
        return WARN, f"gen_trade {info['version']} imports from {info['path']}, not this checkout", \
            "Rerun bash scripts/setup-environment.sh"
    state["native"] = info["native"]
    cli = ROOT / ".venv" / "bin" / "gen-trade"
    if not cli.exists():
        return WARN, f"gen_trade {info['version']} imports, but the gen-trade command is missing", \
            "Rerun bash scripts/setup-environment.sh"
    return OK, f"gen_trade {info['version']} and the gen-trade command", None


def check_backend(state):
    if "native" not in state:
        return INFO, "unknown", None
    if state["native"]:
        return OK, "Rust (gen_trade_native); identical results to the Python fallback", None
    return INFO, "Python fallback (identical results, benchmarks run about 5-10x slower)", \
        "Optional: bash scripts/setup-environment.sh --native"


def check_rust(state):
    record = state.get("record") or {}
    wanted = record.get("native") == "yes" or record.get("profile") == "full"
    saved = {}
    paths_file = TOOLS / "toolchain-paths.sh"
    if paths_file.exists():
        for line in paths_file.read_text().splitlines():
            key, _, value = line.partition("=")
            saved[key] = value.strip("'\"")
    environment = dict(os.environ)
    if saved.get("GEN_TRADE_SAVED_RUSTUP_HOME") and "RUSTUP_HOME" not in environment:
        environment["RUSTUP_HOME"] = saved["GEN_TRADE_SAVED_RUSTUP_HOME"]
    cargo_home = environment.get("CARGO_HOME") or saved.get("GEN_TRADE_SAVED_CARGO_HOME") or str(Path.home() / ".cargo")
    rustup = shutil.which("rustup") or shutil.which("rustup", path=str(Path(cargo_home) / "bin"))
    if not rustup:
        return (WARN if wanted else INFO), "rustup not found", \
            "Only needed for --native/--full; setup installs it" if not wanted else "Rerun bash scripts/setup-environment.sh"
    # Run outside the checkout: rust-toolchain.toml would otherwise trigger a download.
    environment.pop("RUSTUP_TOOLCHAIN", None)
    code, text = run([rustup, "toolchain", "list"], cwd="/", env=environment)
    if code == 0 and any(line.startswith("1.90.0") for line in text.splitlines()):
        return OK, "1.90.0 toolchain installed", None
    return (WARN if wanted else INFO), "pinned 1.90.0 toolchain not installed", \
        "Rerun bash scripts/setup-environment.sh --native" if wanted else "Only needed for --native/--full"


def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def check_docker(state):
    if not shutil.which("docker"):
        return INFO, "not installed (only for optional PostgreSQL/Redis; backtests don't need it)", None
    code, _ = run(["docker", "info"], timeout=10)
    if code != 0:
        return INFO, "installed, daemon not running", "Start Docker to use bash scripts/services.sh"
    pg = int(os.environ.get("GEN_TRADE_PG_PORT", 55432))
    redis = int(os.environ.get("GEN_TRADE_REDIS_PORT", 56379))
    up = [name for name, port in (("PostgreSQL", pg), ("Redis", redis)) if _port_open(port)]
    if up:
        return OK, f"running; listening: {', '.join(up)}", None
    return INFO, "running; services stopped", "Start them: bash scripts/services.sh up -d --wait --wait-timeout 60"


def check_data(state):
    root = ROOT / "data"
    if not root.exists():
        return INFO, "no data/ directory", "Download: gen-trade fetch-data --directory data/stocks-fx-2015-2025"
    found = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = directory / "manifest.json"
        if manifest.exists():
            try:
                count = len(json.loads(manifest.read_text()).get("instruments", {}))
                found.append(f"{directory.name} ({count} instruments)")
            except ValueError:
                found.append(f"{directory.name} (unreadable manifest)")
        else:
            found.append(f"{directory.name} (no manifest: incomplete download)")
    if not found:
        return INFO, "no datasets", "Download: gen-trade fetch-data --directory data/stocks-fx-2015-2025"
    return OK, ", ".join(found), None


def check_runs(state):
    reports = ROOT / "reports"
    if not reports.exists():
        return INFO, "no reports/ directory yet", None
    running, stopped, finished = [], [], 0
    for path in sorted(reports.glob("*/progress.json")):
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        if data.get("status") == "running":
            alive = True
            if data.get("host") == socket.gethostname():
                try:
                    os.kill(int(data.get("pid")), 0)
                except ProcessLookupError:
                    alive = False
                except (PermissionError, TypeError, ValueError, OSError):
                    pass
            (running if alive else stopped).append(path.parent.name)
        elif (path.parent / "summary.json").exists():
            finished += 1
    text = f"{len(running)} running" + (f" ({', '.join(running)})" if running else "")
    text += f" · {finished} finished under reports/"
    if stopped:
        return WARN, text + f" · stopped mid-run: {', '.join(stopped)}", \
            "Rerun the same benchmark command to resume; see gen-trade status"
    return (OK, text, "Follow them: gen-trade status --watch" if running else None)


def check_shell(state):
    if os.environ.get("VIRTUAL_ENV") == str(ROOT / ".venv"):
        return OK, "this shell has the environment activated", None
    return INFO, "not activated in this shell", "source scripts/activate.sh   (bash or zsh)"


def check_machine(state):
    cpus = os.cpu_count() or 1
    free = shutil.disk_usage(ROOT).free / 1e9
    text = f"{cpus} CPUs (benchmarks default to --workers 3) · {free:.0f} GB free disk"
    if free < 5:
        return WARN, text, "Free some disk space; reports with full ledgers are large"
    return INFO, text, None


CHECKS = [("uv", check_uv), ("Python venv", check_venv), ("Setup profile", check_setup_record),
          ("Locked packages", check_lock), ("Extras", check_extras), ("gen-trade", check_package),
          ("Native backend", check_backend), ("Rust", check_rust), ("Docker", check_docker),
          ("Data", check_data), ("Runs", check_runs), ("Shell", check_shell), ("Machine", check_machine)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)
    state, results = {}, []
    for name, check in CHECKS:
        try:
            status, detail, fix = check(state)
        except Exception as error:  # a broken check must not hide the others
            status, detail, fix = WARN, f"check failed: {type(error).__name__}", None
        results.append({"check": name, "status": status, "detail": detail, "fix": fix})
    failed = [row for row in results if row["status"] == FAIL]
    warned = [row for row in results if row["status"] == WARN]
    if args.json:
        print(json.dumps({"root": str(ROOT), "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "results": results, "ready": not failed}, indent=2))
        return 1 if failed else 0
    utf = "utf" in (sys.stdout.encoding or "").lower()
    marks = {OK: "✓" if utf else "ok", WARN: "!" , FAIL: "✗" if utf else "x", INFO: "·"}
    print(f"gen-trade environment doctor  ({ROOT})\n")
    width = max(len(row["check"]) for row in results)
    for row in results:
        print(f"  {marks[row['status']]} {row['check']:<{width}}  {row['detail']}")
        if row["fix"] and row["status"] != OK:
            print(f"    {'':<{width}}  → {row['fix']}")
        elif row["fix"] and row["check"] == "Runs":
            print(f"    {'':<{width}}  → {row['fix']}")
    print()
    if failed:
        print(f"Not ready: {len(failed)} problem(s) above. Follow the → fixes, then rerun bash scripts/doctor.sh.")
    elif warned:
        print(f"Usable, with {len(warned)} warning(s) above.")
    else:
        print("Ready for backtests.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
