"""Live progress for long experiment runs.

The console shows a redrawn status line on a terminal, or one line per finished
job otherwise. A benchmark also keeps ``progress.json`` (current state) and
``progress.log`` (one line per event) in its output directory, so another shell
can follow it with ``gen-trade status``. Progress is observational only: it
never changes job order, seeds or report content.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
from time import perf_counter

STATUS_FILE = "progress.json"
LOG_FILE = "progress.log"
SCHEMA = "gen-trade-progress/1"
MODES = ("auto", "bar", "log", "off")


def timed_call(function, *args):
    """Run ``function`` and return ``(seconds, result)``; picklable for workers."""
    start = perf_counter()
    result = function(*args)
    return perf_counter() - start, result


def job_callback(tracker, job, future) -> None:
    """Process-pool completion hook. Progress must never break an experiment."""
    try:
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            tracker.job_failed(job, error)
        else:
            tracker.job_done(job, future.result()[0])
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def format_duration(seconds) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_bar(fraction: float, width: int = 24, stream=None) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(fraction * width + 0.5)
    encoding = (getattr(stream or sys.stdout, "encoding", None) or "").lower()
    full, empty = ("█", "░") if "utf" in encoding else ("#", "-")
    return full * filled + empty * (width - filled)


def describe_job(job: dict | None, asset: bool = True) -> str:
    if not job:
        return "job"
    seed = f"seed {job['seed']}" if job.get("seed") is not None else None
    parts = (job.get("asset") if asset else None, job.get("test_start"), seed, job.get("method"))
    text = " ".join(str(part) for part in parts if part)
    return text + (f" ({job['phase']})" if job.get("phase") else "")


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty()) and os.environ.get("TERM") != "dumb"
    except (AttributeError, ValueError, OSError):
        return False


class ProgressTracker:
    """Counts finished jobs per asset and estimates the remaining time.

    Thread-safe: process-pool completion callbacks arrive on a helper thread.
    ``note`` keeps the run's ordinary messages on stdout in every mode; ``mode``
    only controls the extra progress display on ``stream`` (stderr).
    """

    def __init__(self, *, mode: str = "auto", stream=None, out=None, directory=None,
                 requested=None, command: str | None = None):
        if mode not in MODES:
            raise ValueError(f"progress mode must be one of {MODES}")
        self.stream = stream if stream is not None else sys.stderr
        self.out = out if out is not None else sys.stdout
        self.mode = ("bar" if _isatty(self.stream) else "log") if mode == "auto" else mode
        self.directory: Path | None = None
        self.requested = list(requested or [])
        self.command = command
        self.total = self.done = self.resumed = self.failed = 0
        self.assets: dict[str, dict] = {}
        self.current: str | None = None
        self.stage_text: str | None = None
        self.recent: deque = deque(maxlen=10)
        self.status = "running"
        self.error: str | None = None
        self.started = time.time()
        self.started_at = _now_iso()
        self._write_failed = False
        self._line_shown = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ticker = None
        if directory is not None:
            self.attach(directory)
        if self.mode == "bar":
            self._ticker = threading.Thread(target=self._tick, name="gen-trade-progress", daemon=True)
            self._ticker.start()

    def attach(self, directory, requested=None) -> None:
        """Start keeping progress.json/progress.log in ``directory``."""
        with self._lock:
            self.directory = Path(directory)
            if requested is not None:
                self.requested = list(requested)
            self._append_log(f"=== run started · pid {os.getpid()} · {self.command or 'gen-trade'}")
        self._write()

    # ----- events -------------------------------------------------------
    def plan(self, asset: str, jobs: int) -> None:
        """Declare (or correct) the number of jobs an asset will run."""
        with self._lock:
            entry = self.assets.setdefault(asset, {"status": "pending", "jobs_done": 0, "jobs_total": 0})
            self.total += jobs - entry["jobs_total"]
            entry["jobs_total"] = jobs

    def begin(self, asset: str, jobs: int) -> None:
        with self._lock:
            self.plan(asset, jobs)
            self.assets[asset]["status"] = "running"
            self.current, self.stage_text = asset, "fitting"
        self._log(f"{asset}: {jobs} jobs queued")
        self._write()

    def resume(self, asset: str) -> None:
        """An asset whose finished report was verified and reused."""
        with self._lock:
            entry = self.assets.setdefault(asset, {"status": "pending", "jobs_done": 0, "jobs_total": 0})
            self.resumed += entry["jobs_total"] - entry["jobs_done"]
            entry.update(status="resumed", jobs_done=entry["jobs_total"])
        self._write()

    def job_done(self, job: dict | None, seconds: float | None) -> None:
        with self._lock:
            self.done += 1
            asset = (job or {}).get("asset")
            if asset in self.assets:
                self.assets[asset]["jobs_done"] += 1
            record = {**(job or {}), "seconds": None if seconds is None else round(seconds, 2),
                      "finished_at": _now_iso()}
            self.recent.append(record)
            took = "" if seconds is None else f" · {seconds:.1f}s"
            self._log(f"{self._counter()} {describe_job(job)}{took} · elapsed {format_duration(self.elapsed)}"
                      f" · ETA {self._eta_text()}")
        self._write()

    def job_failed(self, job: dict | None, error: BaseException) -> None:
        with self._lock:
            self.failed += 1
            self._log(f"FAILED {describe_job(job)}: {type(error).__name__}")
        self._write()

    def stage(self, text: str) -> None:
        with self._lock:
            self.stage_text = text
        self._log(text)
        self._write()

    def asset_finished(self, asset: str) -> None:
        with self._lock:
            if asset in self.assets:
                self.assets[asset]["status"] = "done"
            self.stage_text = None
        self._write()

    def note(self, text: str) -> None:
        """An ordinary run message: always printed to stdout and logged."""
        with self._lock:
            self._clear_line()
            print(text, file=self.out, flush=True)
            self._append_log(text)
            self._render()

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=2)
        with self._lock:
            self.status, self.error, self.stage_text = status, error, None
            self._clear_line()
            summary = f"Progress: {status} · {self.done} jobs run"
            if self.resumed:
                summary += f", {self.resumed} reused from earlier runs"
            summary += f" · {format_duration(self.elapsed)}"
            if self.failed:
                summary += f" · {self.failed} failed"
            if self.mode != "off":
                print(summary, file=self.stream, flush=True)
            self._append_log(summary + (f" · {error}" if error else ""))
        self._write()

    # ----- state --------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def remaining(self) -> int:
        return max(self.total - self.done - self.resumed, 0)

    def eta_seconds(self) -> float | None:
        if self.status != "running" or self.remaining == 0:
            return 0.0 if self.remaining == 0 else None
        if not self.done:
            return None
        return self.elapsed / self.done * self.remaining

    def snapshot(self) -> dict:
        with self._lock:
            finished = [name for name, entry in self.assets.items() if entry["status"] in ("done", "resumed")]
            return {
                "schema": SCHEMA, "status": self.status,
                # Kept from the earlier progress.json format.
                "completed": finished, "requested": self.requested or list(self.assets),
                "elapsed_seconds": round(self.elapsed, 1),
                "pid": os.getpid(), "host": socket.gethostname(), "command": self.command,
                "started_at": self.started_at, "updated_at": _now_iso(),
                "jobs": {"done": self.done, "resumed": self.resumed, "failed": self.failed, "total": self.total,
                         "unit": "method fit/validation/test jobs (fold x seed x method)"},
                "eta_seconds": None if self.eta_seconds() is None else round(self.eta_seconds(), 1),
                "current": {"asset": self.current, "stage": self.stage_text} if self.status == "running" else None,
                "assets": {name: dict(entry) for name, entry in self.assets.items()},
                "recent": list(self.recent),
                "error": self.error,
            }

    # ----- output -------------------------------------------------------
    def _counter(self) -> str:
        finished = self.done + self.resumed
        width = len(str(self.total or 0))
        percent = 100 * finished / self.total if self.total else 0
        return f"[{finished:>{width}}/{self.total} {percent:3.0f}%]"

    def _eta_text(self) -> str:
        eta = self.eta_seconds()
        return "?" if eta is None else f"~{format_duration(eta)}"

    def _log(self, line: str) -> None:
        with self._lock:
            self._append_log(line)
            if self.mode == "log":
                print(line, file=self.stream, flush=True)
            self._render()

    def _append_log(self, line: str) -> None:
        if self.directory is None or self._write_failed:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self.directory / LOG_FILE, "a", encoding="utf-8") as handle:
                handle.write(f"{stamp}  {line}\n")
        except OSError:
            self._disable_files()

    def _write(self) -> None:
        # Jobs finish seconds apart, so writing on every event stays cheap and
        # keeps the file current through long gaps between completions.
        with self._lock:  # one writer at a time, so a stale snapshot never lands last
            if self.directory is None or self._write_failed:
                return
            text = json.dumps(self.snapshot(), indent=2, allow_nan=False) + "\n"
            path = self.directory / STATUS_FILE
            temporary = path.with_name(f".{STATUS_FILE}.{os.getpid()}.tmp")
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary.write_text(text, encoding="utf-8")
                os.replace(temporary, path)
            except OSError:
                self._disable_files()

    def _disable_files(self) -> None:
        if not self._write_failed:
            self._write_failed = True
            print(f"Warning: cannot write progress files in {self.directory}; continuing without them",
                  file=self.stream, flush=True)

    def _status_line(self) -> str:
        finished = self.done + self.resumed
        fraction = finished / self.total if self.total else 0.0
        names = list(self.assets)
        where = ""
        if self.current:
            where = self.current + (f" ({names.index(self.current) + 1}/{len(names)})" if len(names) > 1 else "")
        parts = [f"{where} {progress_bar(fraction, 20, self.stream)} {finished}/{self.total} jobs {100 * fraction:.0f}%".strip(),
                 f"{format_duration(self.elapsed)} elapsed", f"ETA {self._eta_text()}"]
        if self.stage_text and self.stage_text != "fitting":
            parts.append(self.stage_text)
        elif self.recent:
            last = self.recent[-1]
            took = f" {last['seconds']:.1f}s" if last.get("seconds") is not None else ""
            parts.append(f"last: {describe_job(last, asset=False)}{took}")
        return " · ".join(parts)

    def _render(self) -> None:
        if self.mode != "bar" or self.status != "running":
            return
        with self._lock:
            width = max(shutil.get_terminal_size((100, 20)).columns - 1, 20)
            line = self._status_line()
            if len(line) > width:
                line = line[:width - 1] + "…"
            self.stream.write("\r\033[K" + line)
            self.stream.flush()
            self._line_shown = True

    def _clear_line(self) -> None:
        if self._line_shown:
            self.stream.write("\r\033[K")
            self.stream.flush()
            self._line_shown = False

    def _tick(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self._render()
            except Exception:
                return


# ----- reading progress from another process -----------------------------
def _alive(pid) -> bool | None:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, TypeError, ValueError):
        return None
    return True


def read_status(directory) -> dict:
    """Load a run's progress, marking runs whose process has stopped."""
    directory = Path(directory)
    path = directory / STATUS_FILE
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        data = {}
    finished = (directory / "summary.json").exists()
    if data.get("schema") != SCHEMA:
        # Earlier format: completed/requested asset lists only.
        requested = data.get("requested") or []
        completed = data.get("completed") or []
        data = {"legacy": True, "requested": requested, "completed": completed,
                "elapsed_seconds": data.get("elapsed_seconds"),
                "assets": {name: {"status": "done" if name in completed else "pending"} for name in requested}}
        data["state"] = "completed" if finished else ("incomplete" if requested else "unknown")
        return data
    state = data.get("status", "unknown")
    if state == "running":
        if data.get("host") == socket.gethostname() and _alive(data.get("pid")) is False:
            state = "stopped"
        elif data.get("host") != socket.gethostname():
            state = "running (other host)"
    data["state"] = state
    return data


def _ago(iso: str | None) -> str:
    try:
        moment = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return "?"
    return format_duration((datetime.now(timezone.utc) - moment).total_seconds()) + " ago"


def render_status(directory, data: dict | None = None, stream=None) -> str:
    data = read_status(directory) if data is None else data
    state = data["state"]
    lines = [f"{directory}  ·  {state}" + (f" (pid {data['pid']})" if state == "running" and data.get("pid") else "")]
    if data.get("legacy"):
        done, requested = len(data["completed"]), len(data["requested"])
        if requested:
            lines.append(f"  {progress_bar(done / requested, 24, stream)}  {done}/{requested} assets")
        if data.get("elapsed_seconds") is not None:
            lines.append(f"  elapsed {format_duration(data['elapsed_seconds'])} (older progress format: per-asset only)")
        return "\n".join(lines)
    jobs = data.get("jobs", {})
    finished, total = jobs.get("done", 0) + jobs.get("resumed", 0), jobs.get("total", 0)
    fraction = finished / total if total else 0.0
    counts = f"{finished}/{total} jobs  {100 * fraction:.0f}%"
    if jobs.get("resumed"):
        counts += f"  ({jobs['resumed']} reused)"
    if jobs.get("failed"):
        counts += f"  ({jobs['failed']} failed)"
    lines.append(f"  {progress_bar(fraction, 24, stream)}  {counts}")
    timing = f"  elapsed {format_duration(data.get('elapsed_seconds'))}"
    if state == "running" and data.get("eta_seconds") is not None:
        timing += f" · ETA ~{format_duration(data['eta_seconds'])}"
    current = data.get("current") or {}
    if state == "running" and current.get("asset"):
        timing += f" · now: {current['asset']}" + (f", {current['stage']}" if current.get("stage") else "")
    lines.append(timing)
    lines.append(f"  started {str(data.get('started_at', '?'))[:19].replace('T', ' ')} · updated {_ago(data.get('updated_at'))}")
    if data.get("assets"):
        marks = []
        for name, entry in data["assets"].items():
            status = entry.get("status")
            if status in ("done", "resumed"):
                marks.append(f"{name} ✓" + (" (reused)" if status == "resumed" else ""))
            elif status == "running":
                marks.append(f"{name} {entry.get('jobs_done', 0)}/{entry.get('jobs_total', 0)}")
            else:
                marks.append(f"{name} ·")
        lines.append("  assets: " + "   ".join(marks))
    if data.get("recent"):
        lines.append("  recent:")
        for job in list(data["recent"])[-5:]:
            took = "" if job.get("seconds") is None else f"{job['seconds']:>7.1f}s"
            lines.append(f"    {describe_job(job):<48} {took}")
    if data.get("error"):
        lines.append(f"  error: {data['error']}")
    if state == "stopped":
        lines.append("  The process is gone without a final status (killed, crashed or the machine restarted).")
        lines.append("  Rerun the same command to resume: finished assets are verified and reused.")
    if data.get("command"):
        lines.append(f"  command: {data['command']}")
    return "\n".join(lines)


def find_runs(root) -> list[Path]:
    root = Path(root)
    if (root / STATUS_FILE).exists():
        return [root]
    runs = [path.parent for path in root.glob(f"*/{STATUS_FILE}")]
    return sorted(runs, key=lambda path: (path / STATUS_FILE).stat().st_mtime, reverse=True)


def render_overview(runs, stream=None) -> str:
    if not runs:
        return "No runs with progress.json found."
    rows = [("STATUS", "PROGRESS", "ELAPSED", "ETA", "RUN")]
    for run in runs:
        data = read_status(run)
        if data.get("legacy"):
            progress = f"{len(data['completed'])}/{len(data['requested'])} assets"
            eta = "–"
        else:
            jobs = data.get("jobs", {})
            finished, total = jobs.get("done", 0) + jobs.get("resumed", 0), jobs.get("total", 0)
            progress = f"{finished}/{total} {100 * finished / total if total else 0:3.0f}%"
            eta = f"~{format_duration(data['eta_seconds'])}" if data["state"] == "running" and data.get("eta_seconds") is not None else "–"
        rows.append((data["state"], progress, format_duration(data.get("elapsed_seconds")), eta, str(run)))
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    return "\n".join("  ".join(cell.ljust(widths[i]) if i < 4 else cell for i, cell in enumerate(row)) for row in rows)


def show_status(paths, *, watch: float | None = None, root="reports", stream=None) -> int:
    """CLI entry: a detailed view for one run, an overview for several."""
    stream = stream if stream is not None else sys.stdout
    targets = [Path(path) for path in paths] if paths else None

    def render() -> tuple[str, bool]:
        runs = targets if targets is not None else find_runs(root)
        if targets is not None:
            missing = [str(run) for run in runs if not run.exists()]
            if missing:
                raise FileNotFoundError(f"No such run directory: {', '.join(missing)}")
        if len(runs) == 1:
            data = read_status(runs[0])
            return render_status(runs[0], data, stream), data["state"] == "running"
        running = any(read_status(run)["state"] == "running" for run in runs)
        return render_overview(runs, stream), running

    if not watch:
        print(render()[0], file=stream)
        return 0
    clear = _isatty(stream)
    try:
        while True:
            text, running = render()
            if clear:
                stream.write("\033[H\033[2J")
            print(text, file=stream, flush=True)
            if not running:
                return 0
            if not clear:
                print(file=stream)
            time.sleep(watch)
    except KeyboardInterrupt:
        return 0
