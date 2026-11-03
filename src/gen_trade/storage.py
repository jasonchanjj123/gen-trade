"""Optional durable experiment reports and disposable evaluation caching.

Reports are immutable by experiment ID. Cache keys include the data/code/execution
context; cache outages never become a prerequisite for correct computation.
Neither helper logs connection strings or underlying driver exception messages.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any

_LOG = logging.getLogger(__name__)
_PREFIX = "gen-trade:fitness:"


def _canonical(value: dict) -> str:
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("Storage payload must contain finite, native JSON values")

    if type(value) is not dict:
        raise ValueError("Storage payload must be a JSON object")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _experiment_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("experiment_id must be a nonempty string")


class PostgresStore:
    """Save immutable reports; a differing report under an existing ID raises."""

    def __init__(self, dsn: str):
        if not isinstance(dsn, str) or not dsn:
            raise ValueError("A PostgreSQL connection string is required")
        self._dsn = dsn

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install gen-trade[storage] to use PostgreSQL") from exc
        return psycopg.connect(self._dsn, connect_timeout=5)

    @staticmethod
    def _create(db) -> None:
        db.execute("""CREATE TABLE IF NOT EXISTS gen_trade_experiments (
            experiment_id text PRIMARY KEY,
            report jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )""")

    def save_report(self, report: dict, experiment_id: str) -> None:
        _experiment_id(experiment_id)
        canonical = _canonical(report)
        with self._connect() as db:
            self._create(db)
            inserted = db.execute(
                "INSERT INTO gen_trade_experiments (experiment_id, report) VALUES (%s, %s::jsonb) "
                "ON CONFLICT (experiment_id) DO NOTHING RETURNING experiment_id",
                (experiment_id, canonical),
            ).fetchone()
            if inserted is None:
                existing = db.execute(
                    "SELECT report FROM gen_trade_experiments WHERE experiment_id=%s", (experiment_id,)
                ).fetchone()
                # JSONB can normalize numeric spellings (1e2 -> 100, -0.0 -> 0).
                # Compare JSON semantics, not the input's serialized spelling.
                if existing is None or existing[0] != json.loads(canonical):
                    raise ValueError("Experiment ID already exists with a different report")

    def load_report(self, experiment_id: str) -> dict | None:
        _experiment_id(experiment_id)
        with self._connect() as db:
            self._create(db)
            row = db.execute(
                "SELECT report FROM gen_trade_experiments WHERE experiment_id=%s", (experiment_id,)
            ).fetchone()
            return row[0] if row is not None else None


class FitnessCache:
    """Best-effort Redis JSON-object cache; not automatically wired into methods."""

    def __init__(self, redis_url: str, ttl: int = 3600):
        if type(ttl) is not int or ttl <= 0:
            raise ValueError("ttl must be a positive integer number of seconds")
        if not isinstance(redis_url, str) or not redis_url:
            raise ValueError("A Redis connection string is required")
        self.ttl = ttl
        self._client = None
        try:
            import redis
            self._client = redis.Redis.from_url(
                redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2,
            )
        except (ImportError, ValueError, TypeError):
            _LOG.warning("Fitness cache is unavailable; continuing without caching")

    @staticmethod
    def key(chromosome: dict, *, data_hash: str, code_version: str,
            execution: dict, fold: dict, seed: int) -> str:
        if not isinstance(data_hash, str) or not data_hash.strip():
            raise ValueError("data_hash must be nonempty")
        if not isinstance(code_version, str) or not code_version.strip():
            raise ValueError("code_version must be nonempty")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        for value in (chromosome, execution, fold):
            _canonical(value)
        body = _canonical({"schema": 1, "chromosome": chromosome, "data_hash": data_hash,
                           "code_version": code_version, "execution": execution,
                           "fold": fold, "seed": seed})
        return _PREFIX + hashlib.sha256(body.encode()).hexdigest()

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not key.startswith(_PREFIX) or len(key) != len(_PREFIX) + 64:
            raise ValueError("Use a context-complete key from FitnessCache.key")
        if any(char not in "0123456789abcdef" for char in key[len(_PREFIX):]):
            raise ValueError("Malformed fitness-cache key")

    def get(self, key: str) -> dict | None:
        self._validate_key(key)
        if self._client is None:
            return None
        try:
            value = self._client.get(key)
            if value is None:
                return None
            parsed = json.loads(value)
            _canonical(parsed)
            return parsed
        except Exception:
            _LOG.warning("Fitness cache read failed; recomputing without cached values")
            return None

    def set(self, key: str, value: dict) -> None:
        self._validate_key(key)
        encoded = _canonical(value)
        if self._client is None:
            return
        try:
            self._client.set(key, encoded, ex=self.ttl)
        except Exception:
            _LOG.warning("Fitness cache write failed; continuing without caching")
