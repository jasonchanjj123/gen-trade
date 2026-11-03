import logging
import os
import uuid

import pytest

from gen_trade.storage import FitnessCache, PostgresStore


def context(**changes):
    values = dict(data_hash="abc123", code_version="commit-42", execution={"fee": .0025},
                  fold={"start": 10, "stop": 20}, seed=4)
    values.update(changes)
    return values


def key():
    return FitnessCache.key({"weights": [.2, .8]}, **context())


def test_keys_are_order_independent_but_include_all_context():
    first = FitnessCache.key({"a": 1, "b": [2]}, **context())
    assert first == FitnessCache.key({"b": [2], "a": 1}, **context())
    for change in [dict(data_hash="different"), dict(code_version="commit-43"),
                   dict(execution={"fee": .001}), dict(fold={"start": 11, "stop": 20}), dict(seed=5)]:
        assert first != FitnessCache.key({"a": 1, "b": [2]}, **context(**change))
    assert first != FitnessCache.key({"a": 2, "b": [2]}, **context())


@pytest.mark.parametrize("payload", [{"a": float("nan")}, {"a": float("inf")}, {1: "x"},
                                     {"a": (1, 2)}, {"a": object()}, [1, 2]])
def test_non_json_or_nonfinite_keys_rejected(payload):
    with pytest.raises(ValueError):
        FitnessCache.key(payload, **context())


@pytest.mark.parametrize("change", [dict(data_hash=""), dict(code_version=" "), dict(seed=True)])
def test_missing_context_rejected(change):
    with pytest.raises(ValueError):
        FitnessCache.key({}, **context(**change))


class FakeCache:
    def __init__(self):
        self.values = {}
        self.last_ttl = None

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value, *, ex):
        self.values[name] = value
        self.last_ttl = ex


def cache_with(client):
    # No real connection/dependency is needed for pure cache-contract checks.
    cache = object.__new__(FitnessCache)
    cache.ttl = 25
    cache._client = client
    return cache


def test_roundtrip_and_ttl():
    client = FakeCache()
    cache = cache_with(client)
    assert cache.get(key()) is None
    cache.set(key(), {"fitness": [1, 2], "valid": True})
    assert cache.get(key()) == {"fitness": [1, 2], "valid": True}
    assert client.last_ttl == 25


class BrokenCache:
    def get(self, name):
        raise OSError("redis://user:SECRET@private-host")

    def set(self, *args, **kwargs):
        raise OSError("redis://user:SECRET@private-host")


def test_network_failure_is_cache_miss_without_credential_logging(caplog):
    cache = cache_with(BrokenCache())
    with caplog.at_level(logging.WARNING):
        assert cache.get(key()) is None
        cache.set(key(), {"value": 1})
    assert "SECRET" not in caplog.text and "private-host" not in caplog.text
    assert "recomputing" in caplog.text


@pytest.mark.parametrize("corrupt", ["{broken", "[]", "NaN", '{"a":NaN}'])
def test_corrupt_cache_values_are_misses(corrupt):
    client = FakeCache()
    client.values[key()] = corrupt
    assert cache_with(client).get(key()) is None


def test_none_client_is_optional():
    cache = cache_with(None)
    assert cache.get(key()) is None
    cache.set(key(), {})


def test_invalid_keys_and_values_fail_before_network():
    cache = cache_with(BrokenCache())
    with pytest.raises(ValueError):
        cache.get("other-app:key")
    with pytest.raises(ValueError):
        cache.get("gen-trade:fitness:" + "-" + "0" * 63)
    with pytest.raises(ValueError):
        cache.set(key(), {"bad": float("nan")})
    with pytest.raises(ValueError):
        FitnessCache("redis://localhost", ttl=0)


def test_postgres_validation_precedes_optional_driver():
    store = PostgresStore("postgresql://not-connected")
    with pytest.raises(ValueError):
        store.save_report({}, "")
    with pytest.raises(ValueError):
        store.save_report({"bad": float("nan")}, "valid")
    with pytest.raises(ValueError):
        store.load_report(" ")


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("GEN_TRADE_TEST_SERVICES") != "1", reason="Local service checks are opt-in")
def test_real_postgres_immutable_idempotent_report():
    pytest.importorskip("psycopg")
    store = PostgresStore(os.environ["DATABASE_URL"])
    identifier = "storage-integration-" + uuid.uuid4().hex
    report = {"data": "synthetic", "seed": 42, "result": {"return": .01}}
    try:
        assert store.load_report(identifier) is None
        store.save_report(report, identifier)
        store.save_report(report, identifier)
        assert store.load_report(identifier) == report
        with pytest.raises(ValueError, match="different report"):
            store.save_report({"data": "other"}, identifier)
        assert store.load_report(identifier) == report
    finally:
        import psycopg
        with psycopg.connect(os.environ["DATABASE_URL"]) as db:
            db.execute("DELETE FROM gen_trade_experiments WHERE experiment_id=%s", (identifier,))


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("GEN_TRADE_TEST_SERVICES") != "1", reason="Local service checks are opt-in")
def test_real_redis_roundtrip_with_expiry():
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    cache = FitnessCache(os.environ["REDIS_URL"], ttl=30)
    identifier = FitnessCache.key({"id": uuid.uuid4().hex}, **context())
    try:
        cache.set(identifier, {"fitness": [1, 2, 3]})
        assert cache.get(identifier) == {"fitness": [1, 2, 3]}
        assert 0 < client.ttl(identifier) <= 30
    finally:
        client.delete(identifier)
