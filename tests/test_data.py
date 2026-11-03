"""Candle validation and causal cross-market feature alignment."""
import numpy as np
import pandas as pd
import pytest

from gen_trade.data import ChromosomeContext, FeatureGene, align_features, read_market_data, validate_bars


def candles(closes=(100, 101, 102), *, timestamps=None, symbol="BTC/USDT"):
    closes = np.asarray(closes, dtype=float)
    timestamps = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC") if timestamps is None else pd.DatetimeIndex(timestamps)
    return pd.DataFrame({"timestamp":timestamps, "available_at":timestamps + pd.Timedelta(minutes=30),
                         "open":closes, "high":closes * 1.01, "low":closes * 0.99,
                         "close":closes, "volume":100., "symbol":symbol})


def test_returns_independent_frame_converted_to_utc_and_preserves_missing_fx_volume():
    times = pd.date_range("2024-01-01 09:00", periods=3, freq="h", tz="Asia/Taipei")
    bars = candles(timestamps=times)
    bars.loc[1, "volume"] = np.nan
    result = validate_bars(bars)
    assert str(result.timestamp.dt.tz) == "UTC"
    assert result.timestamp.iloc[0].hour == 1
    assert np.isnan(result.volume.iloc[1])
    result.loc[0, "close"] = 90
    assert bars.close.iloc[0] == 100


@pytest.mark.parametrize("column,value", [("open",0), ("close",np.nan), ("high",np.inf), ("low",-1), ("high",99), ("low",101), ("volume",-1), ("volume",np.inf)])
def test_impossible_price_or_volume_is_rejected(column, value):
    bars = candles()
    bars.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_bars(bars)


def test_naive_missing_duplicate_unsorted_and_delayed_times_are_rejected():
    bars = candles()
    for column in ["timestamp", "available_at"]:
        naive = bars.copy()
        naive[column] = naive[column].dt.tz_localize(None)
        with pytest.raises(ValueError):
            validate_bars(naive)
        missing = bars.copy()
        missing.loc[1, column] = pd.NaT
        with pytest.raises(ValueError):
            validate_bars(missing)
        duplicate = bars.copy()
        duplicate.loc[1, column] = duplicate.loc[0, column]
        with pytest.raises(ValueError):
            validate_bars(duplicate)
    with pytest.raises(ValueError):
        validate_bars(bars.iloc[::-1])
    delayed = bars.copy()
    delayed.loc[0, "available_at"] = delayed.timestamp.iloc[1] + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        validate_bars(delayed)
    preopen = bars.copy()
    preopen.loc[0, "available_at"] = preopen.timestamp.iloc[0] - pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        validate_bars(preopen)


def test_schema_empty_and_mixed_instruments_are_rejected():
    bars = candles()
    with pytest.raises(ValueError):
        validate_bars(bars.drop(columns="available_at"))
    with pytest.raises(ValueError):
        validate_bars(bars.iloc[:0])
    mixed = bars.copy()
    mixed.loc[1, "symbol"] = "XAU/USD"
    with pytest.raises(ValueError):
        validate_bars(mixed)
    mixed.loc[1, "symbol"] = None
    with pytest.raises(ValueError):
        validate_bars(mixed)


def test_cross_market_join_uses_availability_and_staleness_without_backfill():
    target = candles((10, 11, 12, 13))
    times = pd.to_datetime(["2024-01-01 00:15Z", "2024-01-01 01:15Z"])
    gold = candles((2000, 2100), timestamps=times, symbol="XAU/USD")
    # Gold values become known at 00:45 and 01:45, after the first two
    # target decisions at 00:30 and 01:30 respectively.
    gene = FeatureGene("XAU/USD", max_age="50min")
    values = align_features(target, {"XAU/USD":gold}, (gene,))[gene.key()]
    np.testing.assert_allclose(values, [np.nan, 2000, 2100, np.nan], equal_nan=True)


def test_lag_and_rolling_warmup_are_never_filled_from_future_observations():
    source = candles((1, 2, 4, 8, 16), symbol="EUR/USD")
    target = candles((10, 11, 12, 13, 14))
    gene = FeatureGene("EUR/USD", lag=1, window=2, max_age="1h")
    values = align_features(target, {"EUR/USD":source}, (gene,))[gene.key()]
    np.testing.assert_allclose(values, [np.nan, np.nan, 1.5, 3, 6], equal_nan=True)
    future = source.copy()
    future.loc[4, ["open", "close"]] = 100
    future.loc[4, "high"] = 101
    future.loc[4, "low"] = 99
    # An altered future observation cannot revise any feature up to it,
    # because the configured lag excludes even the current row.
    changed = align_features(target, {"EUR/USD":future}, (gene,))[gene.key()]
    np.testing.assert_array_equal(changed, values)


def test_missing_or_duplicate_feature_sources_fail_explicitly():
    gene = FeatureGene("gold")
    with pytest.raises(ValueError, match="Missing source"):
        align_features(candles(), {}, (gene,))
    with pytest.raises(ValueError, match="Duplicate"):
        align_features(candles(), {"gold":candles()}, (gene, gene))


@pytest.mark.parametrize("kwargs", [{"source":""}, {"field":"imagined"}, {"lag":-1}, {"lag":0.5}, {"window":0}, {"window":1.5}, {"max_age":"0h"}, {"max_age":"NaT"}])
def test_invalid_feature_genes(kwargs):
    settings = {"source":"BTC/USDT", **kwargs}
    with pytest.raises(ValueError):
        FeatureGene(**settings)


def test_market_data_reader_groups_valid_instruments_and_rejects_missing_symbols(tmp_path):
    path = tmp_path / "bars.csv"
    source = pd.concat([candles(), candles(symbol="XAU/USD")], ignore_index=True)
    source.to_csv(path, index=False)
    result = read_market_data(path)
    assert set(result) == {"BTC/USDT", "XAU/USD"}
    assert len(result["BTC/USDT"]) == 3
    source.loc[0, "symbol"] = None
    source.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_market_data(path)
    source.iloc[:0].to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_market_data(path)


def test_context_preserves_explicit_extensions():
    gene = FeatureGene("XAU/USD", lag=1)
    context = ChromosomeContext("BTC/USDT", entry_hours=(8, 9), features=(gene,))
    assert context.to_dict()["features"][0]["lag"] == 1
    assert context.to_dict()["pair"] == "BTC/USDT"
    with pytest.raises(ValueError):
        ChromosomeContext("")
