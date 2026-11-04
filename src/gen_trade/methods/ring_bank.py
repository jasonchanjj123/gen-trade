"""Training-only TA-Lib strategy pool for IGTSP §3.3.1.

The paper names indicators but omits the complete rule grid and ranking rule.
This explicit catalog supplies state/cross conditions and ranks buy×sell pairs
by training monthly return, then paper drawdown. See docs/IGTSP_RING.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import talib

from ..data import validate_bars

INDICATORS = ("sma5", "ema20", "trima30", "rsi14", "willr14", "mom5", "cci14", "atr14",
              "bbands20", "obv", "mfi14", "adosc", "macd", "stoch", "dmi14", "aroon14",
              "roc10", "trix30", "sar", "stochrsi", "adx14")
RULE_NAMES = tuple(f"{indicator}:{mode}" for indicator in INDICATORS for mode in ("state", "cross"))


@dataclass(frozen=True)
class Strategy:
    buy_rule: str
    sell_rule: str

    def __post_init__(self):
        if self.buy_rule not in RULE_NAMES or self.sell_rule not in RULE_NAMES:
            raise ValueError("Strategy rules must belong to the explicit TA-Lib catalog")

    @property
    def name(self) -> str:
        return f"buy[{self.buy_rule}]/sell[{self.sell_rule}]"


DEFAULT_BANK = tuple(Strategy(buy, sell) for buy in RULE_NAMES for sell in RULE_NAMES)


def indicator_conditions(bars: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame = validate_bars(bars)
    close, high, low, volume = (frame[name].to_numpy(dtype=float) for name in ("close", "high", "low", "volume"))
    base = {}

    def add(name, left, right, reverse=False):
        a, b = np.asarray(left), np.asarray(right)
        valid = np.isfinite(a) & np.isfinite(b)
        buy, sell = valid & (a > b), valid & (a < b)
        base[name] = (sell, buy, valid) if reverse else (buy, sell, valid)

    def levels(name, values, lower, upper):
        valid = np.isfinite(values)
        base[name] = (valid & (values <= lower), valid & (values >= upper), valid)

    add("sma5", close, talib.SMA(close, timeperiod=5))
    add("ema20", close, talib.EMA(close, timeperiod=20))
    add("trima30", close, talib.TRIMA(close, timeperiod=30))
    levels("rsi14", talib.RSI(close, timeperiod=14), 30, 70)
    levels("willr14", talib.WILLR(high, low, close, timeperiod=14), -80, -20)
    add("mom5", talib.MOM(close, timeperiod=5), 0)
    levels("cci14", talib.CCI(high, low, close, timeperiod=14), -100, 100)
    atr = talib.ATR(high, low, close, timeperiod=14)
    change = np.r_[np.nan, np.diff(close)]
    valid = np.isfinite(atr) & np.isfinite(change)
    base["atr14"] = (valid & (change > atr), valid & (change < -atr), valid)
    upper, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    valid = np.isfinite(upper) & np.isfinite(lower)
    base["bbands20"] = (valid & (close < lower), valid & (close > upper), valid)
    obv = talib.OBV(close, volume)
    add("obv", obv, talib.EMA(obv, timeperiod=10))
    levels("mfi14", talib.MFI(high, low, close, volume, timeperiod=14), 20, 80)
    add("adosc", talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10), 0)
    macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    add("macd", macd, signal)
    slow_k, _ = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
    levels("stoch", slow_k, 20, 80)
    plus, minus = talib.PLUS_DI(high, low, close, timeperiod=14), talib.MINUS_DI(high, low, close, timeperiod=14)
    add("dmi14", plus, minus)
    down, up = talib.AROON(high, low, timeperiod=14)
    add("aroon14", up, down)
    add("roc10", talib.ROC(close, timeperiod=10), 0)
    add("trix30", talib.TRIX(close, timeperiod=30), 0)
    add("sar", close, talib.SAR(high, low, acceleration=.02, maximum=.2))
    fast_k, _ = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3)
    levels("stochrsi", fast_k, 20, 80)
    adx = talib.ADX(high, low, close, timeperiod=14)
    valid = np.isfinite(adx) & np.isfinite(plus) & np.isfinite(minus)
    base["adx14"] = (valid & (adx >= 25) & (plus > minus), valid & (adx >= 25) & (plus < minus), valid)
    buys, sells = {}, {}
    for name in INDICATORS:
        buy, sell, valid = base[name]
        buys[f"{name}:state"], sells[f"{name}:state"] = buy, sell
        previous_valid = np.r_[False, valid[:-1]]
        buys[f"{name}:cross"] = buy & previous_valid & ~np.r_[False, buy[:-1]]
        sells[f"{name}:cross"] = sell & previous_valid & ~np.r_[False, sell[:-1]]
    return buys, sells


def strategy_signals(bars: pd.DataFrame, bank: tuple[Strategy, ...] = DEFAULT_BANK) -> np.ndarray:
    if not bank or len({s.name for s in bank}) != len(bank):
        raise ValueError("Strategy bank must be nonempty with unique names")
    buys, sells = indicator_conditions(bars)
    signals = np.empty((len(bars), len(bank)), dtype=np.int8)
    for i, strategy in enumerate(bank):
        buy, sell = buys[strategy.buy_rule], sells[strategy.sell_rule]
        # Conflicting conditions hold; no precedence is specified in the paper.
        signals[:, i] = buy.astype(np.int8) - sell.astype(np.int8)
    return signals
