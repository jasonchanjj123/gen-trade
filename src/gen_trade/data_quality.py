"""Data-quality audit for market datasets.

Every dataset can be checked against:

* data contracts: schema and cross-field rules for its asset class;
* completeness: expected XNYS sessions or FX weekdays, and partial bars;
* accuracy: robust outlier, spike-and-reversal, wick, stale-price and spread screens;
* consistency: split dating, unexplained split-like jumps and dividend sizes;
* reconciliation: the vendor's dividends against its own adjusted closes, and
  independent reference prices (ECB euro reference rates, or a supplied file);
* lineage: raw responses against the manifest, and a rebuild of every dataset
  file from its raw cache with the current code.

Severity ``error`` means results built on the data would be wrong; benchmarks
refuse such a dataset. ``warning`` flags observations for a person to review:
most are genuine market events, and a reviewed exception (who, when, why)
records that decision. ``info`` describes the data without judging it. The
audit never edits, fills or drops an observation. See docs/DATA_QUALITY.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .data import validate_bars

QUALITY_SCHEMA_VERSION = 1
QUALITY_FILE = 'quality.json'
EXCEPTIONS_FILE = 'data-quality-exceptions.json'
MAX_LISTED = 500
# No regular spot-FX session: the market is effectively closed worldwide.
FX_CLOSED_DAYS = ((1, 1), (12, 25))
ECB_PROVIDER = 'ECB euro foreign exchange reference rates (14:15 CET concertation)'
ECB_ENDPOINT = 'https://data-api.ecb.europa.eu/service/data/EXR/D.{currencies}.EUR.SP00.A'

REQUIRED_COLUMNS = {
    'stock': ('symbol', 'session_date', 'asset_class', 'execution_open', 'execution_high', 'execution_low',
              'execution_close', 'execution_volume', 'split_ratio', 'dividend'),
    'fx': ('symbol', 'session_date', 'asset_class', 'bid_open', 'bid_high', 'bid_low', 'bid_close',
           'ask_open', 'ask_high', 'ask_low', 'ask_close', 'bid_feed_volume', 'ask_feed_volume',
           'split_ratio', 'dividend'),
    'generic': (),
}
PRICE_SETS = {'stock': ('', 'execution_'), 'fx': ('', 'bid_', 'ask_'), 'generic': ('',)}
MARKET_TIMEZONE = {'stock': 'America/New_York', 'fx': 'UTC', 'generic': 'UTC'}

# Check id -> (quality dimension, what it verifies). Severity is set per result.
CHECKS = {
    'contract.dataset': ('validity', 'manifest structure, file presence and file hashes'),
    'contract.bars': ('validity', 'timezone-aware, strictly ordered, causal bars with positive finite OHLC'),
    'contract.schema': ('validity', 'columns required by the asset class are present and numeric'),
    'contract.price_sets': ('validity', 'every OHLC set (signal, execution, bid, ask) is positive with high/low containment'),
    'contract.bid_ask': ('validity', 'ask at or above bid at the open and the close'),
    'contract.signal_prices': ('consistency', 'signal OHLCV equals execution data times already-effective splits (stocks) or the bid (FX)'),
    'contract.corporate_actions': ('validity', 'split ratios positive, dividends finite and nonnegative, none for FX'),
    'contract.session_dates': ('consistency', 'session_date is the bar date in the market time zone'),
    'completeness.sessions': ('completeness', 'every expected exchange session (stocks) or FX weekday is present'),
    'completeness.partial_bars': ('completeness', 'activity is not a small fraction of its trailing median (partial candle)'),
    'completeness.reference_coverage': ('completeness', 'every reference-source date inside the data range has a bar'),
    'timeliness.window': ('timeliness', 'the data start and end match the requested window'),
    'accuracy.return_outliers': ('accuracy', 'close-to-close moves within a robust z-score bound'),
    'accuracy.spike_reversals': ('accuracy', 'no large move fully reversed on the next bar (bad-print signature)'),
    'accuracy.wicks': ('accuracy', 'highs and lows within a multiple of the trailing median range'),
    'accuracy.stale_prices': ('accuracy', 'no runs of identical closes and no zero-range bars'),
    'accuracy.intraday_crossed': ('accuracy', 'ask high/low at or above bid high/low (no crossed quote inside the bar)'),
    'accuracy.spreads': ('accuracy', 'opening bid/ask spread within a multiple of its trailing median'),
    'consistency.split_dates': ('consistency', 'no split-sized gap in the split-neutral prices on a split date '
                                               '(the vendor adjusted its history for the event)'),
    'consistency.split_like_jumps': ('consistency', 'no split-sized overnight jump without a split event'),
    'consistency.dividends': ('consistency', 'each dividend is a plausible fraction of the prior close'),
    'reconciliation.vendor_dividends': ('accuracy', "dividend events match the steps in the vendor's adjusted close"),
    'reconciliation.reference_prices': ('accuracy', 'independent reference prices agree with the bars'),
    'lineage.raw_integrity': ('lineage', 'raw responses match their sidecars and the manifest hashes'),
    'lineage.raw_availability': ('lineage', 'the raw responses behind each dataset file are cached'),
    'lineage.rebuild': ('lineage', 'rebuilding from the raw cache with current code reproduces the dataset file'),
    'lineage.normalizer': ('lineage', 'the normalizing code is unchanged since the data were fetched'),
}


@dataclass(frozen=True)
class QualityThresholds:
    """Screening thresholds; robust scales use only rows before the one screened."""

    trailing_window: int = 60
    min_history: int = 20
    outlier_z: float = 10.0
    reversal_z: float = 6.0
    reversal_retained: float = 0.5
    wick_multiple: float = 6.0
    stale_run: int = 3
    partial_activity: float = 0.05
    spread_multiple: float = 10.0
    spread_floor_bps: float = 3.0
    split_like_jump: float = 1.9
    max_dividend_yield: float = 0.10
    edge_days: int = 7
    dividend_rel_tolerance: float = 0.01
    dividend_abs_tolerance: float = 0.001
    reference_close_bps: float = 25.0
    reference_fix_bps: float = 2.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive')
        if self.min_history > self.trailing_window or self.reversal_retained >= 1:
            raise ValueError('min_history must not exceed trailing_window; reversal_retained must be below 1')


def _result(check, severity, flagged=(), *, detail=None, metrics=None, skipped=None) -> dict:
    items = sorted({str(item) for item in flagged})
    if skipped:
        status = 'skipped'
    elif severity == 'info':
        status = 'noted' if items else 'passed'
    else:
        status = 'failed' if items else 'passed'
    record = {'check': check, 'dimension': CHECKS[check][0], 'severity': severity, 'status': status,
              'count': len(items), 'flagged': items[:MAX_LISTED], 'detail': skipped or detail or CHECKS[check][1]}
    if len(items) > MAX_LISTED:
        record['flagged_truncated'] = True
    if metrics:
        record['metrics'] = metrics
    return record


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def quality_code_sha256() -> str:
    """Hash of the audit and normalizing code."""
    digest = hashlib.sha256()
    for name in ('data_quality.py', 'market_data.py', 'data.py'):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def asset_class_of(frame: pd.DataFrame) -> str:
    if 'asset_class' in frame and frame['asset_class'].nunique() == 1:
        kind = str(frame['asset_class'].iloc[0])
        if kind in ('stock', 'fx'):
            return kind
    return 'generic'


def _session_dates(frame: pd.DataFrame, kind: str) -> pd.DatetimeIndex:
    if 'session_date' in frame:
        return pd.DatetimeIndex(pd.to_datetime(frame['session_date'].astype(str), format='%Y-%m-%d'))
    stamps = pd.to_datetime(frame['timestamp'], utc=True).dt.tz_convert(MARKET_TIMEZONE[kind])
    return pd.DatetimeIndex(stamps.dt.tz_localize(None).dt.normalize())


def _fx_closed(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.isin(dates.month * 100 + dates.day, [m * 100 + d for m, d in FX_CLOSED_DAYS])


def _regular(dates: pd.DatetimeIndex, kind: str) -> np.ndarray:
    """Rows that are full sessions: FX weekdays outside closed days; every stock row."""
    if kind == 'fx':
        return (dates.day_of_week < 5) & ~_fx_closed(dates)
    return np.ones(len(dates), dtype=bool)


def _trailing_median(values: pd.Series, regular: np.ndarray, t: QualityThresholds) -> pd.Series:
    base = values.where(regular)
    return base.rolling(t.trailing_window, min_periods=t.min_history).median().shift(1)


def _robust_z(values: pd.Series, regular: np.ndarray, t: QualityThresholds) -> pd.Series:
    center = _trailing_median(values, regular, t)
    spread = _trailing_median((values - center).abs(), regular, t) * 1.4826
    return (values - center) / spread.where(spread > 0)


def _names(bars: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Index:
    """Row labels for flagged items: session dates, or UTC bar times for intraday data."""
    if dates.has_duplicates:
        return pd.Index(pd.to_datetime(bars['timestamp'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
    return pd.Index(dates.strftime('%Y-%m-%d'))


def _labels(names: pd.Index, mask) -> list[str]:
    return list(names[np.asarray(mask, dtype=bool)])


def _window(window, dates: pd.DatetimeIndex):
    """Declared [start, end) as naive dates, or the observed range when undeclared."""
    start, end = window or (None, None)
    if start and end:
        return pd.Timestamp(start), pd.Timestamp(end), True
    return dates[0], dates[-1] + pd.Timedelta(days=1), False


def audit_frame(frame: pd.DataFrame, *, thresholds: QualityThresholds | None = None, window=None) -> list[dict]:
    """Contract, completeness, accuracy and consistency checks for one instrument.

    ``window`` is the requested ``(start, end_exclusive)``; without it the
    observed first and last dates bound the completeness checks.
    """
    t = thresholds or QualityThresholds()
    try:
        bars = validate_bars(frame)
    except (ValueError, TypeError) as exc:
        return [_result('contract.bars', 'error', [str(exc)])]
    checks = [_result('contract.bars', 'error')]
    kind = asset_class_of(bars)
    missing = [name for name in REQUIRED_COLUMNS[kind] if name not in bars]
    non_numeric = [name for name in REQUIRED_COLUMNS[kind] if name in bars and name not in ('symbol', 'session_date', 'asset_class')
                   and not pd.api.types.is_numeric_dtype(bars[name])]
    checks.append(_result('contract.schema', 'error', [f'missing {name}' for name in missing]
                          + [f'non-numeric {name}' for name in non_numeric], metrics={'asset_class': kind}))
    if missing or non_numeric:
        return checks
    dates = _session_dates(bars, kind)
    names = _names(bars, dates)
    if kind != 'generic' and dates.has_duplicates:
        checks.append(_result('contract.session_dates', 'error', [f'duplicate session {d:%Y-%m-%d}'
                                                                  for d in dates[dates.duplicated()]]))
        return checks
    regular = _regular(dates, kind)
    checks += _contract_checks(bars, kind, dates, names)
    checks += _completeness_checks(bars, kind, dates, names, regular, window, t)
    checks += _accuracy_checks(bars, kind, dates, names, regular, t)
    if kind == 'stock':
        checks += _corporate_action_checks(bars, dates, names, t)
    return checks


def _contract_checks(bars, kind, dates, names) -> list[dict]:
    checks = []
    bad = np.zeros(len(bars), dtype=bool)
    for prefix in PRICE_SETS[kind]:
        o, h, l, c = (bars[f'{prefix}{name}'].to_numpy(dtype=float) for name in ('open', 'high', 'low', 'close'))
        with np.errstate(invalid='ignore'):
            bad |= ~np.isfinite(np.c_[o, h, l, c]).all(axis=1) | (np.c_[o, h, l, c] <= 0).any(axis=1)
            bad |= (h < np.maximum.reduce([o, l, c])) | (l > np.minimum.reduce([o, h, c]))
    checks.append(_result('contract.price_sets', 'error', _labels(names, bad)))
    if kind == 'fx':
        crossed = ((bars['ask_open'] < bars['bid_open']) | (bars['ask_close'] < bars['bid_close'])).to_numpy()
        checks.append(_result('contract.bid_ask', 'error', _labels(names, crossed)))
        signal = np.zeros(len(bars), dtype=bool)
        for name in ('open', 'high', 'low', 'close'):
            signal |= (bars[name] != bars[f'bid_{name}']).to_numpy()
        checks.append(_result('contract.signal_prices', 'error', _labels(names, signal),
                              detail='signal OHLC equals the observed bid OHLC'))
        actions = ((bars['split_ratio'] != 1) | (bars['dividend'] != 0)).to_numpy()
    elif kind == 'generic' and not {'split_ratio', 'dividend'} <= set(bars.columns):
        actions = None  # a generic feed without corporate-action columns
    else:
        split = bars['split_ratio'].to_numpy(dtype=float)
        dividend = bars['dividend'].to_numpy(dtype=float)
        actions = ~np.isfinite(split) | (split <= 0) | ~np.isfinite(dividend) | (dividend < 0)
        if kind == 'stock':
            factor = np.cumprod(np.where(actions, 1., split))
            signal = np.zeros(len(bars), dtype=bool)
            for name in ('open', 'high', 'low', 'close'):
                signal |= ~np.isclose(bars[name], bars[f'execution_{name}'] * factor, rtol=1e-9, atol=0)
            volume, executed = bars['volume'].to_numpy(dtype=float), bars['execution_volume'].to_numpy(dtype=float)
            known = np.isfinite(volume) & np.isfinite(executed)
            signal |= known & ~np.isclose(volume * factor, executed, rtol=1e-9, atol=1e-9)
            checks.append(_result('contract.signal_prices', 'error', _labels(names, signal)))
    if actions is not None:
        checks.append(_result('contract.corporate_actions', 'error', _labels(names, actions)))
    if 'session_date' in bars:
        stamps = pd.to_datetime(bars['timestamp'], utc=True).dt.tz_convert(MARKET_TIMEZONE[kind])
        local = pd.DatetimeIndex(stamps.dt.tz_localize(None).dt.normalize())
        checks.append(_result('contract.session_dates', 'error', _labels(names, local != dates)))
    return checks


def _completeness_checks(bars, kind, dates, names, regular, window, t) -> list[dict]:
    checks = []
    start, end, declared = _window(window, dates)
    present = pd.DatetimeIndex(dates)
    if kind == 'stock':
        try:
            import exchange_calendars as xcals
        except ImportError:
            checks.append(_result('completeness.sessions', 'error', skipped='exchange_calendars is not installed'))
        else:
            calendar = xcals.get_calendar('XNYS', start=start.strftime('%Y-%m-%d'),
                                          end=(end - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
            expected = calendar.sessions[(calendar.sessions >= start) & (calendar.sessions < end)]
            expected = pd.DatetimeIndex(expected).tz_localize(None) if expected.tz is not None else pd.DatetimeIndex(expected)
            flagged = [f'missing {d:%Y-%m-%d}' for d in expected.difference(present)]
            flagged += [f'unexpected {d:%Y-%m-%d}' for d in present.difference(expected)]
            checks.append(_result('completeness.sessions', 'error', flagged,
                                  metrics={'expected_sessions': len(expected), 'calendar': 'XNYS'}))
    elif kind == 'fx':
        weekdays = pd.bdate_range(start, end - pd.Timedelta(days=1))
        expected = weekdays[~_fx_closed(weekdays)]
        flagged = [f'missing {d:%Y-%m-%d}' for d in expected.difference(present)]
        flagged += [f'saturday {d:%Y-%m-%d}' for d in present[present.day_of_week == 5]]
        checks.append(_result('completeness.sessions', 'warning', flagged,
                              metrics={'expected_weekdays': len(expected),
                                       'closed_days_excluded': [f'{m:02d}-{d:02d}' for m, d in FX_CLOSED_DAYS]}))
    if declared:
        late = (dates[0] - start).days
        early = (end - pd.Timedelta(days=1) - dates[-1]).days
        flagged = ([f'starts {dates[0]:%Y-%m-%d}, {late} days after the requested start'] if late > t.edge_days else [])
        flagged += ([f'ends {dates[-1]:%Y-%m-%d}, {early} days before the requested end'] if early > t.edge_days else [])
        checks.append(_result('timeliness.window', 'warning', flagged))
    if kind == 'fx':
        activity = bars['bid_feed_volume'].astype(float) + bars['ask_feed_volume'].astype(float)
    else:
        activity = bars['execution_volume' if kind == 'stock' else 'volume'].astype(float)
    if activity.notna().any():
        base = _trailing_median(activity, regular, t)
        partial = regular & ((activity < t.partial_activity * base) | (activity <= 0) | activity.isna()).to_numpy()
        checks.append(_result('completeness.partial_bars', 'warning', _labels(names, partial),
                              metrics={'activity': 'bid+ask feed activity' if kind == 'fx' else 'volume'}))
    return checks


def _accuracy_checks(bars, kind, dates, names, regular, t) -> list[dict]:
    checks = []
    close = bars['close'].astype(float)
    returns = np.log(close).diff()
    z = _robust_z(returns, regular, t)
    outliers = regular & (z.abs() > t.outlier_z).to_numpy()
    checks.append(_result('accuracy.return_outliers', 'warning', _labels(names, outliers),
                          metrics={'max_abs_robust_z': float(np.nanmax(z.abs().where(regular))) if z.notna().any() else None}))
    following, next_z = returns.shift(-1), z.shift(-1)
    reversal = (regular & (z.abs() > t.reversal_z) & (next_z.abs() > t.reversal_z)
                & (np.sign(returns) != np.sign(following))
                & ((returns + following).abs() < t.reversal_retained * np.minimum(returns.abs(), following.abs())))
    checks.append(_result('accuracy.spike_reversals', 'warning', _labels(names, reversal.fillna(False))))
    o, h, l = (bars[name].astype(float) for name in ('open', 'high', 'low'))
    wick = np.maximum(np.log(h / np.maximum(o, close)), np.log(np.minimum(o, close) / l))
    typical = _trailing_median(np.log(h / l), regular, t)
    checks.append(_result('accuracy.wicks', 'warning', _labels(names, regular & (wick > t.wick_multiple * typical).to_numpy())))
    values = close.to_numpy()
    run = np.ones(len(values), dtype=int)
    for i in range(1, len(values)):
        run[i] = run[i - 1] + 1 if values[i] == values[i - 1] else 1
    stale = (run >= t.stale_run) | (regular & (h == l).to_numpy())
    checks.append(_result('accuracy.stale_prices', 'warning', _labels(names, stale)))
    if kind == 'fx':
        crossed = ((bars['ask_high'] < bars['bid_high']) | (bars['ask_low'] < bars['bid_low'])).to_numpy()
        checks.append(_result('accuracy.intraday_crossed', 'warning', _labels(names, crossed)))
        def spread(field):
            bid, ask = bars[f'bid_{field}'].astype(float), bars[f'ask_{field}'].astype(float)
            return (ask - bid) / ((ask + bid) / 2) * 1e4
        opening, closing = spread('open'), spread('close')
        base = _trailing_median(opening, regular, t)
        wide = regular & ((opening > t.spread_multiple * base) & (opening > t.spread_floor_bps)).to_numpy()
        weekday = pd.Series(dates.day_of_week, index=bars.index)
        def profile(values):
            return {name: _finite(values[weekday == number].median())
                    for number, name in enumerate(('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'))
                    if (weekday == number).any()}
        checks.append(_result('accuracy.spreads', 'warning', _labels(names, wide),
                              metrics={'median_open_spread_bps_by_weekday': profile(opening),
                                       'median_close_spread_bps_by_weekday': profile(closing)}))
    return checks


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _corporate_action_checks(bars, dates, names, t) -> list[dict]:
    split = bars['split_ratio'].to_numpy(dtype=float)
    opening = bars['execution_open'].to_numpy(dtype=float)
    previous = np.r_[np.nan, bars['execution_close'].to_numpy(dtype=float)[:-1]]
    with np.errstate(invalid='ignore', divide='ignore'):
        overnight = opening / previous
        residual = np.log(overnight * split)
        misdated = (split != 1) & np.isfinite(residual) & (np.abs(residual) > .5 * np.abs(np.log(split)))
        jumps = (split == 1) & ((overnight >= t.split_like_jump) | (overnight <= 1 / t.split_like_jump))
        dividend_yield = bars['dividend'].to_numpy(dtype=float) / previous
    split_days = [f'{d:%Y-%m-%d} ratio {r:g}, overnight move after the split {np.exp(e) - 1:+.1%}'
                  for d, r, e in zip(dates[split != 1], split[split != 1], residual[split != 1])]
    return [
        _result('consistency.split_dates', 'error', _labels(names, misdated), metrics={'split_days': split_days}),
        _result('consistency.split_like_jumps', 'warning', _labels(names, jumps)),
        _result('consistency.dividends', 'warning', _labels(names, np.nan_to_num(dividend_yield) > t.max_dividend_yield)),
    ]


def reconcile_yahoo_dividends(result: dict, start: str, end: str, thresholds: QualityThresholds | None = None) -> dict:
    """Compare Yahoo dividend events with the steps in Yahoo's own adjusted close.

    Both close and adjusted close are split-adjusted, so their ratio steps only
    at a dividend ex-date ``t``, by ``1 - D_t / C_{t-1}``. A step without an
    event, or an event without a matching step, means the corporate-action
    feed is incomplete or inconsistent.
    """
    t = thresholds or QualityThresholds()
    indicators = result.get('indicators', {})
    adjusted = (indicators.get('adjclose') or [{}])[0].get('adjclose')
    if adjusted is None:
        return _result('reconciliation.vendor_dividends', 'error', skipped='response has no adjusted close')
    dates = pd.DatetimeIndex(pd.to_datetime(result['timestamp'], unit='s', utc=True)
                             .tz_convert('America/New_York').tz_localize(None).normalize())
    if dates.has_duplicates or len(adjusted) != len(dates):
        return _result('reconciliation.vendor_dividends', 'error',
                       skipped='response dates are duplicated or misaligned with the adjusted close')
    close = pd.Series(indicators['quote'][0]['close'], index=dates, dtype=float)
    adjusted = pd.Series(adjusted, index=dates, dtype=float)
    ratio = adjusted / close
    step = ratio.shift(1) / ratio
    implied = (1 - step) * close.shift(1)
    events = {}
    for event in result.get('events', {}).get('dividends', {}).values():
        day = pd.Timestamp(event['date'], unit='s', tz='UTC').tz_convert('America/New_York').tz_localize(None).normalize()
        events[day] = events.get(day, 0.) + float(event['amount'])
    inside = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)) & step.notna().to_numpy()
    flagged, errors = [], []
    for day, amount in sorted(events.items()):
        if pd.Timestamp(start) <= day < pd.Timestamp(end):
            value = implied.get(day, np.nan)
            error = abs(value - amount) if np.isfinite(value) else np.inf
            errors.append(error / amount if amount else error)
            if error > max(t.dividend_abs_tolerance, t.dividend_rel_tolerance * amount):
                flagged.append(f'{day:%Y-%m-%d} event {amount:g} vs adjusted-close step {value:.6g}')
    unexplained = inside & ~dates.isin(list(events)) & ((1 - step).abs() > 5e-5).to_numpy()
    flagged += [f'{d:%Y-%m-%d} adjusted-close step without a dividend event (implied {v:.6g})'
                for d, v in zip(dates[unexplained], implied[unexplained])]
    finite = [e for e in errors if np.isfinite(e)]
    return _result('reconciliation.vendor_dividends', 'error', flagged,
                   metrics={'events': len(errors), 'max_relative_error': max(finite) if finite else None})


def parse_ecb_csv(payload: bytes) -> pd.DataFrame:
    """ECB EXR CSV -> date x currency table of units per EUR."""
    table = pd.read_csv(io.BytesIO(payload), usecols=['CURRENCY', 'CURRENCY_DENOM', 'TIME_PERIOD', 'OBS_VALUE'])
    if table.empty or (table['CURRENCY_DENOM'] != 'EUR').any():
        raise ValueError('ECB reference-rate response must hold EUR-denominated observations')
    table['OBS_VALUE'] = pd.to_numeric(table['OBS_VALUE'], errors='coerce')
    table = table.dropna(subset=['OBS_VALUE'])
    if (table['OBS_VALUE'] <= 0).any() or table.duplicated(['CURRENCY', 'TIME_PERIOD']).any():
        raise ValueError('ECB reference rates must be positive and unique per currency and date')
    return table.pivot(index='TIME_PERIOD', columns='CURRENCY', values='OBS_VALUE').sort_index()


def ecb_pair_rates(rates: pd.DataFrame, pair: str) -> pd.Series:
    """Price of the pair's base in its quote currency from per-EUR reference rates."""
    base, quote = pair.split('/')
    def per_euro(currency):
        if currency == 'EUR':
            return pd.Series(1., index=rates.index)
        if currency not in rates:
            raise ValueError(f'ECB publishes no reference rate for {currency}')
        return rates[currency]
    return (per_euro(quote) / per_euro(base)).dropna()


def ecb_reference(pairs, start: str, end: str, raw_directory: str | Path, *, refresh=False) -> tuple[pd.DataFrame, dict]:
    """Download (once, hash-cached) ECB reference rates and express them per pair as daily fixes."""
    from .market_data import _download
    currencies = sorted({currency for pair in pairs for currency in pair.split('/')} - {'EUR'})
    last = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    url = (ECB_ENDPOINT.format(currencies='+'.join(currencies)) + '?'
           + urlencode({'startPeriod': start, 'endPeriod': last, 'format': 'csvdata'}))
    path = Path(raw_directory) / 'ECB' / f"EXR-{'-'.join(currencies)}-{start}-{last}.csv"
    payload, info = _download(url, path, refresh)
    rates = parse_ecb_csv(payload)
    frames, unpublished = [], []
    for pair in pairs:
        try:
            values = ecb_pair_rates(rates, pair)
        except ValueError:
            unpublished.append(pair)
            continue
        frames.append(pd.DataFrame({'symbol': pair, 'session_date': values.index, 'value': values.to_numpy(),
                                    'kind': 'fix'}))
    reference = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=['symbol', 'session_date', 'value', 'kind']))
    return reference, {**info, 'provider': ECB_PROVIDER, 'kind': 'fix', 'pairs_without_reference': unpublished,
                       'method': 'cross rates from per-EUR fixes; each fix must lie inside the UTC daily bid-low/ask-high range'}


def read_reference(path: str | Path) -> pd.DataFrame:
    """Reference prices from an independent source: symbol, session_date, value and kind.

    ``kind`` is ``close`` (an as-traded official closing price; stocks only)
    or ``fix`` (a price observed at some time during the session, which must
    lie inside the bar's range). An optional ``source`` column names the vendor.
    """
    path = Path(path)
    table = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype={'session_date': str})
    if missing := {'symbol', 'session_date', 'value', 'kind'} - set(table.columns):
        raise ValueError(f'Reference prices need columns {sorted(missing)}')
    table['value'] = pd.to_numeric(table['value'], errors='raise').astype(float)
    if not np.isfinite(table['value']).all() or (table['value'] <= 0).any():
        raise ValueError('Reference prices must be finite and positive')
    if not table['kind'].isin(['close', 'fix']).all():
        raise ValueError("Reference kind must be 'close' or 'fix'")
    table['session_date'] = pd.to_datetime(table['session_date'].astype(str)).dt.strftime('%Y-%m-%d')
    if table.duplicated(['symbol', 'session_date', 'kind']).any():
        raise ValueError('Reference prices must be unique per symbol, date and kind')
    return table


def reconcile_reference(bars: pd.DataFrame, reference: pd.DataFrame, thresholds: QualityThresholds | None = None,
                        *, source: str = 'reference file') -> list[dict]:
    """Closing prices within a tolerance; fixes inside the bar's [low, high] range."""
    t = thresholds or QualityThresholds()
    kind = asset_class_of(bars)
    dates = _session_dates(bars, kind)
    if dates.has_duplicates:
        reason = 'several bars per session date; daily reference prices cannot be matched'
        return [_result('reconciliation.reference_prices', 'warning', skipped=reason),
                _result('completeness.reference_coverage', 'warning', skipped=reason)]
    labels = pd.Index(dates.strftime('%Y-%m-%d'))
    if kind == 'fx':
        low, high, close = bars['bid_low'], bars['ask_high'], (bars['bid_close'] + bars['ask_close']) / 2
    elif kind == 'stock':
        low, high, close = bars['execution_low'], bars['execution_high'], bars['execution_close']
    else:
        low, high, close = bars['low'], bars['high'], bars['close']
    table = pd.DataFrame({'low': low.to_numpy(float), 'high': high.to_numpy(float), 'close': close.to_numpy(float)},
                         index=labels)
    flagged, missing, deviations = [], [], []
    first, last = labels[0], labels[-1]
    for row in reference.itertuples(index=False):
        if not first <= row.session_date <= last:
            continue
        if row.session_date not in table.index:
            missing.append(row.session_date)
            continue
        bar = table.loc[row.session_date]
        if row.kind == 'close' and kind != 'fx':
            deviation = abs(row.value / bar['close'] - 1) * 1e4
            if deviation > t.reference_close_bps:
                flagged.append(f'{row.session_date} close {row.value:g} vs {bar["close"]:g} ({deviation:.1f} bp)')
        else:
            deviation = max(bar['low'] / row.value - 1, row.value / bar['high'] - 1, 0) * 1e4
            if deviation > t.reference_fix_bps:
                flagged.append(f'{row.session_date} fix {row.value:g} outside [{bar["low"]:g}, {bar["high"]:g}] '
                               f'by {deviation:.1f} bp')
        deviations.append(deviation)
    metrics = {'source': source, 'matched': len(deviations),
               'max_deviation_bps': max(deviations) if deviations else None}
    return [_result('reconciliation.reference_prices', 'warning', flagged, metrics=metrics),
            _result('completeness.reference_coverage', 'warning', [f'no bar on {d}' for d in missing],
                    metrics={'source': source})]


def _lineage_checks(directory: Path, manifest: dict, symbol: str, stored: pd.DataFrame, t, rebuild: bool) -> list[dict]:
    from . import market_data as md
    info = manifest['instruments'][symbol]
    start, end = manifest.get('requested_start'), manifest.get('requested_end_exclusive')
    sources = info.get('raw_sources')
    if not sources or not start or not end:
        reason = 'dataset records no raw sources or requested window'
        return [_result(name, 'error' if name != 'lineage.raw_availability' else 'warning', skipped=reason)
                for name in ('lineage.raw_integrity', 'lineage.raw_availability', 'lineage.rebuild')]
    recorded = {source['url']: source['sha256'] for source in sources}
    if '/' in symbol:
        paths = {(year, side): md.fx_raw_path(directory, symbol, year, side)
                 for side in ('BID', 'ASK') for year in md.fx_years(start, end)}
    else:
        paths = {'chart': md.stock_raw_path(directory, symbol)}
    absent, mismatched, payloads = [], [], {}
    for key, path in paths.items():
        sidecar = path.with_suffix(path.suffix + '.meta.json')
        if not path.exists() or not sidecar.exists():
            absent.append(str(path.relative_to(directory)))
            continue
        payload, meta = path.read_bytes(), json.loads(sidecar.read_text())
        if _sha256(payload) != meta.get('sha256') or recorded.get(meta.get('url')) != meta.get('sha256'):
            mismatched.append(str(path.relative_to(directory)))
            continue
        payloads[key] = payload
    checks = [_result('lineage.raw_integrity', 'error', mismatched, metrics={'raw_files': len(paths)}),
              _result('lineage.raw_availability', 'warning', absent)]
    if mismatched or absent:
        return checks + [_result('lineage.rebuild', 'error', skipped='raw cache incomplete or inconsistent')]
    chart = None
    if not rebuild:
        checks.append(_result('lineage.rebuild', 'error', skipped='rebuild not requested'))
    else:
        try:
            if '/' in symbol:
                rebuilt, _ = md.normalize_fx_payloads(payloads, symbol, start, end)
            else:
                chart = md.yahoo_chart_result(payloads['chart'], symbol)
                rebuilt, _ = md.normalize_yahoo_stock(chart, symbol, start, end)
            pd.testing.assert_frame_equal(validate_bars(rebuilt), stored, check_exact=True)
            difference = []
        except AssertionError as exc:
            difference = [' '.join(str(exc).split())[:300]]
        except (ValueError, KeyError, ImportError) as exc:
            difference = [f'rebuild failed: {exc}']
        checks.append(_result('lineage.rebuild', 'error', difference))
    if '/' not in symbol:
        try:
            chart = chart or md.yahoo_chart_result(payloads['chart'], symbol)
            checks.append(reconcile_yahoo_dividends(chart, start, end, t))
        except (ValueError, KeyError, TypeError) as exc:
            checks.append(_result('reconciliation.vendor_dividends', 'error', [f'unreadable vendor response: {exc}']))
    return checks


def load_exceptions(path: str | Path) -> tuple[list[dict], str]:
    """Reviewed exceptions: who decided that flagged dates are genuine, when and why."""
    payload = Path(path).read_bytes()
    document = json.loads(payload)
    entries = document.get('exceptions') if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise ValueError('Exceptions register must be an object with an "exceptions" list')
    for number, entry in enumerate(entries):
        where = f'exception {number}'
        if not isinstance(entry, dict):
            raise ValueError(f'{where} must be an object')
        for key in ('symbol', 'check', 'reason', 'reviewed_by', 'reviewed_on'):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f'{where} needs a nonempty {key}')
        if entry['check'] not in CHECKS:
            raise ValueError(f"{where}: unknown check {entry['check']!r}")
        dates = entry.get('dates')
        if not isinstance(dates, list) or not dates or not all(
                isinstance(d, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', d) for d in dates + [entry['reviewed_on']]):
            raise ValueError(f'{where} needs dates and reviewed_on as YYYY-MM-DD')
    return entries, _sha256(payload)


def apply_exceptions(symbol: str, checks: list[dict], exceptions: list[dict]) -> None:
    """Mark reviewed flagged dates of failed warnings; a fully reviewed check is 'reviewed'."""
    for check in checks:
        if check['severity'] != 'warning' or check['status'] != 'failed':
            continue
        waived = {d for entry in exceptions if entry['symbol'] == symbol and entry['check'] == check['check']
                  for d in entry['dates']}
        if not waived:
            continue
        reviewed = [item for item in check['flagged'] if item[:10] in waived]
        if reviewed:
            check['reviewed'] = reviewed
            check['unreviewed'] = [item for item in check['flagged'] if item[:10] not in waived]
            if not check['unreviewed'] and not check.get('flagged_truncated'):
                check['status'] = 'reviewed'


def _status(checks: list[dict]) -> str:
    if any(c['severity'] == 'error' and c['status'] == 'failed' for c in checks):
        return 'fail'
    if any(c['severity'] == 'warning' and c['status'] == 'failed' for c in checks):
        return 'warn'
    return 'pass'


def _counts(checks: list[dict]) -> dict:
    return {'errors': sum(c['severity'] == 'error' and c['status'] == 'failed' for c in checks),
            'warnings': sum(c['severity'] == 'warning' and c['status'] == 'failed' for c in checks),
            'reviewed': sum(c['status'] == 'reviewed' for c in checks),
            'passed': sum(c['status'] == 'passed' for c in checks),
            'skipped': sum(c['status'] == 'skipped' for c in checks),
            'noted': sum(c['status'] == 'noted' for c in checks)}


def _default_exceptions():
    path = Path(__file__).resolve().parents[2] / EXCEPTIONS_FILE
    return path if path.exists() else None


def audit_dataset(directory: str | Path, *, ecb=False, reference: str | Path | None = None,
                  exceptions: str | Path | None = None, thresholds: QualityThresholds | None = None,
                  rebuild=True, refresh_reference=False) -> dict:
    """Audit a ``fetch-data`` dataset directory; nothing in it is modified.

    ``ecb`` reconciles FX pairs with ECB reference rates (downloaded once into
    ``raw/ECB`` with a hash sidecar). ``reference`` is a file of independent
    prices (see ``read_reference``). ``exceptions`` defaults to
    ``data-quality-exceptions.json`` in the repository root, if present.
    """
    from .market_data import load_dataset
    t = thresholds or QualityThresholds()
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.exists():
        raise ValueError(f'No dataset manifest in {directory}; run gen-trade fetch-data first')
    manifest_bytes = manifest_path.read_bytes()
    exceptions = exceptions if exceptions is not None else _default_exceptions()
    entries, exceptions_sha = load_exceptions(exceptions) if exceptions else ([], None)
    report = {'schema_version': QUALITY_SCHEMA_VERSION, 'kind': 'gen-trade data quality report',
              'generated_at': datetime.now(timezone.utc).isoformat(), 'dataset': str(directory),
              'manifest_sha256': _sha256(manifest_bytes), 'code_sha256': quality_code_sha256(),
              'thresholds': asdict(t), 'rebuild_from_raw': bool(rebuild),
              'exceptions': ({'path': str(exceptions), 'sha256': exceptions_sha, 'entries': len(entries)}
                             if exceptions else None),
              'reference_sources': [], 'instruments': {}}
    try:
        markets, manifest = load_dataset(directory)
    except (ValueError, OSError, KeyError) as exc:
        report['dataset_checks'] = [_result('contract.dataset', 'error', [str(exc)])]
        report.update(status='fail', counts=_counts(report['dataset_checks']))
        return report
    report['dataset_checks'] = [_result('contract.dataset', 'error')]
    window = (manifest.get('requested_start'), manifest.get('requested_end_exclusive'))
    recorded = (manifest.get('lineage') or {}).get('normalizer_sha256')
    from .market_data import normalizer_sha256
    references = []
    if ecb:
        pairs = [symbol for symbol, frame in markets.items() if asset_class_of(frame) == 'fx']
        if pairs:
            start = window[0] or min(_session_dates(markets[p], 'fx')[0] for p in pairs).strftime('%Y-%m-%d')
            end = window[1] or (max(_session_dates(markets[p], 'fx')[-1] for p in pairs)
                                + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            table, info = ecb_reference(pairs, start, end, directory / 'raw', refresh=refresh_reference)
            references.append((table, info['provider']))
            report['reference_sources'].append(info)
    if reference is not None:
        table = read_reference(reference)
        sources = sorted(table['source'].dropna().astype(str).unique()) if 'source' in table else []
        name = ', '.join(sources) or Path(reference).name
        references.append((table, name))
        report['reference_sources'].append({'provider': name, 'path': str(reference),
                                            'sha256': _sha256(Path(reference).read_bytes()), 'kind': 'file'})
    checks_all = list(report['dataset_checks'])
    for symbol, frame in markets.items():
        checks = audit_frame(frame, thresholds=t, window=window)
        checks += _lineage_checks(directory, manifest, symbol, frame, t, rebuild)
        if manifest['instruments'][symbol].get('raw_sources'):
            checks.append(_result('lineage.normalizer', 'info',
                                  [] if recorded == normalizer_sha256() else
                                  ['normalizer changed since the fetch' if recorded else 'fetch predates normalizer hashing'],
                                  detail='the rebuild check decides whether the change matters'))
        for table, name in references:
            rows = table[table['symbol'] == symbol]
            if len(rows):
                checks += reconcile_reference(frame, rows, t, source=name)
        apply_exceptions(symbol, checks, entries)
        report['instruments'][symbol] = {'status': _status(checks), 'asset_class': asset_class_of(frame),
                                         'rows': len(frame), 'counts': _counts(checks), 'checks': checks}
        checks_all += checks
    report['status'] = max([_status(report['dataset_checks'])] + [i['status'] for i in report['instruments'].values()],
                           key=['pass', 'warn', 'fail'].index)
    report['counts'] = _counts(checks_all)
    return report


def audit_market_file(path: str | Path, *, reference=None, exceptions=None, thresholds=None) -> dict:
    """Audit a CSV/Parquet market file of the generic ``compare`` format, per symbol."""
    t = thresholds or QualityThresholds()
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    if 'symbol' not in frame or frame.empty or frame['symbol'].isna().any():
        raise ValueError('Market-data file needs a symbol on every row')
    exceptions = exceptions if exceptions is not None else _default_exceptions()
    entries, exceptions_sha = load_exceptions(exceptions) if exceptions else ([], None)
    table = read_reference(reference) if reference is not None else None
    report = {'schema_version': QUALITY_SCHEMA_VERSION, 'kind': 'gen-trade data quality report',
              'generated_at': datetime.now(timezone.utc).isoformat(), 'dataset': str(path),
              'source_sha256': _sha256(path.read_bytes()), 'code_sha256': quality_code_sha256(),
              'thresholds': asdict(t), 'rebuild_from_raw': False,
              'exceptions': ({'path': str(exceptions), 'sha256': exceptions_sha, 'entries': len(entries)}
                             if exceptions else None),
              'reference_sources': ([{'provider': Path(reference).name, 'path': str(reference),
                                      'sha256': _sha256(Path(reference).read_bytes()), 'kind': 'file'}]
                                    if reference is not None else []),
              'dataset_checks': [], 'instruments': {}}
    checks_all = []
    for symbol, part in frame.groupby('symbol', sort=False):
        symbol = str(symbol)
        checks = audit_frame(part, thresholds=t)
        if table is not None and checks[0]['status'] == 'passed':
            rows = table[table['symbol'] == symbol]
            if len(rows):
                checks += reconcile_reference(validate_bars(part), rows, t, source=Path(reference).name)
        apply_exceptions(symbol, checks, entries)
        report['instruments'][symbol] = {'status': _status(checks), 'asset_class': asset_class_of(part),
                                         'rows': len(part), 'counts': _counts(checks), 'checks': checks}
        checks_all += checks
    report['status'] = max([i['status'] for i in report['instruments'].values()], key=['pass', 'warn', 'fail'].index)
    report['counts'] = _counts(checks_all)
    return report


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return path


def benchmark_quality(data_directory: str | Path, symbols, *, allow_failures=False) -> dict:
    """Quality-gate summary for a benchmark run, bound to the dataset manifest.

    No ``quality.json``: the run is recorded as not audited. A report for a
    different manifest is stale and refused. Failed error checks on the run's
    instruments refuse the run unless ``allow_failures``, which is recorded.
    """
    directory = Path(data_directory)
    path = directory / QUALITY_FILE
    if not path.exists():
        return {'status': 'not audited',
                'note': 'no quality.json in the dataset directory; run gen-trade audit-data to audit it'}
    payload = path.read_bytes()
    report = json.loads(payload)
    if report.get('manifest_sha256') != _sha256((directory / 'manifest.json').read_bytes()):
        raise ValueError(f'{path} audits a different manifest; rerun gen-trade audit-data {directory}')
    symbols = list(symbols)
    instruments = {symbol: report['instruments'][symbol] for symbol in symbols if symbol in report['instruments']}
    if len(instruments) != len(symbols):
        raise ValueError(f'{path} does not cover every benchmark instrument; rerun gen-trade audit-data')
    failed = [f"dataset: {c['check']}" for c in report.get('dataset_checks', [])
              if c['severity'] == 'error' and c['status'] == 'failed']
    failed += [f"{symbol}: {c['check']} ({c['count']})" for symbol, item in instruments.items()
               for c in item['checks'] if c['severity'] == 'error' and c['status'] == 'failed']
    if failed and not allow_failures:
        raise ValueError('Dataset failed its data-quality audit: ' + '; '.join(failed)
                         + f'. Details in {path}; fix the data, or pass --allow-data-quality-failures to run anyway '
                         '(recorded in the report)')
    status = 'fail' if failed else ('warn' if any(i['status'] == 'warn' for i in instruments.values()) else 'pass')
    return {'status': status, 'report_sha256': _sha256(payload), 'generated_at': report.get('generated_at'),
            'audited_with_current_code': report.get('code_sha256') == quality_code_sha256(),
            'rebuild_from_raw': report.get('rebuild_from_raw'),
            'reference_sources': [s.get('provider') for s in report.get('reference_sources', [])],
            'exceptions_sha256': (report.get('exceptions') or {}).get('sha256'),
            'instruments': {symbol: item['status'] for symbol, item in instruments.items()},
            'unreviewed_warnings': {symbol: [f"{c['check']} ({c['count']})" for c in item['checks']
                                             if c['severity'] == 'warning' and c['status'] == 'failed']
                                    for symbol, item in instruments.items() if item['status'] == 'warn'},
            'failed_error_checks': failed, 'failures_allowed': bool(failed)}


def format_scorecard(report: dict) -> str:
    counts = report['counts']
    lines = [f"Data quality: {report['status'].upper()} for {report['dataset']} - "
             f"{counts['errors']} failed error checks, {counts['warnings']} warnings to review, "
             f"{counts['reviewed']} reviewed"]
    for check in report.get('dataset_checks', []):
        if check['status'] == 'failed':
            lines.append(f"  dataset: {check['check']}: {'; '.join(check['flagged'])}")
    if report['instruments']:
        width = max(10, *(len(s) for s in report['instruments']))
        lines.append(f"  {'Instrument':<{width}}  Status  Rows    Flagged checks")
        for symbol, item in report['instruments'].items():
            flagged = [f"{c['check']} ({c['count']}{', ERROR' if c['severity'] == 'error' else ''})"
                       for c in item['checks'] if c['status'] == 'failed']
            lines.append(f"  {symbol:<{width}}  {item['status']:<6}  {item['rows']:<6}  {', '.join(flagged) or '-'}")
    sources = [s['provider'] for s in report.get('reference_sources', [])]
    if sources:
        lines.append('  Reconciled against: ' + '; '.join(sources))
    return '\n'.join(lines)
