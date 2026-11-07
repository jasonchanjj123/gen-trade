"""Data-quality audit, fetch hardening and the benchmark quality gate."""
from __future__ import annotations

import hashlib
import io
import json
import lzma
import shutil
import struct
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

import gen_trade.market_data as md
from gen_trade.cli import main
from gen_trade.data_quality import (QualityThresholds, apply_exceptions, audit_dataset, audit_frame, audit_market_file,
                                    benchmark_quality, ecb_pair_rates, load_exceptions, parse_ecb_csv,
                                    reconcile_reference, reconcile_yahoo_dividends, write_report)

START, END = '2020-01-01', '2021-01-01'


def yahoo_document(*, symbol='TEST', start=START, end=END, seed=0, dividends=(40, 100), split=None,
                   unadjusted_split=None, meta=None):
    """Synthetic Yahoo chart response over XNYS sessions (split-adjusted OHLC, adjclose with dividends)."""
    import exchange_calendars as xcals
    calendar = xcals.get_calendar('XNYS', start=start, end=(pd.Timestamp(end) - pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    schedule = calendar.schedule.loc[start:end]
    schedule = schedule.loc[schedule.index < pd.Timestamp(end)]
    n, rng = len(schedule), np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    opening = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, .002, n))
    if unadjusted_split is not None:
        # The vendor's history ignores a real split: raw prices halve on that day.
        opening[unadjusted_split:] /= 2
        close[unadjusted_split:] /= 2
    high = np.maximum(opening, close) * (1 + np.abs(rng.normal(0, .003, n)))
    low = np.minimum(opening, close) * (1 - np.abs(rng.normal(0, .003, n)))
    volume = rng.integers(800_000, 1_200_000, n).astype(float)
    stamps = [int(t.timestamp()) for t in schedule['open']]
    events = {'dividends': {}, 'splits': {}}
    factor = np.ones(n)
    for k in dividends:
        amount = round(.005 * close[k - 1], 4)
        events['dividends'][str(stamps[k])] = {'date': stamps[k], 'amount': amount}
        factor[:k] *= 1 - amount / close[k - 1]
    for k in [split, unadjusted_split]:
        if k is not None:
            events['splits'][str(stamps[k])] = {'date': stamps[k], 'numerator': 2, 'denominator': 1}
    meta = {'currency': 'USD', 'symbol': symbol, 'instrumentType': 'EQUITY',
            'exchangeTimezoneName': 'America/New_York', 'dataGranularity': '1d', **(meta or {})}
    return {'meta': meta, 'timestamp': stamps, 'events': events,
            'indicators': {'quote': [{'open': opening.tolist(), 'high': high.tolist(), 'low': low.tolist(),
                                      'close': close.tolist(), 'volume': volume.tolist()}],
                           'adjclose': [{'adjclose': (close * factor).tolist()}]}}


def duka_payloads(year=2020, divisor=100000, seed=1, partial=None):
    """Synthetic Dukascopy yearly BID/ASK daily candles; Saturdays and closed days are flat placeholders."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
    price, bid, ask = 1.1, [], []
    for day in days:
        seconds = int((day - pd.Timestamp(f'{year}-01-01')).total_seconds())
        closed = day.day_of_week == 5 or (day.month, day.day) in ((1, 1), (12, 25))
        if closed:
            o = h = l = c = round(price * divisor)
            volume = 0.
        else:
            nxt = price * np.exp(rng.normal(0, .004))
            o, c = round(price * divisor), round(nxt * divisor)
            h, l = round(max(price, nxt) * 1.001 * divisor), round(min(price, nxt) * .999 * divisor)
            volume = 1e3 if day.day_of_week == 6 else 1e5
            if partial is not None and day == pd.Timestamp(partial):
                h, l, volume = max(o, c) + 1, min(o, c) - 1, 20.
            price = nxt
        bid.append(struct.pack('>5If', seconds, o, c, l, h, volume))
        ask.append(struct.pack('>5If', seconds, o + 2, c + 2, l + 2, h + 2, volume))
    return lzma.compress(b''.join(bid)), lzma.compress(b''.join(ask))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def serve(monkeypatch, payloads):
    """Route urlopen to in-memory payloads keyed by URL fragment; record every request."""
    calls = []

    def opener(request, timeout=None):
        url = request.full_url
        calls.append(url)
        for fragment, payload in payloads.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return _Response(payload)
        raise HTTPError(url, 404, 'not found', {}, None)
    monkeypatch.setattr(md, 'urlopen', opener)
    monkeypatch.setattr(md, 'DOWNLOAD_BACKOFF_SECONDS', 0.)
    return calls


def default_payloads(**fx):
    bid, ask = duka_payloads(**fx)
    return {'chart/TEST?': json.dumps({'chart': {'result': [yahoo_document()]}}).encode(),
            'EURUSD/2020/BID': bid, 'EURUSD/2020/ASK': ask}


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    directory = tmp_path_factory.mktemp('dataset')
    monkeypatch = pytest.MonkeyPatch()
    try:
        serve(monkeypatch, default_payloads())
        md.fetch_dataset(directory, start=START, end=END, stocks=('TEST',), fx_pairs=('EUR/USD',))
    finally:
        monkeypatch.undo()
    return directory


@pytest.fixture
def dataset(built, tmp_path):
    target = tmp_path / 'dataset'
    shutil.copytree(built, target)
    return target


def checks_of(report, symbol):
    return {c['check']: c for c in report['instruments'][symbol]['checks']}


def rewrite(directory, symbol, frame):
    """Replace a dataset file and its manifest hash, as edited or stale derived data would look."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    path = directory / manifest['instruments'][symbol]['file']
    frame.to_parquet(path, index=False)
    manifest['instruments'][symbol]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / 'manifest.json').write_text(json.dumps(manifest))


def test_clean_dataset_passes_every_check_including_rebuild_and_vendor_dividends(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    assert manifest['lineage']['normalizer_sha256'] == md.normalizer_sha256()
    report = audit_dataset(dataset)
    assert report['status'] == 'pass', [(s, c['check'], c['flagged']) for s, i in report['instruments'].items()
                                        for c in i['checks'] if c['status'] == 'failed']
    stock, fx = checks_of(report, 'TEST'), checks_of(report, 'EUR/USD')
    for name in ('lineage.raw_integrity', 'lineage.rebuild', 'contract.signal_prices', 'completeness.sessions'):
        assert stock[name]['status'] == fx[name]['status'] == 'passed'
    assert stock['reconciliation.vendor_dividends']['status'] == 'passed'
    assert stock['reconciliation.vendor_dividends']['metrics']['events'] == 2
    assert fx['completeness.sessions']['metrics']['closed_days_excluded'] == ['01-01', '12-25']
    assert report['manifest_sha256'] == hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest()


def test_tampered_raw_cache_and_stale_derived_data_fail_lineage(dataset):
    raw = md.fx_raw_path(dataset, 'EUR/USD', 2020, 'BID')
    raw.write_bytes(raw.read_bytes() + b'\0')
    report = audit_dataset(dataset)
    fx = checks_of(report, 'EUR/USD')
    assert fx['lineage.raw_integrity']['status'] == 'failed' and fx['lineage.rebuild']['status'] == 'skipped'
    assert report['status'] == 'fail'
    shutil.rmtree(dataset / 'raw' / 'EURUSD')
    fx = checks_of(audit_dataset(dataset), 'EUR/USD')
    assert fx['lineage.raw_availability']['status'] == 'failed' and fx['lineage.raw_integrity']['status'] == 'passed'


def test_rebuild_detects_edited_dataset_files_even_with_a_matching_manifest_hash(dataset):
    markets, _ = md.load_dataset(dataset)
    frame = markets['EUR/USD'].copy()
    for name in ('close', 'bid_close'):
        frame.loc[10, name] = frame.loc[10, name] * 1.0001
    frame.loc[10, 'bid_high'] = frame.loc[10, 'high'] = max(frame.loc[10, 'high'], frame.loc[10, 'close'])
    frame.loc[10, 'ask_close'] = max(frame.loc[10, 'ask_close'], frame.loc[10, 'bid_close'])
    frame.loc[10, 'ask_high'] = max(frame.loc[10, 'ask_high'], frame.loc[10, 'ask_close'])
    rewrite(dataset, 'EUR/USD', frame)
    report = audit_dataset(dataset)
    assert checks_of(report, 'EUR/USD')['lineage.rebuild']['status'] == 'failed'
    assert report['status'] == 'fail'
    assert audit_dataset(dataset, rebuild=False)['status'] != 'fail'


def test_contract_violations_are_errors():
    frame, _ = md.normalize_yahoo_stock(yahoo_document(), 'TEST', START, END)
    broken = frame.copy()
    broken.loc[5, 'close'] *= 1.01
    broken.loc[5, 'high'] = max(broken.loc[5, 'high'], broken.loc[5, 'close'])
    broken.loc[7, 'dividend'] = -1.
    checks = {c['check']: c for c in audit_frame(broken)}
    assert checks['contract.signal_prices']['flagged'] == [broken.loc[5, 'session_date']]
    assert checks['contract.corporate_actions']['flagged'] == [broken.loc[7, 'session_date']]
    missing = {c['check']: c for c in audit_frame(frame.drop(columns='execution_open'))}
    assert missing['contract.schema']['flagged'] == ['missing execution_open']
    impossible = frame.copy()
    impossible.loc[3, 'low'] = impossible.loc[3, 'high'] * 2
    assert audit_frame(impossible)[0]['status'] == 'failed'
    fx, _ = md.normalize_fx_payloads(dict(zip([(2020, 'BID'), (2020, 'ASK')], duka_payloads())), 'EUR/USD', START, END)
    crossed = fx.copy()
    crossed.loc[4, 'ask_open'] = crossed.loc[4, 'bid_open'] * .999
    crossed.loc[4, 'ask_low'] = min(crossed.loc[4, 'ask_low'], crossed.loc[4, 'ask_open'])
    checks = {c['check']: c for c in audit_frame(crossed)}
    assert checks['contract.bid_ask']['flagged'] == [crossed.loc[4, 'session_date']]
    assert checks['accuracy.intraday_crossed']['status'] == 'failed'


def test_missing_xnys_session_is_an_error_and_missing_fx_weekday_a_warning():
    stock, _ = md.normalize_yahoo_stock(yahoo_document(), 'TEST', START, END)
    gone = stock.drop(index=50).reset_index(drop=True)
    sessions = {c['check']: c for c in audit_frame(gone, window=(START, END))}['completeness.sessions']
    assert sessions['severity'] == 'error' and sessions['flagged'] == [f"missing {stock.loc[50, 'session_date']}"]
    fx, _ = md.normalize_fx_payloads(dict(zip([(2020, 'BID'), (2020, 'ASK')], duka_payloads())), 'EUR/USD', START, END)
    wednesday = fx.index[fx['session_date'] == '2020-06-10'][0]
    checks = {c['check']: c for c in audit_frame(fx.drop(index=wednesday).reset_index(drop=True), window=(START, END))}
    assert checks['completeness.sessions']['severity'] == 'warning'
    assert checks['completeness.sessions']['flagged'] == ['missing 2020-06-10']
    truncated = fx.iloc[:-30].reset_index(drop=True)
    assert {c['check']: c for c in audit_frame(truncated, window=(START, END))}['timeliness.window']['status'] == 'failed'


def test_partial_fx_candle_is_flagged_but_sundays_and_closed_days_are_not():
    payloads = dict(zip([(2020, 'BID'), (2020, 'ASK')], duka_payloads(partial='2020-06-10')))
    fx, _ = md.normalize_fx_payloads(payloads, 'EUR/USD', START, END)
    checks = {c['check']: c for c in audit_frame(fx, window=(START, END))}
    assert checks['completeness.partial_bars']['flagged'] == ['2020-06-10']
    assert checks['accuracy.spreads']['metrics']['median_open_spread_bps_by_weekday']['wed'] > 0


def test_spike_reversal_stale_run_and_outlier_screens():
    frame, _ = md.normalize_yahoo_stock(yahoo_document(dividends=()), 'TEST', START, END)
    bad = frame.copy()
    for prefix in ('', 'execution_'):
        bad.loc[100, f'{prefix}close'] *= 1.4
        bad.loc[100, f'{prefix}high'] = bad.loc[100, f'{prefix}close']
        bad.loc[101, f'{prefix}open'] = bad.loc[100, f'{prefix}close']
        bad.loc[101, f'{prefix}high'] = max(bad.loc[101, f'{prefix}high'], bad.loc[101, f'{prefix}open'])
    for row in (150, 151, 152):
        for prefix in ('', 'execution_'):
            bad.loc[row, f'{prefix}close'] = bad.loc[149, f'{prefix}close']
            bad.loc[row, f'{prefix}high'] = max(bad.loc[row, f'{prefix}high'], bad.loc[row, f'{prefix}close'],
                                                bad.loc[row, f'{prefix}open'])
            bad.loc[row, f'{prefix}low'] = min(bad.loc[row, f'{prefix}low'], bad.loc[row, f'{prefix}close'],
                                               bad.loc[row, f'{prefix}open'])
    checks = {c['check']: c for c in audit_frame(bad)}
    date = bad.loc[100, 'session_date']
    assert checks['accuracy.spike_reversals']['flagged'] == [date]
    assert date in checks['accuracy.return_outliers']['flagged']
    assert checks['accuracy.stale_prices']['flagged'] == [bad.loc[151, 'session_date'], bad.loc[152, 'session_date']]
    clean = {c['check']: c for c in audit_frame(frame)}
    assert all(clean[name]['status'] == 'passed'
               for name in ('accuracy.spike_reversals', 'accuracy.return_outliers', 'accuracy.stale_prices'))


def test_unadjusted_split_history_and_unrecorded_split_jumps():
    unadjusted, _ = md.normalize_yahoo_stock(yahoo_document(unadjusted_split=120), 'TEST', START, END)
    checks = {c['check']: c for c in audit_frame(unadjusted)}
    assert checks['consistency.split_dates']['flagged'] == [unadjusted.loc[120, 'session_date']]
    assert checks['consistency.split_dates']['severity'] == 'error'
    document = yahoo_document(unadjusted_split=120)
    document['events']['splits'] = {}
    unrecorded, _ = md.normalize_yahoo_stock(document, 'TEST', START, END)
    checks = {c['check']: c for c in audit_frame(unrecorded)}
    assert checks['consistency.split_like_jumps']['flagged'] == [unrecorded.loc[120, 'session_date']]
    adjusted, _ = md.normalize_yahoo_stock(yahoo_document(split=120), 'TEST', START, END)
    checks = {c['check']: c for c in audit_frame(adjusted)}
    assert checks['consistency.split_dates']['status'] == checks['consistency.split_like_jumps']['status'] == 'passed'


def test_vendor_dividends_reconcile_with_adjusted_close():
    document = yahoo_document()
    assert reconcile_yahoo_dividends(document, START, END)['status'] == 'passed'
    dropped = json.loads(json.dumps(document))
    dropped['events']['dividends'].pop(next(iter(dropped['events']['dividends'])))
    result = reconcile_yahoo_dividends(dropped, START, END)
    assert result['status'] == 'failed' and 'without a dividend event' in result['flagged'][0]
    orphan = json.loads(json.dumps(document))
    stamp = document['timestamp'][200]
    orphan['events']['dividends']['x'] = {'date': stamp, 'amount': .5}
    result = reconcile_yahoo_dividends(orphan, START, END)
    assert result['status'] == 'failed' and 'event 0.5' in result['flagged'][0]
    del document['indicators']['adjclose']
    assert reconcile_yahoo_dividends(document, START, END)['status'] == 'skipped'


ECB_CSV = (b'KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n'
           b'EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2020-06-10,1.1300\n'
           b'EXR.D.JPY.EUR.SP00.A,D,JPY,EUR,SP00,A,2020-06-10,122.00\n'
           b'EXR.D.GBP.EUR.SP00.A,D,GBP,EUR,SP00,A,2020-06-10,0.89000\n')


def test_ecb_cross_rates():
    rates = parse_ecb_csv(ECB_CSV)
    assert ecb_pair_rates(rates, 'EUR/USD').iloc[0] == pytest.approx(1.13)
    assert ecb_pair_rates(rates, 'USD/JPY').iloc[0] == pytest.approx(122 / 1.13)
    assert ecb_pair_rates(rates, 'GBP/USD').iloc[0] == pytest.approx(1.13 / .89)
    with pytest.raises(ValueError, match='CHF'):
        ecb_pair_rates(rates, 'USD/CHF')
    with pytest.raises(ValueError):
        parse_ecb_csv(ECB_CSV.replace(b',EUR,', b',USD,'))


def test_ecb_reconciliation_flags_fixes_outside_the_daily_range_and_uncovered_days(dataset, monkeypatch):
    markets, _ = md.load_dataset(dataset)
    fx = markets['EUR/USD']
    inside = (fx['bid_low'] + fx['ask_high']) / 2
    rows = [f'EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,{d},{v:.6f}' for d, v in zip(fx['session_date'], inside)
            if pd.Timestamp(d).day_of_week < 5]
    outside = fx.index[fx['session_date'] == '2020-03-11'][0]
    rows = [row if '2020-03-11' not in row else row.rsplit(',', 1)[0] + f",{fx.loc[outside, 'ask_high'] * 1.01:.6f}"
            for row in rows] + ['EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2020-12-25,1.2']
    csv = ('KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n' + '\n'.join(rows)).encode()
    calls = serve(monkeypatch, {'data-api.ecb.europa.eu': csv})
    report = audit_dataset(dataset, ecb=True)
    fx_checks = checks_of(report, 'EUR/USD')
    assert [f[:10] for f in fx_checks['reconciliation.reference_prices']['flagged']] == ['2020-03-11']
    assert fx_checks['completeness.reference_coverage']['flagged'] == ['no bar on 2020-12-25']
    assert 'D.USD.EUR' in calls[0] and report['reference_sources'][0]['sha256']
    audit_dataset(dataset, ecb=True)
    assert len(calls) == 1  # hash-checked cache reused


def test_reference_file_reconciles_stock_closes(dataset, tmp_path):
    markets, _ = md.load_dataset(dataset)
    stock = markets['TEST']
    reference = pd.DataFrame({'symbol': 'TEST', 'session_date': stock['session_date'],
                              'value': stock['execution_close'], 'kind': 'close', 'source': 'vendor X'})
    reference.loc[30, 'value'] *= 1.01
    path = tmp_path / 'reference.csv'
    reference.to_csv(path, index=False)
    report = audit_dataset(dataset, reference=path)
    check = checks_of(report, 'TEST')['reconciliation.reference_prices']
    assert check['count'] == 1 and check['flagged'][0].startswith(stock.loc[30, 'session_date'])
    assert check['metrics']['source'] == 'vendor X' and check['metrics']['matched'] == len(stock)
    reference.loc[31, 'kind'] = 'midpoint'
    reference.to_csv(path, index=False)
    with pytest.raises(ValueError, match='kind'):
        audit_dataset(dataset, reference=path)


def test_reviewed_exceptions_clear_warnings_with_an_audit_trail(dataset, tmp_path):
    payloads = dict(zip([(2020, 'BID'), (2020, 'ASK')], duka_payloads(partial='2020-06-10')))
    fx, _ = md.normalize_fx_payloads(payloads, 'EUR/USD', START, END)
    checks = audit_frame(fx, window=(START, END))
    entry = {'symbol': 'EUR/USD', 'check': 'completeness.partial_bars', 'dates': ['2020-06-10'],
             'reason': 'vendor feed outage; reviewed against a second source', 'reviewed_by': 'data steward',
             'reviewed_on': '2026-11-07'}
    path = tmp_path / 'exceptions.json'
    path.write_text(json.dumps({'exceptions': [entry]}))
    entries, digest = load_exceptions(path)
    apply_exceptions('EUR/USD', checks, entries)
    partial = {c['check']: c for c in checks}['completeness.partial_bars']
    assert partial['status'] == 'reviewed' and partial['reviewed'] == ['2020-06-10']
    apply_exceptions('GBP/USD', audit_frame(fx, window=(START, END)), entries)
    report = audit_dataset(dataset, exceptions=path)
    assert report['exceptions']['sha256'] == digest
    for broken in ({**entry, 'reason': ''}, {**entry, 'check': 'made.up'}, {**entry, 'dates': ['10/06/2020']}):
        path.write_text(json.dumps({'exceptions': [broken]}))
        with pytest.raises(ValueError):
            load_exceptions(path)


def test_benchmark_gate_requires_a_current_passing_audit(dataset):
    assert benchmark_quality(dataset, ['TEST'])['status'] == 'not audited'
    write_report(audit_dataset(dataset), dataset / 'quality.json')
    gate = benchmark_quality(dataset, ['TEST', 'EUR/USD'])
    assert gate['status'] == 'pass' and gate['audited_with_current_code'] and not gate['failures_allowed']
    markets, _ = md.load_dataset(dataset)
    frame = markets['TEST'].copy()
    frame.loc[5, 'dividend'] = -1.
    rewrite(dataset, 'TEST', frame)
    with pytest.raises(ValueError, match='different manifest'):
        benchmark_quality(dataset, ['TEST'])
    write_report(audit_dataset(dataset), dataset / 'quality.json')
    with pytest.raises(ValueError, match='contract.corporate_actions'):
        benchmark_quality(dataset, ['TEST'])
    assert benchmark_quality(dataset, ['EUR/USD'])['status'] == 'pass'
    allowed = benchmark_quality(dataset, ['TEST'], allow_failures=True)
    assert allowed['status'] == 'fail' and allowed['failures_allowed']


def test_benchmark_archives_the_audit_and_verifier_checks_it(tmp_path, monkeypatch):
    from test_setup import _dataset, _fake_methods, _verifier
    from gen_trade.benchmark import run_benchmark
    _fake_methods(monkeypatch)
    data, output = tmp_path / 'data', tmp_path / 'out'
    _dataset(data)
    write_report(audit_dataset(data), data / 'quality.json')
    options = dict(population_size=4, generations=0, seeds=(1,), workers=1, first_test_year=2019, last_test_year=2020,
                   bootstrap_reps=100, random_timing_reps=10, spa_reps=20, progress='off', symbols=['LIQ'])
    with pytest.raises(ValueError, match='data-quality audit'):
        run_benchmark(data, output, **options)
    summary = run_benchmark(data, output, allow_data_quality_failures=True, **options)
    assert summary['data_quality']['failures_allowed'] and summary['data_quality']['status'] == 'fail'
    assert '## Data quality' in (output / 'report.md').read_text()
    verify = _verifier()
    assert verify(output, data)['status'] == 'passed'
    archived = output / 'data-quality.json'
    archived.write_text(archived.read_text() + ' ')
    with pytest.raises(ValueError, match='Archived data-quality audit'):
        verify(output, data)


def test_cli_audit_writes_report_and_exits_by_status(dataset, capsys):
    assert main(['audit-data', str(dataset)]) == 0
    assert json.loads((dataset / 'quality.json').read_text())['status'] == 'pass'
    assert 'Data quality: PASS' in capsys.readouterr().out
    markets, _ = md.load_dataset(dataset)
    frame = markets['TEST'].copy()
    frame.loc[5, 'split_ratio'] = 0.
    rewrite(dataset, 'TEST', frame)
    assert main(['audit-data', str(dataset), '--no-rebuild']) == 3


def test_generic_market_file_audit_labels_intraday_rows(tmp_path):
    stamps = pd.date_range('2024-01-01', periods=300, freq='h', tz='UTC')
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .002, len(stamps))))
    frame = pd.DataFrame({'symbol': 'BTC/USDT', 'timestamp': stamps, 'available_at': stamps + pd.Timedelta(hours=1),
                          'open': np.r_[close[0], close[:-1]], 'close': close, 'volume': 10.})
    frame['high'] = frame[['open', 'close']].max(axis=1) * 1.001
    frame['low'] = frame[['open', 'close']].min(axis=1) * .999
    frame.loc[200, 'volume'] = 0.
    path = tmp_path / 'bars.csv'
    frame.to_csv(path, index=False)
    report = audit_market_file(path)
    checks = checks_of(report, 'BTC/USDT')
    assert checks['completeness.partial_bars']['flagged'] == ['2024-01-09T08:00:00Z']
    assert report['status'] == 'warn' and report['source_sha256']


def test_resume_on_a_later_day_reuses_cached_yahoo_response(dataset, monkeypatch):
    (dataset / 'manifest.json').unlink()
    for side in ('BID',):
        path = md.fx_raw_path(dataset, 'EUR/USD', 2020, side)
        path.unlink()
        path.with_suffix(path.suffix + '.meta.json').unlink()
    later = md._utcnow() + pd.Timedelta(days=3)
    monkeypatch.setattr(md, '_utcnow', lambda: later)
    calls = serve(monkeypatch, default_payloads())
    manifest = md.fetch_dataset(dataset, start=START, end=END, stocks=('TEST',), fx_pairs=('EUR/USD',))
    assert calls == ['https://datafeed.dukascopy.com/datafeed/EURUSD/2020/BID_candles_day_1.bi5']
    assert manifest['universe'] == ['TEST', 'EUR/USD']
    # A cached response retrieved before the requested end is a different request.
    (dataset / 'manifest.json').unlink()
    beyond = (later + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    with pytest.raises(ValueError, match='different request'):
        md.fetch_dataset(dataset, start=START, end=beyond, stocks=('TEST',), fx_pairs=())


def test_download_retries_transient_errors_and_not_missing_files(tmp_path, monkeypatch):
    attempts = {'n': 0}

    def flaky(request, timeout=None):
        attempts['n'] += 1
        if attempts['n'] < 3:
            raise HTTPError(request.full_url, 503, 'busy', {}, None)
        return _Response(b'payload')
    monkeypatch.setattr(md, 'urlopen', flaky)
    payload, info = md._download('https://example.test/a', tmp_path / 'a.bin', backoff=0)
    assert payload == b'payload' and attempts['n'] == 3 and info['bytes'] == 7
    calls = serve(monkeypatch, {})
    with pytest.raises(RuntimeError, match='404'):
        md._download('https://example.test/missing', tmp_path / 'b.bin', backoff=0)
    assert len(calls) == 1
    serve(monkeypatch, {'empty': b''})
    with pytest.raises(RuntimeError, match='empty'):
        md._download('https://example.test/empty', tmp_path / 'c.bin', backoff=0)


@pytest.mark.parametrize('meta,match', [({'currency': 'EUR'}, 'currency'), ({'symbol': 'OTHER'}, 'symbol'),
                                        ({'exchangeTimezoneName': 'Europe/London'}, 'exchangeTimezoneName'),
                                        ({'dataGranularity': '1wk'}, 'dataGranularity')])
def test_yahoo_metadata_must_describe_a_usd_daily_new_york_equity(meta, match):
    with pytest.raises(ValueError, match=match):
        md.normalize_yahoo_stock(yahoo_document(meta=meta), 'TEST', START, END)


def test_manifest_structure_is_validated(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    for change, match in ((lambda m: m['instruments']['TEST'].update(file='../TEST.parquet'), 'plain file name'),
                          (lambda m: m.update(schema_version=2), 'schema_version'),
                          (lambda m: m['instruments']['TEST'].pop('sha256'), 'sha256'),
                          (lambda m: m['instruments']['TEST'].update(file='absent.parquet'), 'missing')):
        broken = json.loads(json.dumps(manifest))
        change(broken)
        (dataset / 'manifest.json').write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=match):
            md.load_dataset(dataset)
    report = audit_dataset(dataset)
    assert report['status'] == 'fail' and report['dataset_checks'][0]['status'] == 'failed'


def test_reference_reconciliation_skips_intraday_and_thresholds_validate():
    stamps = pd.date_range('2024-01-01', periods=48, freq='h', tz='UTC')
    bars = pd.DataFrame({'symbol': 'X', 'timestamp': stamps, 'available_at': stamps + pd.Timedelta(hours=1),
                         'open': 1., 'high': 1., 'low': 1., 'close': 1., 'volume': 1.})
    reference = pd.DataFrame({'symbol': ['X'], 'session_date': ['2024-01-01'], 'value': [1.], 'kind': ['fix']})
    assert all(c['status'] == 'skipped' for c in reconcile_reference(bars, reference))
    with pytest.raises(ValueError):
        QualityThresholds(outlier_z=-1)
