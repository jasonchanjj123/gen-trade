"""Auditable daily Yahoo equity / Dukascopy bid-ask acquisition.

Raw responses are cached with SHA-256 checks. Yahoo split-adjusted prices are
reconstructed into contemporaneous execution prices; indicator prices use only
splits already effective at each observation. Dividends are separate cash events.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import lzma
from pathlib import Path
import struct
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .data import validate_bars

DEFAULT_STOCKS = ('AAPL','GM','GRPN')
DEFAULT_FX = ('EUR/USD','GBP/USD','USD/JPY')


def _download(url: str, path: Path, refresh=False) -> tuple[bytes, dict]:
    info_path = path.with_suffix(path.suffix+'.meta.json')
    if not refresh and path.exists() and info_path.exists():
        payload, info = path.read_bytes(), json.loads(info_path.read_text())
        if hashlib.sha256(payload).hexdigest() != info['sha256'] or info['url'] != url:
            raise ValueError(f'Raw cache integrity mismatch: {path}')
        return payload, info
    error = None
    for _ in range(2):
        try:
            request = Request(url,headers={'User-Agent':'Mozilla/5.0 (compatible; gen-trade-research)'})
            with urlopen(request,timeout=25) as response:
                payload = response.read()
            info = {'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),
                    'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary = path.with_suffix(path.suffix+'.tmp')
            temporary.write_bytes(payload)
            temporary.replace(path)
            info_path.write_text(json.dumps(info,indent=2)+'\n')
            return payload, info
        except (OSError, ValueError) as exc:
            error = exc
    raise RuntimeError(f'Historical data request failed for {url}: {error}') from error


def normalize_yahoo_stock(result: dict, symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    import exchange_calendars as xcals
    stamps = pd.to_datetime(result['timestamp'],unit='s',utc=True)
    dates = stamps.tz_convert('America/New_York').tz_localize(None).normalize()
    quote = result['indicators']['quote'][0]
    source = pd.DataFrame({name:quote[name] for name in ('open','high','low','close','volume')},index=dates)
    if source.index.duplicated().any() or not source.index.is_monotonic_increasing:
        raise ValueError(f'{symbol}: duplicate or unordered source dates')
    events = result.get('events',{})
    splits = {}
    for event in events.get('splits',{}).values():
        date = pd.Timestamp(event['date'],unit='s',tz='UTC').tz_convert('America/New_York').tz_localize(None).normalize()
        ratio = float(event['numerator'])/float(event['denominator'])
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError('Invalid split ratio')
        splits[date] = splits.get(date,1.)*ratio
    dividends = {}
    for event in events.get('dividends',{}).values():
        date = pd.Timestamp(event['date'],unit='s',tz='UTC').tz_convert('America/New_York').tz_localize(None).normalize()
        dividends[date] = dividends.get(date,0.) + float(event['amount'])
    requested = source.loc[(source.index >= pd.Timestamp(start)) & (source.index < pd.Timestamp(end))].copy()
    if requested.empty:
        raise ValueError(f'{symbol}: no observations in requested interval')
    calendar = xcals.get_calendar('XNYS',start=start,end=end)
    schedule = calendar.schedule.loc[start:end]
    schedule = schedule.loc[schedule.index < pd.Timestamp(end)]
    missing = schedule.index.difference(requested.index)
    unexpected = requested.index.difference(schedule.index)
    if len(missing) or len(unexpected):
        raise ValueError(f'{symbol}: session coverage mismatch: missing={list(missing.strftime("%Y-%m-%d"))}, unexpected={list(unexpected.strftime("%Y-%m-%d"))}')
    if requested.isna().any().any():
        raise ValueError(f'{symbol}: source has missing OHLCV; no prices are interpolated')
    future_factor = np.array([np.prod([r for day,r in splits.items() if day > date]) for date in requested.index])
    effective = np.array([splits.get(date,1.) for date in requested.index])
    past_factor = np.cumprod(effective)
    output = pd.DataFrame({'symbol':symbol,'session_date':requested.index.strftime('%Y-%m-%d'),
                           'timestamp':schedule.loc[requested.index,'open'].array,
                           'available_at':schedule.loc[requested.index,'close'].array})
    for name in ('open','high','low','close'):
        raw = requested[name].to_numpy()*future_factor
        output[f'execution_{name}'] = raw
        output[name] = raw*past_factor
    output['volume'] = requested['volume'].to_numpy()/future_factor/past_factor
    output['execution_volume'] = requested['volume'].to_numpy()/future_factor
    output['split_ratio'] = effective
    output['dividend'] = np.array([dividends.get(day,0.) for day in requested.index])*future_factor
    output['asset_class'] = 'stock'
    output = validate_bars(output)
    return output, {'provider':'Yahoo Finance chart','source_currency':result['meta'].get('currency'),
        'calendar':'XNYS regular sessions including early closes',
        'execution_prices':'contemporaneous prices reconstructed from split-adjusted provider OHLC',
        'signal_prices':'raw execution OHLC multiplied by cumulative already-effective splits; no dividend adjustment',
        'corporate_actions':'quantity/cost-basis adjustment on effective split date; dividend entitlement credited ex-date',
        'split_events':[{'date':d.strftime('%Y-%m-%d'),'ratio':r} for d,r in sorted(splits.items()) if pd.Timestamp(start)<=d<pd.Timestamp(end)],
        'dividend_events':int(np.count_nonzero(output['dividend'])),
        'missing_sessions':[], 'unexpected_sessions':[]}


def decode_duka_daily(payload: bytes, year: int, divisor: int) -> pd.DataFrame:
    raw = lzma.decompress(payload)
    if len(raw)%24:
        raise ValueError('Dukascopy candle payload is not a whole number of 24-byte records')
    records = list(struct.iter_unpack('>5If',raw))
    frame = pd.DataFrame(records,columns=['seconds','open','close','low','high','feed_volume'])
    frame['timestamp'] = pd.Timestamp(f'{year}-01-01',tz='UTC')+pd.to_timedelta(frame.pop('seconds'),unit='s')
    for name in ('open','high','low','close'):
        frame[name] = frame[name]/divisor
    if frame['timestamp'].duplicated().any() or not frame['timestamp'].is_monotonic_increasing:
        raise ValueError('Dukascopy daily record timestamps must be unique and chronological')
    return frame


def normalize_duka_fx(bid: pd.DataFrame, ask: pd.DataFrame, symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    if not bid['timestamp'].equals(ask['timestamp']):
        raise ValueError(f'{symbol}: bid and ask history is not identically aligned')
    frame = pd.DataFrame({'symbol':symbol,'timestamp':bid['timestamp']})
    for side, source in (('bid',bid),('ask',ask)):
        for name in ('open','high','low','close'):
            frame[f'{side}_{name}'] = source[name].to_numpy()
        frame[f'{side}_feed_volume'] = source['feed_volume'].to_numpy()
    prices = frame[[f'{side}_{field}' for side in ('bid','ask') for field in ('open','high','low','close')]]
    empty = (prices == 0).all(axis=1)
    flat = ((bid[['open','high','low','close']].nunique(axis=1)==1)
            & (ask[['open','high','low','close']].nunique(axis=1)==1)
            & (bid['feed_volume']==0) & (ask['feed_volume']==0))
    keep_range = (frame['timestamp'] >= pd.Timestamp(start,tz='UTC')) & (frame['timestamp'] < pd.Timestamp(end,tz='UTC'))
    excluded = int(((empty | flat) & keep_range).sum())
    frame = frame.loc[keep_range & ~(empty|flat)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f'{symbol}: no observed quotes')
    crossed = np.zeros(len(frame),dtype=bool)
    for name in ('open','high','low','close'):
        crossed |= (frame[f'ask_{name}'] < frame[f'bid_{name}']).to_numpy()
    quarantined = frame.loc[crossed,'timestamp'].dt.strftime('%Y-%m-%d').tolist()
    if crossed.mean() > .01:
        raise ValueError(f'{symbol}: more than 1% of observed daily quote records are crossed')
    frame = frame.loc[~crossed].reset_index(drop=True)
    for name in ('open','high','low','close'):
        frame[name] = frame[f'bid_{name}']
    frame['available_at'] = frame['timestamp']+pd.Timedelta(days=1)
    frame['session_date'] = frame['timestamp'].dt.strftime('%Y-%m-%d')
    frame['volume'] = np.nan  # Broker-side activity is not consolidated spot-FX volume.
    frame['asset_class'] = 'fx'
    frame['split_ratio'], frame['dividend'] = 1., 0.
    frame = validate_bars(frame)
    return frame, {'provider':'Dukascopy historical bid/ask daily candles','bar_timezone':'UTC',
        'signal_prices':'observed bid OHLC','execution_prices':'separate historical bid and ask opens/closes',
        'volume_policy':'feed activity retained separately; consolidated volume unknown; volume-dependent indicator rules excluded',
        'excluded_empty_or_inactive_flat_days':excluded,
        'quarantined_crossed_quote_dates':quarantined,
        'observed_sunday_bars':int((frame['timestamp'].dt.day_of_week==6).sum()),
        'gap_policy':'no interpolation or synthetic holiday/weekend candles'}


def fetch_dataset(directory: str | Path, *, start='2015-01-01', end='2026-01-01',
                  stocks=DEFAULT_STOCKS, fx_pairs=DEFAULT_FX, refresh=False) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if not refresh and (directory/'manifest.json').exists():
        _, manifest=load_dataset(directory)
        if (manifest['requested_start']!=start or manifest['requested_end_exclusive']!=end
                or manifest['universe']!=list(stocks)+list(fx_pairs)):
            raise ValueError('Existing dataset manifest has different dates/universe; use a new directory or explicit refresh')
        print('Verified cached historical dataset manifest.',flush=True)
        return manifest
    tasks = {}
    # Include later split events solely to undo the provider's retrospective split adjustment.
    retrieval_end = max(pd.Timestamp(end,tz='UTC'),pd.Timestamp.now(tz='UTC').ceil('D'))
    t1, t2 = int(pd.Timestamp(start,tz='UTC').timestamp()), int(retrieval_end.timestamp())
    for symbol in stocks:
        query = urlencode({'period1':t1,'period2':t2,'interval':'1d','events':'div,splits','includeAdjustedClose':'true'})
        tasks[('stock',symbol)] = ('https://query1.finance.yahoo.com/v8/finance/chart/'+symbol+'?'+query,
                                    directory/'raw'/f'{symbol}-chart.json')
    years = range(pd.Timestamp(start).year,pd.Timestamp(end).year+(pd.Timestamp(end).dayofyear != 1))
    for pair in fx_pairs:
        instrument = pair.replace('/','')
        for year in years:
            for side in ('BID','ASK'):
                tasks[('fx',pair,year,side)] = (f'https://datafeed.dukascopy.com/datafeed/{instrument}/{year}/{side}_candles_day_1.bi5',
                                              directory/'raw'/instrument/f'{year}-{side}.bi5')
    responses = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_download,url,path,refresh):key for key,(url,path) in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            responses[key] = future.result()
            print(f'Data files verified: {len(responses)}/{len(tasks)}',flush=True)
    normalized, instruments = {}, {}
    for symbol in stocks:
        payload, source = responses[('stock',symbol)]
        document = json.loads(payload)
        results = document.get('chart',{}).get('result')
        if not results:
            raise ValueError(f'{symbol}: Yahoo returned no historical chart')
        frame, metadata = normalize_yahoo_stock(results[0],symbol,start,end)
        metadata['raw_sources'] = [source]
        normalized[symbol], instruments[symbol] = frame, metadata
    for pair in fx_pairs:
        sides, sources = {}, []
        divisor = 1000 if pair.endswith('/JPY') else 100000
        for side in ('BID','ASK'):
            chunks = []
            for year in years:
                payload, info = responses[('fx',pair,year,side)]
                chunks.append(decode_duka_daily(payload,year,divisor))
                sources.append(info)
            sides[side] = pd.concat(chunks,ignore_index=True)
        frame, metadata = normalize_duka_fx(sides['BID'],sides['ASK'],pair,start,end)
        metadata['raw_sources'] = sources
        normalized[pair], instruments[pair] = frame, metadata
    for symbol, frame in normalized.items():
        path = directory/(symbol.replace('/','_')+'.parquet')
        frame.to_parquet(path,index=False)
        elapsed = (frame['timestamp'].iloc[-1]-frame['timestamp'].iloc[0]).total_seconds()/86400+1
        instruments[symbol].update({'file':path.name,'rows':len(frame),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'start':frame['timestamp'].iloc[0].isoformat(),'end':frame['available_at'].iloc[-1].isoformat(),
            'periods_per_year':len(frame)/(elapsed/365.2425)})
    manifest = {'schema_version':1,'data_kind':'real historical market data','requested_start':start,'requested_end_exclusive':end,
                'universe':list(normalized),'universe_selection':'user-selected fixed instruments; no survivorship-free universe claim',
                'instruments':instruments}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def load_dataset(directory: str | Path) -> tuple[dict[str,pd.DataFrame], dict]:
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    markets = {}
    for symbol, info in manifest['instruments'].items():
        path = directory/info['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Dataset integrity mismatch: {symbol}')
        markets[symbol] = validate_bars(pd.read_parquet(path))
    return markets, manifest


def merge_fx_weekend_stubs(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """EXTENSION (audit item 13): fold each Sunday FX stub into the next session.

    Dukascopy's Sunday daily candle covers only the few hours after the weekly
    open, yet indicators would count it as a full period. The stub is merged
    causally into the following bar (within two days): open quotes from the
    stub, close quotes and availability from the following bar, high/low as
    extremes and feed volumes summed. The merged bar keeps the stub's opening
    timestamp, so no information becomes available earlier than before.
    """
    data = validate_bars(frame).copy()
    stamps = pd.DatetimeIndex(data['timestamp'])
    drop = []
    merged = 0
    for i in np.flatnonzero(stamps.day_of_week == 6):
        nxt = i + 1
        if nxt >= len(data) or stamps[nxt] - stamps[i] > pd.Timedelta(days=2):
            continue
        for prefix in ('', 'bid_', 'ask_'):
            high, low = f'{prefix}high', f'{prefix}low'
            if high in data:
                data.loc[nxt, high] = max(data.loc[i, high], data.loc[nxt, high])
                data.loc[nxt, low] = min(data.loc[i, low], data.loc[nxt, low])
            opening = f'{prefix}open'
            if opening in data:
                data.loc[nxt, opening] = data.loc[i, opening]
        for name in ('bid_feed_volume', 'ask_feed_volume'):
            if name in data:
                data.loc[nxt, name] = data.loc[i, name] + data.loc[nxt, name]
        data.loc[nxt, 'timestamp'] = data.loc[i, 'timestamp']
        drop.append(i)
        merged += 1
    data = data.drop(index=drop).reset_index(drop=True)
    return validate_bars(data), merged
