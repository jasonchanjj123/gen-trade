from datetime import datetime,timezone
import copy
import lzma
import struct
import numpy as np
import pandas as pd
import pytest
from gen_trade.market_data import normalize_yahoo_stock,decode_duka_daily,normalize_duka_fx


def test_yahoo_reconstruction_has_causal_split_neutral_signals_and_separate_dividends():
    times=pd.to_datetime(['2020-01-02 14:30','2020-01-03 14:30','2020-01-06 14:30'],utc=True)
    q={'open':[50,51,52],'high':[51,52,53],'low':[49,50,51],'close':[50,51,52],'volume':[200,200,200]}
    document={'timestamp':[int(t.timestamp()) for t in times],'indicators':{'quote':[q]},'meta':{'currency':'USD'},
        'events':{'splits':{'a':{'date':int(times[-1].timestamp()),'numerator':2,'denominator':1}},
                  'dividends':{'a':{'date':int(times[1].timestamp()),'amount':.5}}}}
    frame,info=normalize_yahoo_stock(document,'TEST','2020-01-02','2020-01-07')
    np.testing.assert_allclose(frame['execution_open'],[100,102,52])
    np.testing.assert_allclose(frame['open'],[100,102,104])
    np.testing.assert_allclose(frame['volume'],[100,100,100])
    np.testing.assert_allclose(frame['dividend'],[0,1,0])
    assert frame['timestamp'].iloc[0]==times[0]
    assert frame['available_at'].iloc[0]==pd.Timestamp('2020-01-02 21:00',tz='UTC')
    # A later provider split-adjustment revision must reconstruct the same earlier market facts.
    revised=copy.deepcopy(document)
    for name in ('open','high','low','close'):
        revised['indicators']['quote'][0][name]=[v/2 for v in q[name]]
    revised['indicators']['quote'][0]['volume']=[v*2 for v in q['volume']]
    revised['events']['dividends']['a']['amount']/=2
    revised['events']['splits']['b']={'date':int(pd.Timestamp('2020-01-08',tz='UTC').timestamp()),'numerator':2,'denominator':1}
    after,_=normalize_yahoo_stock(revised,'TEST','2020-01-02','2020-01-07')
    pd.testing.assert_frame_equal(frame,after)


def test_yahoo_rejects_missing_sessions_instead_of_filling_prices():
    times=pd.to_datetime(['2020-01-02 14:30','2020-01-06 14:30'],utc=True)
    document={'timestamp':[int(t.timestamp()) for t in times],
      'indicators':{'quote':[{name:[100,100] for name in ('open','high','low','close','volume')}]},'meta':{}}
    with pytest.raises(ValueError,match='session coverage'):
        normalize_yahoo_stock(document,'TEST','2020-01-02','2020-01-07')


def test_native_duka_daily_record_order_scaling_and_alignment():
    raw=struct.pack('>5If',0,110000,112000,109000,113000,123.)
    decoded=decode_duka_daily(lzma.compress(raw),2020,100000)
    assert decoded['open'].iloc[0]==1.1 and decoded['close'].iloc[0]==1.12
    assert decoded['low'].iloc[0]==1.09 and decoded['high'].iloc[0]==1.13
    ask=decoded.copy()
    for name in ('open','high','low','close'):ask[name]+=.0002
    frame,info=normalize_duka_fx(decoded,ask,'EUR/USD','2020-01-01','2020-01-02')
    assert np.isnan(frame['volume']).all()
    assert frame['open'].iloc[0]==frame['bid_open'].iloc[0]
    assert frame['ask_close'].iloc[0]>frame['bid_close'].iloc[0]
    bad=ask.copy();bad['timestamp']+=pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='aligned'):
        normalize_duka_fx(decoded,bad,'EUR/USD','2020-01-01','2020-01-02')
