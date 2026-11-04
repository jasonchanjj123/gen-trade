from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from gen_trade.backtest import ExecutionConfig,PortfolioSignals,run_backtest,run_portfolio_backtest
from gen_trade.instruments import InstrumentSpec
from gen_trade.market_execution import MarketArrays,execute_batch
from gen_trade.methods.ring_fitness import PreparedMarket,monthly_trade_profiles


def bars(prices,times=None):
    p=np.asarray(prices,dtype=float)
    t=pd.date_range('2020-01-01 09:00',periods=len(p),tz='UTC') if times is None else pd.to_datetime(times,utc=True)
    return pd.DataFrame({'timestamp':t,'available_at':t+pd.Timedelta(hours=7),'open':p,'high':p*1.01,
                         'low':p*.99,'close':p,'volume':100.,'split_ratio':1.,'dividend':0.})


def fx(bid,ask,pair='EUR/USD'):
    f=bars(bid)
    for name in ('open','high','low','close'):
        f['bid_'+name]=f[name]
        f['ask_'+name]=np.array(ask)*(1.01 if name=='high' else .99 if name=='low' else 1)
    f['volume']=np.nan
    return f


def test_stock_dividend_entitlement_split_quantity_and_no_double_adjustment():
    f=bars([100,100,51,55])
    f.loc[2,'split_ratio']=2.
    f.loc[2,'dividend']=1.
    config=ExecutionConfig(initial_cash=1000,fee=0,instrument=InstrumentSpec('TEST','stock',quantity_step=1))
    result=run_backtest(f,[1,0,-1,0],config)
    assert result.total_return==pytest.approx(.12)
    assert result.equity[-1]==1120
    trade=result.trades[0]
    assert trade.quantity==10 and trade.exit_quantity==20
    assert trade.entry_price==100 and trade.exit_price==55
    assert trade.dividend_income==20 and trade.cash_pnl==120
    assert trade.return_fraction==pytest.approx(.12)
    # Buying on the ex-date does not grant prior holders' dividend.
    late=run_backtest(f,[0,1,-1,0],config)
    assert late.ledger['dividends']==0


def test_whole_share_rounding_minimum_commission_and_true_sleeve_capital():
    f=bars([60,60,60])
    spec=InstrumentSpec('TEST','stock',quantity_step=1,minimum_commission=1)
    config=ExecutionConfig(initial_cash=100,fee=0,instrument=spec)
    result=run_backtest(f,[1,-1,0],config)
    assert result.trades[0].quantity==1
    assert result.equity[-1]==98
    assert result.ledger['commissions']==2
    plan=PortfolioSignals(('one','two'),np.array([[1,1],[-1,-1],[0,0]]),(.5,.5))
    split=run_portfolio_backtest(f,plan,config)
    assert not split.trades and split.equity[-1]==100  # Each $50 sleeve cannot buy a $60 share.


def test_unleveraged_cash_stock_does_not_invent_margin_calls_on_large_drawdown():
    f=bars([100,100,30,20])
    config=ExecutionConfig(initial_cash=1000,fee=0,instrument=InstrumentSpec('TEST','stock'))
    result=run_backtest(f,[1,0,0,0],config)
    assert len(result.trades)==1 and result.trades[0].reason=='sample_end'
    assert result.equity[-1]==200


def test_fx_bid_ask_spread_is_paid_once_each_side_without_fabricated_mid_fills():
    f=fx([1.,1.,1.1],[1.02,1.02,1.12])
    spec=InstrumentSpec('EUR/USD','fx',base_currency='EUR',quote_currency='USD')
    config=ExecutionConfig(initial_cash=1020,fee=0,instrument=spec)
    result=run_backtest(f,[1,-1,0],config)
    assert result.trades[0].quantity==pytest.approx(1000)
    assert result.trades[0].entry_price==1.02 and result.trades[0].exit_price==1.1
    assert result.equity[-1]==pytest.approx(1100)
    assert result.trades[0].cash_pnl==pytest.approx(80)


def test_usdjpy_quote_profit_and_loss_convert_to_usd_at_correct_side():
    f=fx([100,100,110],[102,102,112])
    spec=InstrumentSpec('USD/JPY','fx',base_currency='USD',quote_currency='JPY')
    config=ExecutionConfig(initial_cash=1020,fee=0,instrument=spec)
    result=run_backtest(f,[1,-1,0],config)
    quantity=result.trades[0].quantity
    expected=quantity*(110-102)/112
    assert result.trades[0].cash_pnl==pytest.approx(expected)
    assert result.equity[-1]==pytest.approx(1020+expected)
    loss=fx([100,100,90],[102,102,92])
    lost=run_backtest(loss,[1,-1,0],config)
    assert lost.trades[0].cash_pnl==pytest.approx(lost.trades[0].quantity*(90-102)/90)


def test_short_stock_pays_dividend_and_fx_short_closes_on_ask():
    f=bars([100,100,90,80]);f.loc[2,'dividend']=2.
    spec=InstrumentSpec('TEST','stock',allow_short=True)
    result=run_backtest(f,[-1,0,1,0],ExecutionConfig(initial_cash=1000,fee=0,instrument=spec))
    assert result.ledger['dividends']==-20
    assert result.trades[0].cash_pnl==180
    q=fx([1.,1.,.9],[1.02,1.02,.92])
    spec=InstrumentSpec('EUR/USD','fx',base_currency='EUR',allow_short=True)
    result=run_backtest(q,[-1,1,0],ExecutionConfig(initial_cash=1000,fee=0,instrument=spec))
    assert result.trades[0].entry_price==1 and result.trades[0].exit_price==.92
    assert result.trades[0].cash_pnl==pytest.approx(80)


def test_calendar_financing_includes_weekend_and_explicit_rollover_overrides_assumption():
    f=fx([1.,1.,1.],[1.,1.,1.])
    f['timestamp']=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06'],utc=True)
    f['available_at']=f['timestamp']+pd.Timedelta(hours=24)
    spec=InstrumentSpec('EUR/USD','fx',base_currency='EUR',financing_long=.036525)
    config=ExecutionConfig(initial_cash=1000,fee=0,instrument=spec)
    result=run_backtest(f,[1,-1,0],config)
    assert result.ledger['financing']==pytest.approx(.3)  # Three calendar days at $0.10/day.
    f['rollover_long_per_base_unit_quote']=.0002
    f['rollover_short_per_base_unit_quote']=-.0002
    credited=run_backtest(f,[1,-1,0],config)
    assert credited.ledger['financing']==pytest.approx(-.2)
    assert credited.equity[-1]==pytest.approx(1000.2)


def test_leverage_and_margin_exit_are_explicit_and_cannot_reenter_same_open():
    f=fx([1.,1.,.7,.65,.6],[1.,1.,.7,.65,.6])
    spec=InstrumentSpec('EUR/USD','fx',base_currency='EUR',leverage=2)
    result=run_backtest(f,[1,1,1,0,0],ExecutionConfig(initial_cash=1000,fee=0,instrument=spec))
    assert result.trades[0].quantity==2000
    assert result.trades[0].exit_index==3 and result.trades[0].reason=='margin_call'
    assert result.trades[0].cash_pnl==pytest.approx(-700)


def test_batched_moments_months_and_path_match_scalar_ledgers_with_split_dividend_and_stops():
    f=bars([100,101,99,51,53,50,48,52]);f.loc[3,'split_ratio']=2;f.loc[4,'dividend']=.3
    spec=InstrumentSpec('TEST','stock',quantity_step=1,minimum_commission=1,commission_per_unit=.005,allow_short=True)
    config=ExecutionConfig(initial_cash=10000,fee=.001,slippage_bps=2,take_profit=.03,stop_loss=.02,instrument=spec)
    signals=np.random.default_rng(4).choice([-1,0,1],size=(len(f),7))
    outcome=execute_batch(MarketArrays.from_bars(f,spec),signals,config,record=True)
    profile=monthly_trade_profiles(PreparedMarket.from_bars(f,spec),signals,config)
    for i in range(7):
        scalar=run_backtest(f,signals[:,i],config)
        np.testing.assert_allclose(outcome.equity[:,i],scalar.equity)
        assert outcome.trade_counts[i]==len(scalar.trades)
        assert outcome.trade_return_sum[i]==pytest.approx(scalar.trade_returns.sum())
        assert outcome.trade_return_std[i]==pytest.approx(np.std(scalar.trade_returns) if scalar.trades else 0)
        assert outcome.paper_total_return[i]==pytest.approx(scalar.paper_total_return)
        assert profile.trade_counts[i]==len(scalar.trades)


def test_cross_currency_requires_explicit_contemporaneous_conversion_inputs():
    f=fx([1,1,1],[1.01,1.01,1.01])
    spec=InstrumentSpec('EUR/GBP','fx',base_currency='EUR',quote_currency='GBP')
    with pytest.raises(ValueError,match='conversion'):
        run_backtest(f,[1,0,0],ExecutionConfig(instrument=spec))


def test_reverse_split_preserves_position_value_and_original_trade_units():
    f=bars([10,10,200,220]);f.loc[2,'split_ratio']=.05
    config=ExecutionConfig(initial_cash=1000,fee=0,instrument=InstrumentSpec('TEST','stock'))
    result=run_backtest(f,[1,0,-1,0],config)
    assert result.trades[0].quantity==100 and result.trades[0].exit_quantity==5
    assert result.equity[-1]==1100 and result.trades[0].cash_pnl==100
