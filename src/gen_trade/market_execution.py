"""Batched account-currency stock/FX execution with an auditable scalar ledger.

Every optimizer and the held-out backtest use the same prepared array engine.
Daily candles cannot resolve intrabar stop order: exits observed at close fill
next open. Corporate actions are applied before new orders on the effective date.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import validate_bars


@dataclass(frozen=True)
class MarketArrays:
    bid_open: np.ndarray
    ask_open: np.ndarray
    bid_close: np.ndarray
    ask_close: np.ndarray
    conversion_bid_open: np.ndarray
    conversion_ask_open: np.ndarray
    conversion_bid_close: np.ndarray
    conversion_ask_close: np.ndarray
    split: np.ndarray
    dividend: np.ndarray
    times: tuple
    known: tuple
    time_ns: np.ndarray
    known_ns: np.ndarray
    hours: np.ndarray
    open_month: np.ndarray
    close_month: np.ndarray
    months: tuple
    rollover_long: np.ndarray | None
    rollover_short: np.ndarray | None

    @classmethod
    def from_bars(cls,bars,spec):
        frame = validate_bars(bars)
        def values(name,default=None):
            result = frame[name].to_numpy(dtype=float) if name in frame else np.full(len(frame),default,dtype=float)
            if not np.isfinite(result).all():
                raise ValueError(f'Execution column {name} must be finite')
            return result
        if spec.asset_class=='fx':
            if any(name not in frame for name in ('bid_open','ask_open','bid_close','ask_close')):
                raise ValueError('FX execution requires observed bid and ask opens/closes')
            bo,ao,bc,ac = (values(name) for name in ('bid_open','ask_open','bid_close','ask_close'))
        else:
            raw_open = values('execution_open') if 'execution_open' in frame else values('open')
            raw_close = values('execution_close') if 'execution_close' in frame else values('close')
            if spec.minimum_spread == 0:
                half = spec.spread_bps/20000
                bo,ao,bc,ac = raw_open*(1-half),raw_open*(1+half),raw_close*(1-half),raw_close*(1+half)
            else:
                # Full spread = max(proportional spread, absolute tick floor), split evenly.
                half_open = np.maximum(raw_open*spec.spread_bps/20000, spec.minimum_spread/2)
                half_close = np.maximum(raw_close*spec.spread_bps/20000, spec.minimum_spread/2)
                bo,ao,bc,ac = raw_open-half_open,raw_open+half_open,raw_close-half_close,raw_close+half_close
        if any((v<=0).any() for v in (bo,ao,bc,ac)) or (ao<bo).any() or (ac<bc).any():
            raise ValueError('Execution quotes must be positive, aligned and uncrossed')
        if spec.quote_currency==spec.account_currency:
            cbo,cao,cbc,cac = (np.ones(len(frame)) for _ in range(4))
        elif spec.base_currency==spec.account_currency and spec.asset_class=='fx':
            cbo,cao,cbc,cac = 1/ao,1/bo,1/ac,1/bc
        else:
            names = ('quote_to_account_bid_open','quote_to_account_ask_open',
                     'quote_to_account_bid_close','quote_to_account_ask_close')
            if any(name not in frame for name in names):
                raise ValueError('Cross-currency execution requires explicit contemporaneous bid/ask conversion columns')
            cbo,cao,cbc,cac = (values(name) for name in names)
            if any((v<=0).any() for v in (cbo,cao,cbc,cac)) or (cao<cbo).any() or (cac<cbc).any():
                raise ValueError('Currency conversion quotes must be positive and uncrossed')
        times,known = pd.DatetimeIndex(frame['timestamp']),pd.DatetimeIndex(frame['available_at'])
        first = times[0].year*12+times[0].month-1
        last = known[-1].year*12+known[-1].month-1
        months = tuple(f'{i//12:04d}-{i%12+1:02d}' for i in range(first,last+1))
        split,dividend = values('split_ratio',1.),values('dividend',0.)
        if (split<=0).any() or (dividend<0).any() or (spec.asset_class=='fx' and (split!=1).any()):
            raise ValueError('Invalid corporate actions')
        roll_names=('rollover_long_per_base_unit_quote','rollover_short_per_base_unit_quote')
        if any(name in frame for name in roll_names) and not all(name in frame for name in roll_names):
            raise ValueError('Explicit rollover requires both long and short quote-currency amounts')
        rolls = tuple(values(name) for name in roll_names) if all(name in frame for name in roll_names) else (None,None)
        return cls(bo,ao,bc,ac,cbo,cao,cbc,cac,split,dividend,tuple(times),tuple(known),
                   times.as_unit('ns').asi8,known.as_unit('ns').asi8,times.hour.to_numpy(),
                   np.asarray(times.year*12+times.month-1-first),np.asarray(known.year*12+known.month-1-first),months,*rolls)


@dataclass
class MarketBatch:
    equity: np.ndarray | None
    terminal_equity: np.ndarray
    trade_counts: np.ndarray
    trade_return_sum: np.ndarray
    trade_return_std: np.ndarray
    paper_total_return: np.ndarray
    monthly_returns: np.ndarray
    monthly_counts: np.ndarray
    trades: list
    fees: np.ndarray
    financing: np.ndarray
    dividends: np.ndarray
    months: tuple
    trade_log_growth: np.ndarray | None = None

    @property
    def compounded_trade_return(self) -> np.ndarray:
        """Scale-free sum of log(1+q) mapped back to a compounded fraction."""
        return np.expm1(self.trade_log_growth) if self.trade_log_growth is not None else None


def execute_batch(market,signals,config,*,initial_cash=None,take_profit=None,stop_loss=None,record=False):
    from .backtest import Trade
    spec = config.instrument
    signals=np.asarray(signals)
    if signals.ndim==1:
        signals=signals[:,None]
    if signals.ndim!=2 or signals.shape[0]!=len(market.times) or not np.isin(signals,[-1,0,1]).all():
        raise ValueError('Execution needs an aligned n-by-strategy decision matrix')
    m=signals.shape[1]
    cash=np.full(m,config.initial_cash,dtype=float) if initial_cash is None else np.broadcast_to(initial_cash,(m,)).astype(float).copy()
    initial=cash.copy()
    if not np.isfinite(cash).all() or (cash<0).any():
        raise ValueError('Initial sleeve capital must be finite and nonnegative')
    tp=np.full(m,np.nan if config.take_profit is None else config.take_profit) if take_profit is None else np.broadcast_to(take_profit,(m,))
    sl=np.full(m,np.nan if config.stop_loss is None else config.stop_loss) if stop_loss is None else np.broadcast_to(stop_loss,(m,))
    quantity=np.zeros(m); entry_fill=np.zeros(m); entry_notional=np.zeros(m); entry_capital=np.ones(m)
    original_quantity=np.ones(m); original_fill=np.zeros(m); principal_flow=np.zeros(m)
    entry_index=np.zeros(m,dtype=int); entry_time=np.zeros(m,dtype=np.int64)
    position_fees=np.zeros(m); position_carry=np.zeros(m); position_income=np.zeros(m)
    pending=np.zeros(m,dtype=np.int8)
    total_fees=np.zeros(m); total_carry=np.zeros(m); total_dividends=np.zeros(m)
    counts=np.zeros(m,dtype=int); return_sum=np.zeros(m); means=np.zeros(m); m2=np.zeros(m); log_growth=np.zeros(m)
    unit_profit=np.zeros(m); first_unit_capital=np.zeros(m)
    monthly_sum=np.zeros((len(market.months),m)); monthly_counts=np.zeros_like(monthly_sum,dtype=int)
    equity_path=np.empty((len(market.times)+1,m)) if record else None
    if record: equity_path[0]=cash
    ledger=[]
    slip=config.slippage_bps/10000
    fee_rate=config.fee+spec.commission_bps/10000
    holding_ns=pd.Timedelta(days=config.max_holding_days).value if config.max_holding_days is not None else None
    year_ns=365.25*86400*1e9

    def convert(amount,bid,ask):
        return amount*np.where(amount>=0,bid,ask)

    def notional(price,units,cb,ca):
        return np.abs(units)*price*((cb+ca)/2)

    def commission(units,value):
        return np.where(units!=0,np.maximum(spec.minimum_commission,spec.commission_per_unit*np.abs(units))+fee_rate*value,0.)

    def equity(bid,ask,cb,ca):
        mark=np.where(quantity>=0,bid,ask)
        if spec.asset_class=='stock':
            return cash+convert(quantity*mark,cb,ca)
        return cash+convert(quantity*(mark-entry_fill),cb,ca)

    def accrue(duration,bid,ask,cb,ca):
        nonlocal cash
        if duration<=0 or not np.any(quantity): return
        held=quantity!=0
        value=notional((bid+ask)/2,quantity,cb,ca)
        rate=np.where(quantity>0,spec.financing_long,spec.financing_short)
        charge=value*rate*duration/year_ns
        if spec.asset_class=='fx' and market.rollover_long is not None:
            charge[:]=0
        charge+=np.where(held,np.maximum(-cash,0)*spec.debit_interest_rate*duration/year_ns,0)
        cash-=charge; position_carry[:]+=charge; total_carry[:]+=charge

    def close(mask,bid,ask,cb,ca,index,month,time,reason):
        nonlocal cash,quantity
        ids=np.flatnonzero(mask)
        if not ids.size:return
        fill=np.where(quantity>0,bid*(1-slip),ask*(1+slip))
        value=notional(fill,quantity,cb,ca)
        exit_fee=commission(quantity,value)
        if spec.asset_class=='stock':
            pnl=principal_flow+convert(quantity*fill,cb,ca)+position_income-position_carry-position_fees-exit_fee
            cash[ids]+=convert(quantity*fill,cb,ca)[ids]-exit_fee[ids]
        else:
            gross=convert(quantity*(fill-entry_fill),cb,ca)
            pnl=gross+position_income-position_carry-position_fees-exit_fee
            cash[ids]+=gross[ids]-exit_fee[ids]
        total_fees[ids]+=exit_fee[ids]
        q=pnl[ids]/entry_capital[ids]
        old=counts[ids].copy(); counts[ids]+=1
        delta=q-means[ids]; means[ids]+=delta/counts[ids]; m2[ids]+=delta*(q-means[ids])
        return_sum[ids]+=q
        log_growth[ids]+=np.log1p(np.maximum(q,-1+1e-12))
        monthly_sum[month,ids]+=q; monthly_counts[month,ids]+=1
        unit_profit[ids]+=pnl[ids]/original_quantity[ids]
        first=ids[old==0]; first_unit_capital[first]=entry_capital[first]/original_quantity[first]
        if record:
            for j,k in enumerate(ids):
                why=reason if isinstance(reason,str) else str(reason[k])
                ledger.append((int(k),Trade(int(entry_index[k]),index,market.times[int(entry_index[k])],time,
                    float(original_fill[k]),float(fill[k]),float(q[j]),why=='sample_end',why,
                    quantity=float(original_quantity[k]),exit_quantity=float(abs(quantity[k])),cash_pnl=float(pnl[k]),
                    fees=float(position_fees[k]+exit_fee[k]),financing=float(position_carry[k]),
                    dividend_income=float(position_income[k]),account_currency=spec.account_currency,
                    unit_net_profit=float(pnl[k]/original_quantity[k]),
                    unit_entry_cost=float(entry_capital[k]/original_quantity[k]))))
        quantity[ids]=0; pending[ids]=0

    last_equity=cash.copy()
    for i in range(len(market.times)):
        if i:
            accrue(market.time_ns[i]-market.known_ns[i-1],market.bid_close[i-1],market.ask_close[i-1],
                   market.conversion_bid_close[i-1],market.conversion_ask_close[i-1])
        held=quantity!=0
        if spec.asset_class=='stock':
            quantity*=market.split[i]
            entry_fill=np.divide(entry_fill,market.split[i])
            income=convert(quantity*market.dividend[i],market.conversion_bid_open[i],market.conversion_ask_open[i])
            cash+=income; position_income+=income; total_dividends+=income
        signal=signals[i-1] if i and market.known_ns[i-1]<=market.time_ns[i] else np.zeros(m,dtype=np.int8)
        deadline=(quantity!=0)&(market.time_ns[i]>=entry_time+holding_ns) if holding_ns is not None else np.zeros(m,dtype=bool)
        exits=(quantity!=0)&((pending!=0)|deadline|((quantity>0)&(signal==-1))|((quantity<0)&(signal==1)))
        reasons=np.where(deadline,'holding_deadline',np.where(pending==3,'margin_call',np.where(pending==2,'stop_loss',
                np.where(pending==1,'profit_target','signal'))))
        close(exits,market.bid_open[i],market.ask_open[i],market.conversion_bid_open[i],market.conversion_ask_open[i],
              i,int(market.open_month[i]),market.times[i],reasons)
        allowed=config.entry_hours is None or market.hours[i] in config.entry_hours
        if allowed:
            direction=np.where(signal==1,1,np.where((signal==-1)&spec.allow_short,-1,0))
            enter=(quantity==0)&~exits&(direction!=0)&(cash>0)
            fill=np.where(direction>0,market.ask_open[i]*(1+slip),market.bid_open[i]*(1-slip))
            unit_value=fill*((market.conversion_bid_open[i]+market.conversion_ask_open[i])/2)
            if spec.asset_class=='stock':
                unit_value=fill*np.where(direction>0,market.conversion_ask_open[i],market.conversion_bid_open[i])
            denominator=unit_value/spec.leverage+fee_rate*unit_value
            maximum=np.minimum(np.maximum(cash-spec.minimum_commission,0)/denominator,
                               cash/(denominator+spec.commission_per_unit))
            if spec.quantity_step:
                maximum=np.floor(maximum/spec.quantity_step+1e-12)*spec.quantity_step
            new=direction*np.maximum(maximum,0)
            fee=commission(new,notional(fill,new,market.conversion_bid_open[i],market.conversion_ask_open[i]))
            ids=np.flatnonzero(enter&(new!=0))
            quantity[ids]=new[ids]; entry_fill[ids]=fill[ids]; original_quantity[ids]=np.abs(new[ids])
            original_fill[ids]=fill[ids]
            entry_notional[ids]=(np.abs(new)*unit_value)[ids]
            entry_capital[ids]=entry_notional[ids]/spec.leverage+fee[ids]
            position_fees[ids]=fee[ids]; position_carry[ids]=0; position_income[ids]=0
            entry_index[ids]=i; entry_time[ids]=market.time_ns[i]
            if spec.asset_class=='stock':
                flow=convert(-new*fill,market.conversion_bid_open[i],market.conversion_ask_open[i])
                cash[ids]+=flow[ids]; principal_flow[ids]=flow[ids]
            cash[ids]-=fee[ids]; total_fees[ids]+=fee[ids]
        accrue(market.known_ns[i]-market.time_ns[i],market.bid_open[i],market.ask_open[i],
               market.conversion_bid_open[i],market.conversion_ask_open[i])
        if spec.asset_class=='fx' and market.rollover_long is not None:
            payment=np.abs(quantity)*np.where(quantity>=0,market.rollover_long[i],market.rollover_short[i])
            credit=convert(payment,market.conversion_bid_close[i],market.conversion_ask_close[i])
            cash+=credit; position_carry-=credit; total_carry-=credit
        close_quote=np.where(quantity>=0,market.bid_close[i],market.ask_close[i])
        movement=np.divide(close_quote-entry_fill,entry_fill,out=np.zeros(m),where=entry_fill>0)*np.sign(quantity)
        pending[:]=0
        pending[(quantity!=0)&np.isfinite(tp)&(movement>=tp)]=1
        pending[(quantity!=0)&np.isfinite(sl)&(movement<=-sl)]=2
        last_equity=equity(market.bid_close[i],market.ask_close[i],market.conversion_bid_close[i],market.conversion_ask_close[i])
        margin_applicable=(spec.asset_class=='fx') | (spec.leverage>1) | (quantity<0)
        pending[(quantity!=0)&margin_applicable&(last_equity<=entry_notional/spec.leverage*spec.maintenance_margin_fraction)]=3
        if record:equity_path[i+1]=last_equity
    close(quantity!=0,market.bid_close[-1],market.ask_close[-1],market.conversion_bid_close[-1],market.conversion_ask_close[-1],
          len(market.times)-1,int(market.close_month[-1]),market.known[-1],'sample_end')
    last_equity=cash.copy()
    if record:equity_path[-1]=cash
    monthly=np.divide(monthly_sum,monthly_counts,out=np.zeros_like(monthly_sum),where=monthly_counts>0)
    std=np.sqrt(np.maximum(np.divide(m2,counts,out=np.zeros_like(m2),where=counts>0),0))
    paper=np.divide(100*unit_profit,first_unit_capital,out=np.zeros(m),where=first_unit_capital>0)
    return MarketBatch(equity_path,last_equity,counts,return_sum,std,paper,monthly.T,monthly_counts.T,ledger,
                       total_fees,total_carry,total_dividends,market.months,log_growth)


def run_market_backtest(bars,signals,config):
    from .backtest import BacktestResult
    market=MarketArrays.from_bars(bars,config.instrument)
    outcome=execute_batch(market,np.asarray(signals),config,record=True)
    trades=tuple(t for column,t in outcome.trades if column==0)
    return BacktestResult(outcome.equity[:,0],np.array([t.return_fraction for t in trades]),trades,config,market.known,
        ledger={'account_currency':config.instrument.account_currency,'commissions':float(outcome.fees[0]),
                'financing':float(outcome.financing[0]),'dividends':float(outcome.dividends[0]),
                'execution_model':config.instrument.asset_class})
