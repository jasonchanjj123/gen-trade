from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from gen_trade.backtest import ExecutionConfig,run_backtest
from gen_trade.instruments import InstrumentSpec
from gen_trade.methods.moo3 import MOO3,Node,evaluate_tree
from gen_trade.methods.indicators import raw_features
from gen_trade.methods.mstgam import MSTGAM,paper_sharpe
from gen_trade.methods.igtsp_ring import IGTSPRing,portfolio_fitness
from gen_trade.runner import synthetic_markets


def stock_frame():
    f=synthetic_markets(160,seed=7)['BTC/USDT'].copy()
    f['symbol']='TEST'
    for name in ('open','high','low','close'):
        f[name]/=300
        f['execution_'+name]=f[name]
        f.loc[80:,'execution_'+name]/=2
    f['split_ratio']=1.;f.loc[80,'split_ratio']=2.
    f['dividend']=0.;f.loc[95,'dividend']=.3
    return f


def node(value):
    if 'constant' in value:return Node('constant',value=value['constant'])
    if 'feature' in value:return Node('feature',value=value['feature'])
    return Node(value['operator'],tuple(node(child) for child in value['children']))


def test_market_moo3_front_objectives_match_scalar_costed_trade_ledgers():
    f=stock_frame()
    config=ExecutionConfig(initial_cash=100000,fee=0,max_holding_days=5,take_profit=.05,
            instrument=InstrumentSpec('TEST','stock',quantity_step=1,minimum_commission=1,commission_per_unit=.005))
    model=MOO3(population_size=8,generations=1,seed=11,execution=config).fit(f)
    features=model._scaler.transform(raw_features(f,model.threshold))
    for candidate in model.diagnostics['pareto_front']:
        signals=evaluate_tree(node(candidate['rule']),features,len(f)).astype(np.int8)
        result=run_backtest(f,signals,config)
        assert candidate['paper_total_return_pct']==pytest.approx(result.paper_total_return)
        assert candidate['mean_trade_return']==pytest.approx(np.mean(result.trade_returns))
        assert candidate['trade_return_std']==pytest.approx(np.std(result.trade_returns))


def test_market_mstgam_winner_fitness_matches_scalar_costed_ledger():
    f=stock_frame()
    config=ExecutionConfig(initial_cash=100000,fee=0,
            instrument=InstrumentSpec('TEST','stock',quantity_step=1,minimum_commission=1,commission_per_unit=.005))
    model=MSTGAM(population_size=8,generations=1,seed=11,execution=config).fit(f)
    result=run_backtest(f,model.predict(f),config)
    assert model.diagnostics['best_fitness']==pytest.approx(paper_sharpe(result.trade_returns))


def test_nonlinear_ring_fitness_uses_actual_sleeve_capital_not_full_account_per_strategy():
    f=stock_frame()
    config=ExecutionConfig(initial_cash=100000,fee=0,
            instrument=InstrumentSpec('TEST','stock',quantity_step=1,minimum_commission=1,commission_per_unit=.005))
    model=IGTSPRing(population_size=8,generations=1,island_count=4,seed=11,execution=config).fit(f)
    plan=model.predict(f)
    months=model.diagnostics['training_months']
    paths=[];counts=[]
    for i,weight in enumerate(plan.weights):
        means=np.zeros(len(months));n=np.zeros(len(months),dtype=int)
        if weight:
            result=run_backtest(f,plan.signals[:,i],replace(config,initial_cash=config.initial_cash*weight,
                               take_profit=plan.take_profit,stop_loss=plan.stop_loss))
            for trade in result.trades:
                index=months.index(trade.exit_time.strftime('%Y-%m'))
                means[index]+=trade.return_fraction;n[index]+=1
        means=np.divide(means,n,out=np.zeros_like(means),where=n>0)
        paths.append(np.r_[1,np.cumprod(1+means)]);counts.append(n.sum())
    expected=portfolio_fitness(model._selected,np.array(paths),np.array(counts))
    actual=model.diagnostics['selected_training']
    for key in ('fitness','paper_portfolio_return','paper_portfolio_risk','group_balance'):
        assert actual[key]==pytest.approx(expected[key])
