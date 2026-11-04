import numpy as np
import pandas as pd
import pytest
from gen_trade.benchmark import calendar_folds,stitch


def test_calendar_folds_match_years_for_stock_and_sunday_fx_observations():
    for frequency in ('B','D'):
        dates=pd.date_range('2015-01-01','2025-12-31',freq=frequency,tz='UTC')
        bars=pd.DataFrame({'timestamp':dates,'session_date':dates.strftime('%Y-%m-%d')})
        folds=calendar_folds(bars)
        assert len(folds)==7
        for year,fold in zip(range(2019,2026),folds):
            assert bars['session_date'].iloc[fold.test_start].startswith(str(year))
            assert bars['session_date'].iloc[fold.test_stop-1].startswith(str(year))
            assert bars['session_date'].iloc[fold.train_start].startswith(str(year-4))
            assert fold.validation_start-fold.train_stop==5
            assert fold.test_start-fold.validation_stop==5
        assert all(a.test_stop==b.test_start for a,b in zip(folds,folds[1:]))


def test_stitch_compounds_frozen_nonoverlapping_fold_paths_without_selecting_seeds():
    folds=[]
    for index,values in enumerate(([100,110,120],[100,90,80])):
        path={'initial_cash':100,'equity':values,'timestamps':[f'202{index}-01-01T00:00:00+00:00',f'202{index}-01-02T00:00:00+00:00'],
              'trades':[],'ledger':None}
        folds.append({'base_seed':11,'bounds':{'test_start':index*2},'methods':{'moo3':{'backtest':path}}})
    result=stitch({'folds':folds},'moo3',11)
    np.testing.assert_allclose(result['equity'],[1,1.1,1.2,1.08,.96])
    assert len(result['timestamps'])==4
    broken={'folds':[folds[0],folds[0]]}
    with pytest.raises(ValueError,match='duplicated'):
        stitch(broken,'moo3',11)


def test_no_validation_trades_deploys_cash_and_preserves_trials_without_refitting(monkeypatch):
    from gen_trade import experiments
    from gen_trade.backtest import ExecutionConfig
    from gen_trade.experiments import ExperimentJob,run_experiment_or_cash
    from gen_trade.runner import synthetic_markets
    fits=[]
    class Inactive:
        def __init__(self,**kwargs):pass
        def fit(self,bars,extra_features=None):fits.append(len(bars));return self
        def predict(self,bars,extra_features=None):return np.zeros(len(bars),dtype=np.int8)
        def parameters(self):return {}
        @property
        def diagnostics(self):return {'fitted':True}
    monkeypatch.setattr(experiments,'method_class',lambda name:Inactive)
    bars=synthetic_markets(120)['BTC/USDT']
    job=ExperimentJob('moo3',bars,{},60,65,85,90,4,0,11,ExecutionConfig(),[{},{}])
    report,result=run_experiment_or_cash(job)
    assert fits==[60,60]
    assert report['status']=='cash_fallback_no_eligible_configuration'
    assert report['selection']['selected_candidate'] is None
    assert len(report['selection']['trials'])==2
    assert report['selection']['test_used_for_selection'] is False
    np.testing.assert_array_equal(result.equity,np.full(31,10000.))
