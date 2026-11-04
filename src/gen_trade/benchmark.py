"""Calendar walk-forward, resumable six-instrument benchmark with full ledgers."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .backtest import BacktestResult,ExecutionConfig
from .data import ChromosomeContext
from .experiments import METHOD_NAMES,DEFAULT_CANDIDATES
from .instruments import benchmark_spec
from .market_data import load_dataset
from .runner import Fold,compare_methods
from .statistics import detailed_statistics,holm_adjust,paired_comparison


def calendar_folds(bars,*,first_test_year=2019,last_test_year=2025,train_years=3,validation_years=1,embargo_bars=5):
    if any(type(v) is not int for v in (first_test_year,last_test_year,train_years,validation_years,embargo_bars)):
        raise ValueError('Calendar fold settings must be integers')
    if train_years<1 or validation_years<1 or embargo_bars<0 or last_test_year<first_test_year:
        raise ValueError('Invalid calendar fold settings')
    dates=np.asarray(bars['session_date'] if 'session_date' in bars else bars['timestamp'].dt.strftime('%Y-%m-%d'))
    folds=[]
    for year in range(first_test_year,last_test_year+1):
        start=year-train_years-validation_years
        train_start=int(np.searchsorted(dates,f'{start}-01-01'))
        val_start=int(np.searchsorted(dates,f'{year-validation_years}-01-01'))
        test_start=int(np.searchsorted(dates,f'{year}-01-01'))
        test_stop=int(np.searchsorted(dates,f'{year+1}-01-01'))
        train_stop=val_start-embargo_bars
        val_stop=test_start-embargo_bars
        if not train_start<train_stop<=val_start<val_stop<=test_start<test_stop:
            raise ValueError(f'Insufficient data for calendar test year {year}')
        if dates[train_start][:4]!=str(start) or dates[test_stop-1][:4]!=str(year):
            raise ValueError(f'Incomplete requested calendar interval for {year}')
        folds.append(Fold(train_start,train_stop,test_start,test_stop,val_start,val_stop))
    return folds


def _code_hash():
    root=Path(__file__).resolve().parents[2]
    paths=sorted(list((root/'src/gen_trade').rglob('*.py'))+[root/'pyproject.toml',root/'uv.lock'])
    digest=hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def _path_statistics(equity,timestamps,trade_returns,*,periods_per_year,bootstrap_reps,seed):
    # Detailed statistics requires matching trade objects; only return fractions
    # are used by it, so retained ledger objects remain available in asset files.
    result=BacktestResult(np.asarray(equity),np.asarray(trade_returns),tuple(None for _ in trade_returns),
                          ExecutionConfig(initial_cash=1),tuple(pd.to_datetime(timestamps,utc=True)))
    return detailed_statistics(result,periods_per_year=periods_per_year,bootstrap_reps=bootstrap_reps,seed=seed),result


def stitch(asset_report,name,seed,*,stress=False,baseline=False):
    folds=[f for f in asset_report['folds'] if f['base_seed']==seed]
    folds.sort(key=lambda f:f['bounds']['test_start'])
    wealth=[1.];timestamps=[];trade_returns=[];cash_count=0
    expenses={'commissions':0.,'financing':0.,'dividends':0.}
    for fold in folds:
        if baseline:
            path=fold['baseline_cost_stress']['backtest'] if stress else fold['baseline_backtest']
        else:
            method=fold['methods'][name]
            path=method['cost_stress']['backtest'] if stress else method['backtest']
            cash_count+=method.get('status')=='cash_fallback_no_eligible_configuration'
        curve=np.asarray(path['equity'],dtype=float)/path['initial_cash']
        wealth.extend((wealth[-1]*curve[1:]).tolist())
        timestamps.extend(path['timestamps'])
        trade_returns.extend(trade['return_fraction'] for trade in path['trades'])
        if path.get('ledger'):
            for key in expenses:expenses[key]+=path['ledger'][key]
    if len(set(timestamps))!=len(timestamps) or timestamps!=sorted(timestamps):
        raise ValueError('Stitched held-out periods are duplicated or unordered')
    return {'equity':wealth,'timestamps':timestamps,'trade_returns':trade_returns,
            'cash_fallback_folds':cash_count,'sum_of_independently_capitalized_fold_ledger':expenses}


def summarize(reports,manifest,settings):
    records=[];pairs=[];paths={}
    for symbol,report in reports.items():
        annual=manifest['instruments'][symbol]['periods_per_year']
        for seed in settings['seeds']:
            results={}
            for name in (*METHOD_NAMES,'buy_and_hold'):
                base=stitch(report,name,seed,baseline=name=='buy_and_hold')
                stress=stitch(report,name,seed,stress=True,baseline=name=='buy_and_hold')
                stats,result=_path_statistics(base['equity'],base['timestamps'],base['trade_returns'],
                    periods_per_year=annual,bootstrap_reps=settings['bootstrap_reps'],seed=seed)
                stress_stats,_=_path_statistics(stress['equity'],stress['timestamps'],stress['trade_returns'],
                    periods_per_year=annual,bootstrap_reps=settings['bootstrap_reps'],seed=seed)
                records.append({'symbol':symbol,'asset_class':'fx' if '/' in symbol else 'stock',
                    'method':name,'seed':seed,'statistics':stats,'cost_stress_statistics':stress_stats,
                    'cash_fallback_folds':base['cash_fallback_folds'],
                    'fold_ledger_totals':base['sum_of_independently_capitalized_fold_ledger']})
                paths[(symbol,name,seed)]=base
                if name!='buy_and_hold':results[name]=result
            for i,first in enumerate(METHOD_NAMES):
                for second in METHOD_NAMES[i+1:]:
                    comparison=paired_comparison(results[first],results[second],periods_per_year=annual,
                        bootstrap_reps=settings['bootstrap_reps'],seed=seed)
                    pairs.append({'symbol':symbol,'seed':seed,'first':first,'second':second,**comparison})
    adjusted=holm_adjust(p['paired_return_difference_inference']['hac_pvalue_mean_zero'] for p in pairs)
    for pair,value in zip(pairs,adjusted):
        pair['holm_hac_pvalue_declared_suite_family']=value
        pair['suite_family_size']=len(pairs)
    return records,pairs,paths


def _render_report(directory,summary,paths):
    directory=Path(directory)
    def percent(value):return 'undefined' if value is None else f'{value*100:.2f}%'
    def number(value):return 'undefined' if value is None else f'{value:.2f}'
    lines=['# Stock / FX historical benchmark','',
        'Real daily market data, 2015–2025; rolling calendar-year tests from 2019 through 2025.',
        'All three methods and all declared seeds are retained. Figures are descriptive held-out results.',
        '',f"Search per fit: population {summary['settings']['population_size']}, generations {summary['settings']['generations']}; seeds {summary['settings']['seeds']}.",
        'Each fold fits two candidates, selects on validation, then refits through validation end. Test costs are applied to the same frozen strategies in both scenarios.',
        '', '| Instrument | Method | Median OOS return | Median annual return | Median Sharpe | Median max DD | Stress OOS return | Cash-fallback folds |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for symbol in summary['universe']:
        for name in (*METHOD_NAMES,'buy_and_hold'):
            rows=[r for r in summary['runs'] if r['symbol']==symbol and r['method']==name]
            def median(field,stress=False):
                values=[r['cost_stress_statistics' if stress else 'statistics'][field] for r in rows]
                values=[v for v in values if v is not None]
                return float(np.median(values)) if values else None
            lines.append(f"| {symbol} | {name} | {percent(median('net_total_return'))} | {percent(median('annualized_return'))} | {number(median('sharpe'))} | {percent(median('max_drawdown'))} | {percent(median('net_total_return',True))} | {sum(r['cash_fallback_folds'] for r in rows)} |")
    lines+=['','![Held-out equity](equity.png)','','## Data and execution',
        '', '- Stocks: contemporaneous execution prices, causal split-neutral indicator prices, split quantity/cost-basis changes, ex-date dividend entitlements, and NYSE sessions.',
        '- FX: historical Dukascopy bid/ask opens/closes; account currency USD with side-aware JPY conversion. Observed partial Sunday bars remain in the dataset.',
        '- All benchmark positions are long-only at 1× exposure, with continuous fractional sizing and zero idle-cash interest. The engine separately supports quantity steps, minimum commissions, shorting and leverage.',
        '- $100,000 starting capital per independently fitted fold. Normalized OOS paths compound each year; the benchmark cost model is scale invariant.',
        '- Stock assumed full spread 2 bps, extra slippage 1 bp per side, commission $0.005/share with no minimum.',
        '- FX observed spread plus 0.1 bp slippage and 0.35 bp commission per side; financing debit 2% annualized on held notional over actual calendar time. This is a declared scenario, not historical broker swap data.',
        '- Stress: double assumed commission, slippage, modeled stock spread and financing. Observed FX bid/ask data stay unchanged. No strategy is refitted on stress/test returns.',
        '- No eligible training/validation configuration deploys flat cash for that fold and retains the failure record. Such folds are never deleted or retuned on test outcomes.',
        '', '## Interpretation limits', '',
        'This fixed, user-selected six-instrument universe is not a survivorship-free stock universe. Market-impact, partial fills, taxes, actual broker swap histories and point-in-time vendor revisions are not modeled. Dividends are credited as ex-date entitlements rather than delayed payment-date cash.',
        'Stops/targets are observed at candle close and execute next open. Daily high/low paths are not invented. All code, data, cost assumptions and candidate settings are recorded in the asset JSON reports.',
        'HAC p-values and stationary-block intervals are descriptive. Holm adjustment covers the declared asset × seed × pair family; it does not correct genetic search, configuration research, or unreported trials.',
        'Per-asset JSON files retain every trial, frozen parameter set, equity observation, trade ledger and paper-specific metric. The summary JSON retains every seed, not a best-seed selection.']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(15,11),sharex=True,layout='constrained')
    colors={'moo3':'#2563eb','mstgam':'#059669','igtsp_ring':'#ea580c','buy_and_hold':'#6b7280'}
    for ax,symbol in zip(axes.flat,summary['universe']):
        for name in (*METHOD_NAMES,'buy_and_hold'):
            series=[paths[(symbol,name,seed)] for seed in summary['settings']['seeds']]
            values=np.array([s['equity'][1:] for s in series])
            dates=pd.to_datetime(series[0]['timestamps'],utc=True)
            ax.plot(dates,np.median(values,axis=0),label=name,color=colors[name],lw=1.5,ls='--' if name=='buy_and_hold' else '-')
            if name!='buy_and_hold':ax.fill_between(dates,values.min(axis=0),values.max(axis=0),color=colors[name],alpha=.1)
        ax.axhline(1,color='#aaa',lw=.7);ax.set_title(symbol,loc='left',fontweight='bold')
        ax.set_ylabel('Net wealth / initial wealth');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('2019–2025 held-out equity • median and range across declared seeds',fontsize=15)
    fig.savefig(directory/'equity.png',dpi=170)
    plt.close(fig)


def run_benchmark(data_directory,output_directory,*,population_size=64,generations=20,seeds=(11,22,33),workers=3,
                  first_test_year=2019,last_test_year=2025,embargo_bars=5,bootstrap_reps=500,resume=True,
                  symbols=None):
    markets,manifest=load_dataset(data_directory)
    symbols=list(symbols) if symbols is not None else list(markets)
    if not symbols or len(set(symbols))!=len(symbols) or set(symbols)-set(markets):
        raise ValueError('Benchmark symbols must be a unique nonempty subset of the downloaded universe')
    settings={'population_size':population_size,'generations':generations,'seeds':list(seeds),'workers':workers,
              'first_test_year':first_test_year,'last_test_year':last_test_year,'embargo_bars':embargo_bars,
              'bootstrap_reps':bootstrap_reps,'train_years':3,'validation_years':1,'initial_cash':100000,
              'candidate_configurations':DEFAULT_CANDIDATES,'on_no_eligible':'cash'}
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True)
    source_hash=_code_hash();reports={};start=perf_counter()
    for symbol in symbols:
        signature=hashlib.sha256(json.dumps({'settings':settings,'data':manifest['instruments'][symbol]['sha256'],
            'source':source_hash,'instrument':asdict(benchmark_spec(symbol))},sort_keys=True).encode()).hexdigest()
        path=directory/(symbol.replace('/','_')+'.json')
        if resume and path.exists():
            cached=json.loads(path.read_text())
            if cached.get('benchmark_signature')!=signature:
                raise ValueError(f'{symbol}: existing report belongs to different code/data/settings; use a new output directory or no-resume')
            reports[symbol]=cached
            print(f'Verified completed asset: {symbol}',flush=True)
            continue
        frame=markets[symbol]
        folds=calendar_folds(frame,first_test_year=first_test_year,last_test_year=last_test_year,embargo_bars=embargo_bars)
        print(f'Running {symbol}: {len(folds)} calendar folds × {len(seeds)} seeds × 3 methods',flush=True)
        report=compare_methods(markets,ChromosomeContext(symbol),train_bars=756,validation_bars=252,test_bars=252,
            embargo_bars=embargo_bars,max_folds=len(folds),population_size=population_size,generations=generations,
            seeds=tuple(seeds),fee=0,slippage_bps=1 if '/' not in symbol else .1,
            periods_per_year=manifest['instruments'][symbol]['periods_per_year'],bootstrap_reps=bootstrap_reps,
            workers=workers,instrument=benchmark_spec(symbol),stress_instrument=benchmark_spec(symbol,stress=True),
            initial_cash=100000,fold_plan=folds,include_paths=True,on_no_eligible='cash')
        report.update(benchmark_signature=signature,data_provenance=manifest['instruments'][symbol],
                      calendar_protocol={'training_years':3,'validation_years':1,'test_years':list(range(first_test_year,last_test_year+1))})
        temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,allow_nan=False)+'\n');temporary.replace(path)
        reports[symbol]=report
        (directory/'progress.json').write_text(json.dumps({'completed':list(reports),'requested':symbols,'elapsed_seconds':perf_counter()-start},indent=2)+'\n')
        print(f'Finished {symbol}: {len(report["folds"])*3} method runs saved',flush=True)
    records,pairs,paths=summarize(reports,manifest,settings)
    summary={'schema_version':1,'data_kind':'real historical market data','universe':symbols,'settings':settings,
             'source_sha256':source_hash,'elapsed_seconds':perf_counter()-start,'runs':records,'paired_comparisons':pairs,
             'method_runs':sum(len(report['folds'])*3 for report in reports.values()),
             'cash_fallback_method_folds':sum(row['cash_fallback_folds'] for row in records),
             'cost_scenarios':'primary and doubled assumed costs; same frozen models',
             'dataset_manifest':manifest}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    _render_report(directory,summary,paths)
    print(f'Completed benchmark: {summary["method_runs"]} method runs; report {directory/"report.md"}',flush=True)
    return summary
