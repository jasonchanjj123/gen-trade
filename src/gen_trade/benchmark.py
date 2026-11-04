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
from .experiments import METHOD_NAMES,CANDIDATE_PROFILES
from .instruments import benchmark_spec
from .market_data import load_dataset, merge_fx_weekend_stubs
from .runner import Fold,compare_methods
from .statistics import (deflated_sharpe_ratio,detailed_statistics,holm_adjust,paired_comparison,pbo_cscv,
                         probabilistic_sharpe_ratio,spa_test)

PROTOCOLS = {
    'original':{'compare':{}, 'fx_financing':.02, 'stock_minimum_spread':0., 'merge_fx_weekend_stubs':False},
    # EXTENSION (docs/EXTENSIONS.md): audit items 3, 6, 8 and 13.
    'audited':{'compare':{'selection':'pooled','deploy':'validated','min_validation_trades':2,
                          'validation_curves':True,'exposure_controls_enabled':True},
               'fx_financing':0., 'stock_minimum_spread':.01, 'merge_fx_weekend_stubs':True},
}


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
    native=[path for path in [*(root/'native'/'src').rglob('*.rs'),root/'native'/'Cargo.toml',root/'native'/'Cargo.lock']
            if path.exists()]
    paths=sorted(list((root/'src/gen_trade').rglob('*.py'))+[root/'pyproject.toml',root/'uv.lock']+native)
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


def stitch_ensemble(asset_report,name,seeds,*,stress=False):
    """EXTENSION: equal-capital mix of the seeds' frozen models in each fold.

    Each test year splits capital equally across the seeds' independently
    fitted models (no rebalancing inside the year), then compounds across
    years. Exact for the scale-invariant benchmark cost model.
    """
    members=[stitch(asset_report,name,seed,stress=stress) for seed in seeds]
    timestamps=members[0]['timestamps']
    if any(m['timestamps']!=timestamps for m in members):
        raise ValueError('Seed paths are not aligned')
    by_fold={}
    for fold in asset_report['folds']:
        if fold['base_seed'] in seeds:
            method=fold['methods'][name]
            path=method['cost_stress']['backtest'] if stress else method['backtest']
            by_fold.setdefault(fold['index'],[]).append(np.asarray(path['equity'],dtype=float)/path['initial_cash'])
    wealth=[1.]
    for index in sorted(by_fold):
        curve=np.mean(by_fold[index],axis=0)
        wealth.extend((wealth[-1]*curve[1:]).tolist())
    return {'equity':wealth,'timestamps':timestamps,'trade_returns':[r for m in members for r in m['trade_returns']],
            'cash_fallback_folds':sum(m['cash_fallback_folds'] for m in members),
            'sum_of_independently_capitalized_fold_ledger':None}


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
        annual=_periods_per_year(report,manifest,symbol)
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
    # EXTENSION: equal-capital seed ensembles per instrument and method.
    for symbol,report in reports.items():
        annual=_periods_per_year(report,manifest,symbol)
        for name in METHOD_NAMES:
            base=stitch_ensemble(report,name,settings['seeds'])
            stress=stitch_ensemble(report,name,settings['seeds'],stress=True)
            stats,_=_path_statistics(base['equity'],base['timestamps'],base['trade_returns'],
                periods_per_year=annual,bootstrap_reps=settings['bootstrap_reps'],seed=0)
            stress_stats,_=_path_statistics(stress['equity'],stress['timestamps'],stress['trade_returns'],
                periods_per_year=annual,bootstrap_reps=settings['bootstrap_reps'],seed=0)
            records.append({'symbol':symbol,'asset_class':'fx' if '/' in symbol else 'stock','method':name,
                'seed':'ensemble','statistics':stats,'cost_stress_statistics':stress_stats,
                'cash_fallback_folds':base['cash_fallback_folds'],'fold_ledger_totals':None})
            paths[(symbol,name,'ensemble')]=base
    return records,pairs,paths


def _returns(path):
    equity=np.asarray(path['equity'],dtype=float)
    return np.diff(equity)/equity[:-1]


def _periods_per_year(report,manifest,symbol):
    return report.get('periods_per_year_used',manifest['instruments'][symbol]['periods_per_year'])


def selection_bias(reports,manifest,settings,paths,*,blocks=8,spa_reps=1000):
    """EXTENSION: DSR, PBO (CSCV) and Hansen SPA over the reported held-out paths.

    Trials per instrument are the method x seed paths (all retained, none
    selected on test). DSR deflates each path's Sharpe by the expected maximum
    of that many unskilled trials; PBO asks whether ranking those paths on some
    years predicts their rank on the others; SPA tests whether any method beats
    buy-and-hold. None of these corrects the genetic search inside each fit.
    """
    output=[]
    for symbol,report in reports.items():
        keys=[(name,seed) for name in METHOD_NAMES for seed in settings['seeds']]
        matrix=np.column_stack([_returns(paths[(symbol,name,seed)]) for name,seed in keys])
        sharpe=np.array([np.mean(c)/np.std(c,ddof=1) if np.std(c,ddof=1)>0 else 0. for c in matrix.T])
        variance=float(np.var(sharpe,ddof=1)) if len(sharpe)>1 else 0.
        dsr={f'{name}:{seed}':{**deflated_sharpe_ratio(matrix[:,i],len(keys),variance),
                               'probabilistic_sharpe_ratio':probabilistic_sharpe_ratio(matrix[:,i]),
                               'sharpe_per_period':float(sharpe[i])}
             for i,(name,seed) in enumerate(keys)}
        try:
            pbo=pbo_cscv(matrix,blocks)
        except ValueError as error:
            pbo={'pbo':None,'note':str(error)}
        benchmark=_returns(paths[(symbol,'buy_and_hold',settings['seeds'][0])])
        spa={str(seed):spa_test(benchmark,np.column_stack([_returns(paths[(symbol,name,seed)]) for name in METHOD_NAMES]),
                                reps=spa_reps,seed=int(seed)) for seed in settings['seeds']}
        spa['ensemble']=spa_test(benchmark,np.column_stack([_returns(paths[(symbol,name,'ensemble')]) for name in METHOD_NAMES]),
                                 reps=spa_reps,seed=0)
        output.append({'symbol':symbol,'trials':len(keys),'trial_sharpe_variance_per_period':variance,
                       'deflated_sharpe':dsr,'pbo_cscv':pbo,'spa_vs_buy_and_hold':spa})
    return output


def exposure_summary(reports):
    """Median exposure controls per instrument and method across folds/seeds."""
    rows=[]
    for symbol,report in reports.items():
        for name in METHOD_NAMES:
            controls=[f['exposure_controls'][name] for f in report['folds'] if 'exposure_controls' in f]
            if not controls:continue
            def med(values):
                values=[v for v in values if v is not None]
                return float(np.median(values)) if values else None
            rows.append({'symbol':symbol,'method':name,'fold_seed_runs':len(controls),
                'median_exposure':med(c['mean_exposure'] for c in controls),
                'median_beta':med(c['alpha_beta_vs_buy_and_hold']['beta'] for c in controls),
                'median_alpha_annualized':med(c['alpha_beta_vs_buy_and_hold']['alpha_annualized'] for c in controls),
                'share_beating_exposure_matched_blend':float(np.mean([
                    (c['minus_exposure_matched_blend']['paired_return_difference_inference']['mean'] or 0)>0 for c in controls])),
                'median_random_timing_percentile':med(c['random_timing']['percentile'] if c['random_timing'] else None
                                                       for c in controls),
                'test_years_without_trades':int(sum(f['methods'][name]['statistics']['trade_count']==0
                                                    for f in report['folds']))})
    return rows


def _render_report(directory,summary,paths):
    directory=Path(directory)
    def percent(value):return 'undefined' if value is None else f'{value*100:.2f}%'
    def number(value):return 'undefined' if value is None else f'{value:.2f}'
    lines=['# Stock / FX historical benchmark','',
        'Real daily market data, 2015–2025; rolling calendar-year tests from 2019 through 2025.',
        'All three methods and all declared seeds are retained. Figures are descriptive held-out results.',
        '',f"Search per fit: population {summary['settings']['population_size']}, generations {summary['settings']['generations']}; seeds {summary['settings']['seeds']}.",
        ('Each fold fits two candidates, selects on validation, then refits through validation end.' if summary['settings'].get('protocol','original')=='original'
         else 'Each fold fits two candidates per seed, picks ONE configuration per method from seed-mean validation scores (>=2 validation trades), and deploys the validated training-only models.')
        +' Test costs are applied to the same frozen strategies in both scenarios.',
        '', '| Instrument | Method | Median OOS return | Median annual return | Median Sharpe | Median max DD | Stress OOS return | Cash-fallback folds |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for symbol in summary['universe']:
        for name in (*METHOD_NAMES,'buy_and_hold'):
            rows=[r for r in summary['runs'] if r['symbol']==symbol and r['method']==name and r['seed']!='ensemble']
            def median(field,stress=False):
                values=[r['cost_stress_statistics' if stress else 'statistics'][field] for r in rows]
                values=[v for v in values if v is not None]
                return float(np.median(values)) if values else None
            lines.append(f"| {symbol} | {name} | {percent(median('net_total_return'))} | {percent(median('annualized_return'))} | {number(median('sharpe'))} | {percent(median('max_drawdown'))} | {percent(median('net_total_return',True))} | {sum(r['cash_fallback_folds'] for r in rows)} |")
    settings=summary['settings']
    extended=settings.get('protocol','original')!='original' or settings.get('method_profile','paper')!='paper'
    if extended:
        lines+=['',f"Method profile: **{settings.get('method_profile','paper')}**; protocol: **{settings.get('protocol','original')}** (docs/EXTENSIONS.md).",
                '','## Equal-capital seed ensemble','',
                '| Instrument | Method | OOS return | Annual return | Sharpe | Max DD | Stress OOS return |',
                '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for row in [r for r in summary['runs'] if r['seed']=='ensemble']:
            st=row['statistics']
            lines.append(f"| {row['symbol']} | {row['method']} | {percent(st['net_total_return'])} | {percent(st['annualized_return'])} | {number(st['sharpe'])} | {percent(st['max_drawdown'])} | {percent(row['cost_stress_statistics']['net_total_return'])} |")
        if summary.get('exposure_controls'):
            lines+=['','## Exposure controls (medians over fold-seed test years)','',
                    'Beating the blend = share of test years whose mean daily return exceeds a static cash/buy-and-hold mix with the same mean exposure. Random-timing percentile = rank of the net return among random entry times with the same holding durations and costs (single-position methods).',
                    '','| Instrument | Method | Exposure | Beta | Alpha (annual) | Beating blend | Random-timing pct | Test years without trades |',
                    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            for row in summary['exposure_controls']:
                lines.append(f"| {row['symbol']} | {row['method']} | {number(row['median_exposure'])} | {number(row['median_beta'])} | {percent(row['median_alpha_annualized'])} | {percent(row['share_beating_exposure_matched_blend'])} | {number(row['median_random_timing_percentile'])} | {row['test_years_without_trades']} / {row['fold_seed_runs']} |")
    if summary.get('selection_bias'):
        lines+=['','## Selection-bias statistics','',
                'DSR deflates each method-seed path by the expected best of all method-seed trials for that instrument. PBO (CSCV, 8 blocks) is the probability that the path ranked best on half the years falls below the median on the other half. SPA tests H0: no method beats buy-and-hold (consistent p-value).',
                '','| Instrument | Best DSR (method:seed) | PBO | SPA p (ensembles) | SPA p range (seeds) |','| --- | --- | ---: | ---: | --- |']
        for row in summary['selection_bias']:
            best=max(row['deflated_sharpe'].items(),key=lambda kv:-1 if kv[1]['deflated_sharpe_ratio'] is None else kv[1]['deflated_sharpe_ratio'])
            seeds=[v['pvalue_consistent'] for k,v in row['spa_vs_buy_and_hold'].items() if k!='ensemble' and v.get('pvalue_consistent') is not None]
            spa_range=f"{min(seeds):.2f}–{max(seeds):.2f}" if seeds else 'undefined'
            lines.append(f"| {row['symbol']} | {number(best[1]['deflated_sharpe_ratio'])} ({best[0]}) | {number(row['pbo_cscv'].get('pbo'))} | {number(row['spa_vs_buy_and_hold']['ensemble'].get('pvalue_consistent'))} | {spa_range} |")
    fx_line=('- FX observed spread plus 0.1 bp slippage and 0.35 bp commission per side; financing debit 2% annualized on held notional over actual calendar time. This is a declared scenario, not historical broker swap data.'
             if settings.get('protocol','original')=='original' else
             '- FX observed spread plus 0.1 bp slippage and 0.35 bp commission per side; primary scenario has zero carry, stress applies the original 2%/yr debit. Historical carry signs are not modeled. Sunday stub candles are merged into the next session.')
    stock_line=('- Stock assumed full spread 2 bps, extra slippage 1 bp per side, commission $0.005/share with no minimum.'
                if settings.get('protocol','original')=='original' else
                '- Stock full spread max(2 bps, $0.01 tick), extra slippage 1 bp per side, commission $0.005/share with no minimum.')
    lines+=['','![Held-out equity](equity.png)','','## Data and execution',
        '', '- Stocks: contemporaneous execution prices, causal split-neutral indicator prices, split quantity/cost-basis changes, ex-date dividend entitlements, and NYSE sessions.',
        '- FX: historical Dukascopy bid/ask opens/closes; account currency USD with side-aware JPY conversion. Observed partial Sunday bars remain in the dataset.',
        '- All benchmark positions are long-only at 1× exposure, with continuous fractional sizing and zero idle-cash interest. The engine separately supports quantity steps, minimum commissions, shorting and leverage.',
        '- $100,000 starting capital per independently fitted fold. Normalized OOS paths compound each year; the benchmark cost model is scale invariant.',
        stock_line, fx_line,
        '- Stress: double assumed commission, slippage, modeled stock spread and financing. Observed FX bid/ask data stay unchanged. No strategy is refitted on stress/test returns.',
        '- No eligible training/validation configuration deploys flat cash for that fold and retains the failure record. Such folds are never deleted or retuned on test outcomes.',
        '', '## Interpretation limits', '',
        'This fixed, user-selected six-instrument universe is not a survivorship-free stock universe. Market-impact, partial fills, taxes, actual broker swap histories and point-in-time vendor revisions are not modeled. Dividends are credited as ex-date entitlements rather than delayed payment-date cash.',
        'Stops/targets are observed at candle close and execute next open. Daily high/low paths are not invented. All code, data, cost assumptions and candidate settings are recorded in the asset JSON reports.',
        'HAC p-values and stationary-block intervals are descriptive. Holm adjustment covers the declared asset × seed × pair family; it does not correct genetic search, configuration research, or unreported trials. DSR, PBO and SPA, when shown, address selection among the reported paths only.',
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
                  symbols=None,method_profile='paper',protocol='original',random_timing_reps=200,pbo_blocks=8,
                  spa_reps=1000):
    if method_profile not in CANDIDATE_PROFILES or protocol not in PROTOCOLS:
        raise ValueError(f'method_profile must be one of {sorted(CANDIDATE_PROFILES)}; protocol one of {sorted(PROTOCOLS)}')
    rules=PROTOCOLS[protocol]
    markets,manifest=load_dataset(data_directory)
    symbols=list(symbols) if symbols is not None else list(markets)
    if not symbols or len(set(symbols))!=len(symbols) or set(symbols)-set(markets):
        raise ValueError('Benchmark symbols must be a unique nonempty subset of the downloaded universe')
    candidates=CANDIDATE_PROFILES[method_profile]
    settings={'population_size':population_size,'generations':generations,'seeds':list(seeds),'workers':workers,
              'first_test_year':first_test_year,'last_test_year':last_test_year,'embargo_bars':embargo_bars,
              'bootstrap_reps':bootstrap_reps,'train_years':3,'validation_years':1,'initial_cash':100000,
              'candidate_configurations':candidates,'on_no_eligible':'cash'}
    if method_profile!='paper' or protocol!='original':
        # Extension settings enter the signature only when used, so original
        # reports keep resuming under the original settings dictionary.
        settings.update(method_profile=method_profile,protocol=protocol,protocol_rules=rules,
                        random_timing_reps=random_timing_reps)
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True)
    source_hash=_code_hash();reports={};start=perf_counter()
    for symbol in symbols:
        spec=benchmark_spec(symbol,fx_financing=rules['fx_financing'],stock_minimum_spread=rules['stock_minimum_spread'])
        stress_spec=benchmark_spec(symbol,stress=True,fx_financing=rules['fx_financing'],
                                   stock_minimum_spread=rules['stock_minimum_spread'])
        signature=hashlib.sha256(json.dumps({'settings':settings,'data':manifest['instruments'][symbol]['sha256'],
            'source':source_hash,'instrument':asdict(spec)},sort_keys=True).encode()).hexdigest()
        path=directory/(symbol.replace('/','_')+'.json')
        if resume and path.exists():
            cached=json.loads(path.read_text())
            if cached.get('benchmark_signature')!=signature:
                raise ValueError(f'{symbol}: existing report belongs to different code/data/settings; use a new output directory or no-resume')
            reports[symbol]=cached
            print(f'Verified completed asset: {symbol}',flush=True)
            continue
        frame=markets[symbol]
        periods_per_year=manifest['instruments'][symbol]['periods_per_year']
        merged_stubs=0
        if rules['merge_fx_weekend_stubs'] and '/' in symbol:
            frame,merged_stubs=merge_fx_weekend_stubs(frame)
            span=(frame['available_at'].iloc[-1]-frame['timestamp'].iloc[0]).total_seconds()/(365.25*86400)
            periods_per_year=float(len(frame)/span)
        run_markets={**markets,symbol:frame}
        folds=calendar_folds(frame,first_test_year=first_test_year,last_test_year=last_test_year,embargo_bars=embargo_bars)
        print(f'Running {symbol}: {len(folds)} calendar folds × {len(seeds)} seeds × 3 methods',flush=True)
        report=compare_methods(run_markets,ChromosomeContext(symbol),train_bars=756,validation_bars=252,test_bars=252,
            embargo_bars=embargo_bars,max_folds=len(folds),population_size=population_size,generations=generations,
            seeds=tuple(seeds),fee=0,slippage_bps=1 if '/' not in symbol else .1,
            periods_per_year=periods_per_year,bootstrap_reps=bootstrap_reps,
            workers=workers,instrument=spec,stress_instrument=stress_spec,
            initial_cash=100000,fold_plan=folds,include_paths=True,on_no_eligible='cash',
            method_configs={name:list(values) for name,values in candidates.items()},
            random_timing_reps=random_timing_reps,**rules['compare'])
        report.update(benchmark_signature=signature,data_provenance=manifest['instruments'][symbol],
                      calendar_protocol={'training_years':3,'validation_years':1,'test_years':list(range(first_test_year,last_test_year+1))})
        if method_profile!='paper' or protocol!='original':
            report.update(periods_per_year_used=periods_per_year,merged_fx_weekend_stubs=merged_stubs,
                          method_profile=method_profile,protocol=protocol)
        temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,allow_nan=False)+'\n');temporary.replace(path)
        reports[symbol]=report
        (directory/'progress.json').write_text(json.dumps({'completed':list(reports),'requested':symbols,'elapsed_seconds':perf_counter()-start},indent=2)+'\n')
        print(f'Finished {symbol}: {len(report["folds"])*3} method runs saved',flush=True)
    records,pairs,paths=summarize(reports,manifest,settings)
    summary={'schema_version':2,'data_kind':'real historical market data','universe':symbols,'settings':settings,
             'source_sha256':source_hash,'elapsed_seconds':perf_counter()-start,'runs':records,'paired_comparisons':pairs,
             'method_runs':sum(len(report['folds'])*3 for report in reports.values()),
             'cash_fallback_method_folds':sum(row['cash_fallback_folds'] for row in records if row['seed']!='ensemble'),
             'cost_scenarios':'primary and doubled assumed costs; same frozen models',
             'selection_bias':selection_bias(reports,manifest,settings,paths,blocks=pbo_blocks,spa_reps=spa_reps),
             'exposure_controls':exposure_summary(reports),
             'dataset_manifest':manifest}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    _render_report(directory,summary,paths)
    print(f'Completed benchmark: {summary["method_runs"]} method runs; report {directory/"report.md"}',flush=True)
    return summary
