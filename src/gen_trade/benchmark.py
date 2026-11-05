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
from .statistics import (deflated_sharpe_ratio,detailed_statistics,excess_over_blend,holm_adjust,paired_comparison,
                         pbo_cscv,probabilistic_sharpe_ratio,rademacher_antiserum,spa_excess,spa_test,stepm_excess)

PROTOCOLS = {
    'original':{'compare':{}, 'fx_financing':.02, 'stock_minimum_spread':0., 'merge_fx_weekend_stubs':False},
    # EXTENSION (docs/EXTENSIONS.md): audit items 3, 6, 8 and 13.
    'audited':{'compare':{'selection':'pooled','deploy':'validated','min_validation_trades':2,
                          'validation_curves':True,'exposure_controls_enabled':True},
               'fx_financing':0., 'stock_minimum_spread':.01, 'merge_fx_weekend_stubs':True},
    # EXTENSION (review 2026-11-05): audited, plus configuration choice on the
    # validation information ratio of excess over the exposure-matched blend.
    'audited2':{'compare':{'selection':'pooled','deploy':'validated','min_validation_trades':2,
                           'validation_curves':True,'exposure_controls_enabled':True,'validation_metric':'blend_excess'},
                'fx_financing':0., 'stock_minimum_spread':.01, 'merge_fx_weekend_stubs':True},
}
LOCKBOX_FILE='lockbox.json'


def profile_hash(candidates):
    """Content hash of a method profile's candidate configurations."""
    return hashlib.sha256(json.dumps(candidates,sort_keys=True).encode()).hexdigest()


def freeze_profiles(path,*,first_test_year,symbols=(),profiles=None,note=''):
    """EXTENSION: declare a lockbox and freeze the named profiles before using it.

    Benchmarks refuse lockbox instruments or test years from ``first_test_year``
    on unless explicitly unlocked, and an unlock only runs a profile whose
    candidate configurations still hash to the frozen value.
    """
    names=list(profiles) if profiles else sorted(CANDIDATE_PROFILES)
    if set(names)-set(CANDIDATE_PROFILES):
        raise ValueError(f'Unknown profiles: {sorted(set(names)-set(CANDIDATE_PROFILES))}')
    record={'created':pd.Timestamp.now(tz='UTC').isoformat(),'first_test_year':int(first_test_year),
            'symbols':sorted(symbols),'frozen_profiles':{name:profile_hash(CANDIDATE_PROFILES[name]) for name in names},
            'code_sha256_at_freeze':_code_hash(),'note':note or 'Untouched held-out data; unlock once, after freezing.'}
    Path(path).write_text(json.dumps(record,indent=2)+'\n')
    return record


def check_lockbox(lockbox,*,symbols,last_test_year,method_profile,candidates,unlock):
    """Refuse lockbox data unless unlocking a frozen, unchanged profile."""
    if lockbox is None:
        return None
    touched=sorted(set(symbols)&set(lockbox.get('symbols',[])))
    future=last_test_year>=int(lockbox['first_test_year'])
    if not touched and not future:
        return {'lockbox_touched':False}
    if not unlock:
        raise ValueError(f"Lockbox: symbols {touched} / test years >= {lockbox['first_test_year']} are held out; "
                         'freeze the profile and pass unlock to run them once')
    frozen=lockbox.get('frozen_profiles',{}).get(method_profile)
    if frozen is None or frozen!=profile_hash(candidates):
        raise ValueError(f'Lockbox: profile {method_profile!r} is not frozen or changed since freezing')
    return {'lockbox_touched':True,'unlocked_symbols':touched,'unlocked_from_year':int(lockbox['first_test_year']),
            'profile_hash':frozen,'lockbox_created':lockbox.get('created'),
            'code_changed_since_freeze':lockbox.get('code_sha256_at_freeze')!=_code_hash()}


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
    used=report.get('periods_per_year_used')
    return used if used is not None else manifest['instruments'][symbol]['periods_per_year']


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
        row={'symbol':symbol,'trials':len(keys),'trial_sharpe_variance_per_period':variance,
             'deflated_sharpe':dsr,'pbo_cscv':pbo,'spa_vs_buy_and_hold':spa}
        if report['folds'] and all('exposure_controls' in f for f in report['folds']):
            row.update(excess_tests(report,settings['seeds'],periods_per_year=_periods_per_year(report,manifest,symbol),
                                    spa_reps=spa_reps))
        output.append(row)
    return output


def _coefficient(controls,kind):
    if kind=='blend':
        return float(controls['mean_exposure'])
    beta=controls['alpha_beta_vs_buy_and_hold'].get('beta')
    return float(beta) if beta is not None else 0.


def stitch_excess(asset_report,name,seeds,kind='blend'):
    """EXTENSION (review 2026-11-05): held-out per-period excess returns.

    ``kind='blend'`` subtracts the fold's exposure-matched cash/buy-and-hold
    blend (coefficient = the fold's mean exposure); ``kind='beta'`` subtracts
    the fold's beta times buy-and-hold, so the mean is the regression alpha.
    ``seeds`` is one seed or a list; a list is the equal-capital ensemble, whose
    coefficient is the seed mean (exposure is linear in capital at the start
    of each year). Returns (excess, test year of each period).
    """
    seeds=list(seeds) if isinstance(seeds,(list,tuple)) else [seeds]
    by_fold={}
    for fold in asset_report['folds']:
        if fold['base_seed'] not in seeds:
            continue
        if 'exposure_controls' not in fold:
            raise ValueError('Excess returns need folds with exposure controls (audited protocol)')
        path=fold['methods'][name]['backtest']
        entry=by_fold.setdefault(fold['bounds']['test_start'],{'curves':[],'coefficients':[],
            'baseline':fold['baseline_backtest'],'year':fold['test_first_open'][:4]})
        entry['curves'].append(np.asarray(path['equity'],dtype=float)/path['initial_cash'])
        entry['coefficients'].append(_coefficient(fold['exposure_controls'][name],kind))
    excess,years=[],[]
    for start in sorted(by_fold):
        entry=by_fold[start]
        if len(entry['curves'])!=len(seeds):
            raise ValueError('Every seed needs every test fold for an excess path')
        curve=np.mean(entry['curves'],axis=0)
        values=excess_over_blend(np.diff(curve)/curve[:-1],_returns(entry['baseline']),float(np.mean(entry['coefficients'])))
        excess.append(values);years.extend([entry['year']]*len(values))
    return np.concatenate(excess),years


def excess_tests(report,seeds,*,periods_per_year,spa_reps=1000):
    """EXTENSION: SPA and StepM on excess over the blend / beta hedge, plus RAS.

    These replace buy-and-hold as the null for timing skill: a method at 10%
    exposure cannot beat buy-and-hold in a rising market whatever its timing.
    RAS bounds every method-seed path's excess Sharpe at once.
    """
    def matrix(kind,seed):
        return np.column_stack([stitch_excess(report,name,seeds if seed=='ensemble' else seed,kind)[0]
                                for name in METHOD_NAMES])
    labels=[*seeds,'ensemble']
    output={'spa_excess_vs_blend':{str(s):spa_excess(matrix('blend',s),reps=spa_reps,seed=0 if s=='ensemble' else int(s))
                                   for s in labels},
            'spa_beta_hedged':{str(s):spa_excess(matrix('beta',s),reps=spa_reps,seed=0 if s=='ensemble' else int(s))
                               for s in labels},
            'stepm_excess_vs_blend':stepm_excess(matrix('blend','ensemble'),METHOD_NAMES,reps=spa_reps,seed=0)}
    keys=[(name,seed) for name in METHOD_NAMES for seed in seeds]
    paths=np.column_stack([stitch_excess(report,name,seed,'blend')[0] for name,seed in keys])
    output['ras_excess_vs_blend']={'paths':[f'{name}:{seed}' for name,seed in keys],
                                   **rademacher_antiserum(paths,reps=2000,seed=0,periods_per_year=periods_per_year)}
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


def _pooled(values,reference):
    if len(values)<3:
        return {'cells':len(values),'mean':None,'reference':reference}
    x=np.array(list(values.values()))
    years={}
    for (_,year),value in values.items():
        years.setdefault(year,[]).append(value)
    yearly=np.array([np.mean(v) for _,v in sorted(years.items())])
    def t(sample):
        if len(sample)<2:return None
        sd=float(np.std(sample,ddof=1))
        return None if sd==0 else float((np.mean(sample)-reference)/(sd/np.sqrt(len(sample))))
    return {'cells':int(len(x)),'mean':float(x.mean()),'reference':reference,'above_reference':int(np.sum(x>reference)),
            't_cells_independent':t(x),'years':int(len(yearly)),'t_year_clustered':t(yearly)}


def pooled_controls(reports,manifest):
    """EXTENSION (review 2026-11-05): cross-instrument timing evidence per method.

    Cells are instrument x test year with seeds averaged (seeds share the
    year). Each metric reports the mean over cells, a t-statistic treating
    cells as independent, and a year-clustered t over the yearly
    cross-instrument means, which allows for common shocks within a year.
    """
    rows=[]
    for name in METHOD_NAMES:
        cells={}
        for symbol,report in reports.items():
            annual=_periods_per_year(report,manifest,symbol)
            for fold in report['folds']:
                controls=fold.get('exposure_controls',{}).get(name)
                if controls is None:
                    continue
                cell=cells.setdefault((symbol,fold['test_first_open'][:4]),{'blend':[],'alpha':[],'timing':[]})
                mean=controls['minus_exposure_matched_blend']['paired_return_difference_inference']['mean']
                if mean is not None:cell['blend'].append(mean*annual)
                alpha=controls['alpha_beta_vs_buy_and_hold'].get('alpha_annualized')
                if alpha is not None:cell['alpha'].append(alpha)
                if controls.get('random_timing'):cell['timing'].append(controls['random_timing']['percentile'])
        if not cells:
            continue
        row={'method':name,'instrument_years':len(cells)}
        for key,reference,label in (('blend',0.,'excess_vs_blend_annualized'),('alpha',0.,'alpha_annualized'),
                                    ('timing',.5,'random_timing_percentile')):
            row[label]=_pooled({k:float(np.mean(v[key])) for k,v in cells.items() if v[key]},reference)
        rows.append(row)
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
    if summary.get('reanalysis'):
        aligned=summary['reanalysis'].get('posthoc_aligned_random_timing_runs',0)
        lines[2:2]=[f"Reanalysis of `{summary['reanalysis']['source_directory']}` with the current statistics; per-asset reports unchanged, no strategy refitted. Ring random timing added post hoc for {summary['reanalysis']['posthoc_ring_random_timing_folds']} fold-seed runs"
                    +(f"; random timing recomputed with cadence-aligned placements for {aligned} method runs." if aligned else "."),'']
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
                    'Beating the blend = share of test years whose mean daily return exceeds a static cash/buy-and-hold mix with the same mean exposure. Random-timing percentile = rank of the net return among random entry times with the same holding durations and costs (per sleeve for Ring when computed).',
                    '','| Instrument | Method | Exposure | Beta | Alpha (annual) | Beating blend | Random-timing pct | Test years without trades |',
                    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            for row in summary['exposure_controls']:
                lines.append(f"| {row['symbol']} | {row['method']} | {number(row['median_exposure'])} | {number(row['median_beta'])} | {percent(row['median_alpha_annualized'])} | {percent(row['share_beating_exposure_matched_blend'])} | {number(row['median_random_timing_percentile'])} | {row['test_years_without_trades']} / {row['fold_seed_runs']} |")
    if summary.get('pooled_controls'):
        def tval(value):return '–' if value is None else f'{value:+.2f}'
        lines+=['','## Pooled timing evidence (all instruments)','',
                'Cells are instrument × test year with seeds averaged. t (cells) treats cells as independent; t (years) uses the yearly cross-instrument means, allowing common shocks within a year. Random timing is per sleeve for Ring.',
                '','| Method | Cells | Excess vs blend (annual) | t (cells) | t (years) | Alpha (annual) | t (years) | Random-timing pct | t (years) |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in summary['pooled_controls']:
            blend,alpha,timing=row['excess_vs_blend_annualized'],row['alpha_annualized'],row['random_timing_percentile']
            lines.append(f"| {row['method']} | {row['instrument_years']} | {percent(blend['mean'])} | {tval(blend.get('t_cells_independent'))} | {tval(blend.get('t_year_clustered'))} | {percent(alpha['mean'])} | {tval(alpha.get('t_year_clustered'))} | {number(timing['mean'])} | {tval(timing.get('t_year_clustered'))} |")
    if summary.get('selection_bias'):
        lines+=['','## Selection-bias statistics','',
                'DSR deflates each method-seed path by the expected best of all method-seed trials for that instrument. PBO (CSCV, 8 blocks) is the probability that the path ranked best on half the years falls below the median on the other half. SPA tests H0: no method beats buy-and-hold (consistent p-value).',
                '','| Instrument | Best DSR (method:seed) | PBO | SPA p (ensembles) | SPA p range (seeds) |','| --- | --- | ---: | ---: | --- |']
        for row in summary['selection_bias']:
            best=max(row['deflated_sharpe'].items(),key=lambda kv:-1 if kv[1]['deflated_sharpe_ratio'] is None else kv[1]['deflated_sharpe_ratio'])
            seeds=[v['pvalue_consistent'] for k,v in row['spa_vs_buy_and_hold'].items() if k!='ensemble' and v.get('pvalue_consistent') is not None]
            spa_range=f"{min(seeds):.2f}–{max(seeds):.2f}" if seeds else 'undefined'
            lines.append(f"| {row['symbol']} | {number(best[1]['deflated_sharpe_ratio'])} ({best[0]}) | {number(row['pbo_cscv'].get('pbo'))} | {number(row['spa_vs_buy_and_hold']['ensemble'].get('pvalue_consistent'))} | {spa_range} |")
        if any('spa_excess_vs_blend' in row for row in summary['selection_bias']):
            lines+=['','## Tests against the exposure-matched blend','',
                    'SPA with a zero benchmark on excess returns: H0 = no method has positive mean excess over its exposure-matched blend (or, beta-hedged, positive alpha). StepM lists methods whose excess is positive at 5% familywise error (ensembles). RAS: method-seed paths whose excess Sharpe stays positive after the Rademacher data-snooping haircut.',
                    '','| Instrument | SPA p, excess vs blend (ensembles) | SPA p range (seeds) | SPA p, beta-hedged (ensembles) | StepM superior | RAS positive paths |',
                    '| --- | ---: | --- | ---: | --- | --- |']
            for row in summary['selection_bias']:
                if 'spa_excess_vs_blend' not in row:continue
                spa,hedged=row['spa_excess_vs_blend'],row['spa_beta_hedged']
                seeds=[v['pvalue_consistent'] for k,v in spa.items() if k!='ensemble' and v.get('pvalue_consistent') is not None]
                spa_range=f"{min(seeds):.2f}–{max(seeds):.2f}" if seeds else 'undefined'
                ras=row['ras_excess_vs_blend']
                positive=[p for p,ok in zip(ras['paths'],ras['rademacher_positive']) if ok]
                lines.append(f"| {row['symbol']} | {number(spa['ensemble'].get('pvalue_consistent'))} | {spa_range} | {number(hedged['ensemble'].get('pvalue_consistent'))} | {', '.join(row['stepm_excess_vs_blend']['superior_models']) or 'none'} | {', '.join(positive) or 'none'} |")
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
                  spa_reps=1000,method_configs=None,lockbox=None,unlock_lockbox=False):
    if method_configs is not None:
        method_profile='custom'
        if set(method_configs)!=set(METHOD_NAMES):
            raise ValueError(f'method_configs must give candidates for every method: {METHOD_NAMES}')
        candidates={name:tuple(dict(c) for c in method_configs[name]) for name in METHOD_NAMES}
    elif method_profile not in CANDIDATE_PROFILES:
        raise ValueError(f'method_profile must be one of {sorted(CANDIDATE_PROFILES)}')
    else:
        candidates=CANDIDATE_PROFILES[method_profile]
    if protocol not in PROTOCOLS:
        raise ValueError(f'protocol must be one of {sorted(PROTOCOLS)}')
    rules=PROTOCOLS[protocol]
    markets,manifest=load_dataset(data_directory)
    symbols=list(symbols) if symbols is not None else list(markets)
    if not symbols or len(set(symbols))!=len(symbols) or set(symbols)-set(markets):
        raise ValueError('Benchmark symbols must be a unique nonempty subset of the downloaded universe')
    if lockbox is None and (Path(__file__).resolve().parents[2]/LOCKBOX_FILE).exists():
        lockbox=Path(__file__).resolve().parents[2]/LOCKBOX_FILE
    lock=check_lockbox(json.loads(Path(lockbox).read_text()) if lockbox else None,symbols=symbols,
                       last_test_year=last_test_year,method_profile=method_profile,candidates=candidates,
                       unlock=unlock_lockbox)
    settings={'population_size':population_size,'generations':generations,'seeds':list(seeds),'workers':workers,
              'first_test_year':first_test_year,'last_test_year':last_test_year,'embargo_bars':embargo_bars,
              'bootstrap_reps':bootstrap_reps,'train_years':3,'validation_years':1,'initial_cash':100000,
              'candidate_configurations':candidates,'on_no_eligible':'cash'}
    if method_profile!='paper' or protocol!='original':
        # Extension settings enter the signature only when used, so original
        # reports keep resuming under the original settings dictionary.
        settings.update(method_profile=method_profile,protocol=protocol,protocol_rules=rules,
                        random_timing_reps=random_timing_reps)
    if lock and lock.get('lockbox_touched'):
        settings['lockbox']=lock
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
                          method_profile=method_profile,benchmark_protocol=protocol)
        if not isinstance(report.get('protocol'),dict):
            raise ValueError(f'{symbol}: report protocol block was overwritten')
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
             'pooled_controls':pooled_controls(reports,manifest),
             'dataset_manifest':manifest}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    _render_report(directory,summary,paths)
    print(f'Completed benchmark: {summary["method_runs"]} method runs; report {directory/"report.md"}',flush=True)
    return summary


def _execution_from(record):
    from dataclasses import fields
    from .instruments import InstrumentSpec
    values={f.name:record[f.name] for f in fields(ExecutionConfig) if f.name in record}
    if values.get('instrument') is not None:
        values['instrument']=InstrumentSpec(**values['instrument'])
    if values.get('entry_hours') is not None:
        values['entry_hours']=tuple(values['entry_hours'])
    return ExecutionConfig(**values)


def _result_from(path,execution):
    from .backtest import Trade
    trades=tuple(Trade(**{**row,'entry_time':pd.Timestamp(row['entry_time']),'exit_time':pd.Timestamp(row['exit_time'])})
                 for row in path['trades'])
    return BacktestResult(np.asarray(path['equity'],dtype=float),np.asarray([t.return_fraction for t in trades]),trades,
                          execution,tuple(pd.to_datetime(path['timestamps'],utc=True)))


def posthoc_timing(reports,data_directory,settings,manifest,reps=200,aligned=False):
    """EXTENSION: fill or correct random-timing controls in finished reports.

    Adds Ring's missing controls and, with ``aligned``, recomputes the control
    of every method that decided weekly or monthly with entries restricted to
    its decision points (the earlier value is kept as
    ``random_timing_unaligned``). Each fold's test window is rebuilt from the
    same dataset (hash checked) and protocol, and the run is refused unless
    next-open buy-and-hold on it reproduces the stored baseline equity. Only
    stored trades are used; no strategy is refitted or re-predicted.
    """
    from dataclasses import replace
    from .backtest import run_backtest
    from .methods.cadence import decision_bars
    from .runner import random_timing
    markets,data_manifest=load_dataset(data_directory)
    rules=settings.get('protocol_rules') or PROTOCOLS[settings.get('protocol','original')]
    events=[]
    for symbol,report in reports.items():
        if data_manifest['instruments'][symbol]['sha256']!=manifest['instruments'][symbol]['sha256']:
            raise ValueError(f'{symbol}: dataset differs from the one the benchmark used')
        frame=markets[symbol]
        if rules.get('merge_fx_weekend_stubs') and '/' in symbol:
            frame,_=merge_fx_weekend_stubs(frame)
        for fold in report['folds']:
            if 'exposure_controls' not in fold:
                continue
            bounds=fold['bounds']
            test=frame.iloc[bounds['test_start']:bounds['test_stop']].reset_index(drop=True)
            previous=frame['timestamp'].iloc[bounds['test_start']-1] if bounds['test_start'] else None
            checked=False
            for name in METHOD_NAMES:
                controls=fold['exposure_controls'][name]
                method=fold['methods'][name]
                cadence=method.get('parameters',{}).get('decision_cadence','daily')
                missing=name=='igtsp_ring' and controls.get('random_timing') is None and cadence=='daily'
                realign=aligned and cadence!='daily'
                if not (missing or realign):
                    continue
                execution=_execution_from(method['parameters']['execution'])
                if not checked:
                    plain=replace(execution,take_profit=None,stop_loss=None,max_holding_days=None)
                    baseline=run_backtest(test,np.ones(len(test),dtype=np.int8),plain)
                    stored=np.asarray(fold['baseline_backtest']['equity'],dtype=float)
                    if baseline.equity.shape!=stored.shape or not np.allclose(baseline.equity,stored,rtol=1e-10,atol=1e-6):
                        raise ValueError(f'{symbol} fold {fold["index"]}: reconstructed test window does not reproduce the stored baseline')
                    checked=True
                decision=decision_bars(test['timestamp'],cadence,previous) if cadence!='daily' else None
                timing,note=random_timing(_result_from(method['backtest'],execution),test,execution,reps=reps,
                                          seed=int(fold['effective_seed']),portfolio=name=='igtsp_ring',decision=decision)
                if realign and controls.get('random_timing') is not None:
                    controls['random_timing_unaligned']=controls['random_timing']
                controls['random_timing']=timing
                controls['random_timing_note']=note or ('recomputed post hoc with cadence-aligned placements'
                                                        if realign else 'computed post hoc from stored trades')
                events.append({'symbol':symbol,'method':name,'fold':fold['index'],'seed':fold['base_seed'],
                               'test_year':fold['test_first_open'][:4],'cadence':cadence,'aligned':bool(realign),
                               'random_timing':timing,'note':note})
    return events


posthoc_ring_timing=posthoc_timing  # earlier name


def reanalyze(source_directory,output_directory,*,data_directory=None,random_timing_reps=200,spa_reps=1000,pbo_blocks=8,
              aligned_timing=False):
    """EXTENSION (review 2026-11-05): recompute a finished benchmark's summary.

    Per-asset reports are read and never modified; the new summary, report and
    chart go to a separate directory. With ``data_directory`` Ring's missing
    random-timing controls are added post hoc (see ``posthoc_ring_timing``).
    """
    source,output=Path(source_directory),Path(output_directory)
    if source.resolve()==output.resolve():
        raise ValueError('Write the reanalysis to a new directory; source reports stay unchanged')
    old=json.loads((source/'summary.json').read_text())
    settings,manifest,symbols=old['settings'],old['dataset_manifest'],old['universe']
    reports={symbol:json.loads((source/(symbol.replace('/','_')+'.json')).read_text()) for symbol in symbols}
    if aligned_timing and not data_directory:
        raise ValueError('Cadence-aligned timing needs the data directory')
    events=posthoc_timing(reports,data_directory,settings,manifest,random_timing_reps,aligned_timing) if data_directory else []
    records,pairs,paths=summarize(reports,manifest,settings)
    summary={**{key:old[key] for key in ('schema_version','data_kind','universe','settings','source_sha256') if key in old},
             'reanalysis':{'source_directory':str(source),'reanalysis_code_sha256':_code_hash(),
                           'posthoc_ring_random_timing_folds':sum(e['method']=='igtsp_ring' and not e['aligned'] for e in events),
                           'posthoc_aligned_random_timing_runs':sum(e['aligned'] for e in events),
                           'note':'Statistics recomputed from unchanged per-asset reports; no strategy was refitted.'},
             'runs':records,'paired_comparisons':pairs,
             'method_runs':sum(len(report['folds'])*3 for report in reports.values()),
             'cash_fallback_method_folds':sum(row['cash_fallback_folds'] for row in records if row['seed']!='ensemble'),
             'cost_scenarios':old.get('cost_scenarios'),
             'selection_bias':selection_bias(reports,manifest,settings,paths,blocks=pbo_blocks,spa_reps=spa_reps),
             'exposure_controls':exposure_summary(reports),
             'pooled_controls':pooled_controls(reports,manifest),
             'dataset_manifest':manifest}
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if events:
        (output/'posthoc_random_timing.json').write_text(json.dumps(events,indent=2,allow_nan=False)+'\n')
    _render_report(output,summary,paths)
    return summary
