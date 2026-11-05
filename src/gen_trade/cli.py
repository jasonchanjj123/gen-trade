"""Small offline experiment CLI. All generated artifacts carry their data provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import ChromosomeContext, FeatureGene, read_market_data
from .runner import compare_methods, synthetic_markets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch=commands.add_parser('fetch-data',help='Download and audit the approved daily stock/FX histories')
    fetch.add_argument('--directory',type=Path,default=Path('data/stocks-fx-2015-2025'))
    fetch.add_argument('--start',default='2015-01-01')
    fetch.add_argument('--end',default='2026-01-01',help='Exclusive end date')
    fetch.add_argument('--refresh',action='store_true')
    benchmark=commands.add_parser('benchmark',help='Run all three methods on calendar-year stock/FX folds')
    benchmark.add_argument('--data-directory',type=Path,default=Path('data/stocks-fx-2015-2025'))
    benchmark.add_argument('--output',type=Path,default=Path('reports/stocks-fx-2015-2025'))
    benchmark.add_argument('--population',type=int,default=64)
    benchmark.add_argument('--generations',type=int,default=20)
    benchmark.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    benchmark.add_argument('--workers',type=int,default=3)
    benchmark.add_argument('--bootstrap-reps',type=int,default=500)
    benchmark.add_argument('--first-test-year',type=int,default=2019)
    benchmark.add_argument('--last-test-year',type=int,default=2025)
    benchmark.add_argument('--embargo-bars',type=int,default=5)
    benchmark.add_argument('--symbols',nargs='+',help='Optional subset of the downloaded universe')
    benchmark.add_argument('--no-resume',action='store_true')
    benchmark.add_argument('--method-profile',choices=('paper','improved','robust','robust_weekly','robust_monthly'),default='paper',
                           help='paper-faithful candidates, or labeled extension variants (docs/EXTENSIONS.md)')
    benchmark.add_argument('--method-configs',type=Path,help='JSON mapping every method to its candidate list (profile "custom")')
    benchmark.add_argument('--protocol',choices=('original','audited','audited2'),default='original',
                           help='original; audited (pooled seed selection, validated deployment, exposure controls, audited costs); audited2 (audited + blend-excess validation score)')
    benchmark.add_argument('--lockbox',type=Path,help='Lockbox declaration (default: lockbox.json in the repository root, if present)')
    benchmark.add_argument('--unlock-lockbox',action='store_true',help='Run held-out lockbox data once with a frozen profile')
    freeze=commands.add_parser('freeze-profiles',help='Declare a lockbox and freeze method profiles before touching it')
    freeze.add_argument('--output',type=Path,default=Path('lockbox.json'))
    freeze.add_argument('--first-test-year',type=int,required=True)
    freeze.add_argument('--symbols',nargs='*',default=[],help='Instruments reserved for the lockbox')
    freeze.add_argument('--profiles',nargs='*',help='Profiles to freeze (default: all)')
    freeze.add_argument('--note',default='')
    benchmark.add_argument('--random-timing-reps',type=int,default=200)
    reanalyze=commands.add_parser('reanalyze',help='Recompute a finished benchmark summary with the current statistics')
    reanalyze.add_argument('source',type=Path,help='Finished benchmark directory (left unchanged)')
    reanalyze.add_argument('--output',type=Path,required=True,help='New directory for summary.json/report.md')
    reanalyze.add_argument('--data-directory',type=Path,help='Dataset used by the benchmark; enables post hoc Ring random timing')
    reanalyze.add_argument('--random-timing-reps',type=int,default=200)
    reanalyze.add_argument('--spa-reps',type=int,default=1000)
    reanalyze.add_argument('--aligned-timing',action='store_true',
                           help='Recompute random timing of weekly/monthly methods with entries at their decision points')
    for command in ("demo", "compare"):
        p = commands.add_parser(command)
        if command == "compare":
            p.add_argument("data",help="CSV or Parquet containing UTC OHLCV and available_at")
            p.add_argument("--periods-per-year",type=float,required=True)
        else:
            p.add_argument("--bars",type=int,default=480)
            p.add_argument("--periods-per-year",type=float,default=252.)
        p.add_argument("--pair",default="BTC/USDT")
        p.add_argument("--cross-market-features",action="store_true",help="Add lagged XAU/USD and EUR/USD closes to MOO3")
        p.add_argument("--entry-hours",type=int,nargs="+",help="Permitted actual entry UTC hours")
        p.add_argument("--population",type=int,default=16)
        p.add_argument("--generations",type=int,default=2)
        p.add_argument("--train-bars",type=int,default=240)
        p.add_argument("--validation-bars",type=int,default=60,help="Separate configuration selection window; 0 requires one fixed config per method")
        p.add_argument("--test-bars",type=int,default=80)
        p.add_argument("--embargo-bars",type=int,default=5)
        p.add_argument("--folds",type=int,default=2)
        p.add_argument("--seed",type=int,default=42)
        p.add_argument("--seeds",type=int,nargs="+",help="Run every method for every seed; no held-out seed selection")
        p.add_argument("--workers",type=int,default=3,help="Independent method/fold/seed processes; 1 for serial")
        p.add_argument("--island-workers",type=int,default=1,help="Parallel ring islands; requires --workers 1")
        p.add_argument("--method-configs",type=Path,help="JSON mapping method names to lists of validation candidate settings")
        p.add_argument("--fee",type=float,default=.0025,help="Common fractional per-side fee for comparison")
        p.add_argument("--slippage-bps",type=float,default=0.)
        p.add_argument("--bootstrap-reps",type=int,default=300)
        p.add_argument("--output",type=Path,default=Path("reports/comparison.json"))
        p.add_argument("--persist",action="store_true",help="Also save to PostgreSQL using DATABASE_URL")
        p.add_argument("--experiment-id",help="Required with --persist; immutable result ID")
    args = parser.parse_args(argv)
    if getattr(args,'persist',False) and not args.experiment_id:
        parser.error("--persist requires --experiment-id")
    try:
        if args.command=='fetch-data':
            from .market_data import fetch_dataset
            manifest=fetch_dataset(args.directory,start=args.start,end=args.end,refresh=args.refresh)
            print(f"Verified {len(manifest['universe'])} instruments in {args.directory}")
            return 0
        if args.command=='reanalyze':
            from .benchmark import reanalyze as run_reanalysis
            summary=run_reanalysis(args.source,args.output,data_directory=args.data_directory,
                                   random_timing_reps=args.random_timing_reps,spa_reps=args.spa_reps,
                                   aligned_timing=args.aligned_timing)
            print(f"Reanalyzed {len(summary['universe'])} instruments into {args.output}")
            return 0
        if args.command=='benchmark':
            from .benchmark import run_benchmark
            run_benchmark(args.data_directory,args.output,population_size=args.population,generations=args.generations,
                          seeds=tuple(args.seeds),workers=args.workers,first_test_year=args.first_test_year,
                          last_test_year=args.last_test_year,embargo_bars=args.embargo_bars,bootstrap_reps=args.bootstrap_reps,
                          resume=not args.no_resume,symbols=args.symbols,method_profile=args.method_profile,
                          protocol=args.protocol,random_timing_reps=args.random_timing_reps,
                          method_configs=json.loads(args.method_configs.read_text()) if args.method_configs else None,
                          lockbox=args.lockbox,unlock_lockbox=args.unlock_lockbox)
            return 0
        if args.command=='freeze-profiles':
            from .benchmark import freeze_profiles
            record=freeze_profiles(args.output,first_test_year=args.first_test_year,symbols=args.symbols,
                                   profiles=args.profiles,note=args.note)
            print(f"Froze {len(record['frozen_profiles'])} profile(s); lockbox from {record['first_test_year']} in {args.output}")
            return 0
        markets = synthetic_markets(args.bars,args.seed) if args.command == "demo" else read_market_data(args.data)
        genes = tuple(FeatureGene(source,lag=1,max_age="4D") for source in ("XAU/USD","EUR/USD")) if args.cross_market_features else ()
        context = ChromosomeContext(args.pair,tuple(args.entry_hours) if args.entry_hours else None,genes)
        report = compare_methods(markets,context,train_bars=args.train_bars,test_bars=args.test_bars,
                                 validation_bars=args.validation_bars,workers=args.workers,island_workers=args.island_workers,
                                 seeds=tuple(args.seeds) if args.seeds else None,
                                 method_configs=json.loads(args.method_configs.read_text()) if args.method_configs else None,
                                 embargo_bars=args.embargo_bars,max_folds=args.folds,
                                 population_size=args.population,generations=args.generations,
                                 seed=args.seed,fee=args.fee,slippage_bps=args.slippage_bps,
                                 periods_per_year=args.periods_per_year,bootstrap_reps=args.bootstrap_reps,
                                 synthetic=args.command == "demo")
        text = json.dumps(report,indent=2,allow_nan=False)+"\n"
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
        if args.persist:
            import os
            from .storage import PostgresStore
            PostgresStore(os.environ["DATABASE_URL"]).save_report(report,args.experiment_id)
        print(f"Saved {len(report['folds'])} chronological fold/seed run(s), 3 methods each, to {args.output}")
        if args.command == "demo":
            print("Synthetic data only; this is a software validation, not evidence of trading performance.")
        return 0
    except (ValueError,KeyError,RuntimeError,OSError,TypeError) as exc:
        print(f"Experiment failed: {exc}",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
