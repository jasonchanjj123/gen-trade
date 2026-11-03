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
        p.add_argument("--test-bars",type=int,default=80)
        p.add_argument("--embargo-bars",type=int,default=5)
        p.add_argument("--folds",type=int,default=2)
        p.add_argument("--seed",type=int,default=42)
        p.add_argument("--fee",type=float,default=.0025,help="Common fractional per-side fee for comparison")
        p.add_argument("--slippage-bps",type=float,default=0.)
        p.add_argument("--bootstrap-reps",type=int,default=300)
        p.add_argument("--output",type=Path,default=Path("reports/comparison.json"))
        p.add_argument("--persist",action="store_true",help="Also save to PostgreSQL using DATABASE_URL")
        p.add_argument("--experiment-id",help="Required with --persist; immutable result ID")
    args = parser.parse_args(argv)
    if args.persist and not args.experiment_id:
        parser.error("--persist requires --experiment-id")
    try:
        markets = synthetic_markets(args.bars,args.seed) if args.command == "demo" else read_market_data(args.data)
        genes = tuple(FeatureGene(source,lag=1,max_age="4D") for source in ("XAU/USD","EUR/USD")) if args.cross_market_features else ()
        context = ChromosomeContext(args.pair,tuple(args.entry_hours) if args.entry_hours else None,genes)
        report = compare_methods(markets,context,train_bars=args.train_bars,test_bars=args.test_bars,
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
        print(f"Saved {len(report['folds'])} chronological fold(s) to {args.output}")
        if args.command == "demo":
            print("Synthetic data only; this is a software validation, not evidence of trading performance.")
        return 0
    except (ValueError,KeyError,RuntimeError) as exc:
        print(f"Experiment failed: {exc}",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
