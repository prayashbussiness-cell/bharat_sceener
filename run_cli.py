#!/usr/bin/env python3
"""CLI wrapper around app.engine_core.run_screen() - unchanged usage from before."""
import argparse
import json

from app.engine_core import run_screen, BUCKET_WEIGHTS, UNIVERSE_URLS


def main():
    p = argparse.ArgumentParser(description="Bharat Outperformer-style Indian equity screener")
    p.add_argument("--universe", default="nifty250", choices=list(UNIVERSE_URLS))
    p.add_argument("--universe-csv", default=None)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default=None)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--max-per-sector", type=int, default=4)
    p.add_argument("--pledge-csv", default="pledge_overrides.csv")
    p.add_argument("--cache-hours", type=float, default=12.0)
    p.add_argument("--weights", default=None)
    args = p.parse_args()

    weights = json.loads(args.weights) if args.weights else BUCKET_WEIGHTS
    result = run_screen(
        universe=args.universe, universe_csv=args.universe_csv, top=args.top,
        fast=args.fast, max_per_sector=args.max_per_sector,
        pledge_csv=args.pledge_csv, cache_hours=args.cache_hours, weights=weights,
    )
    out = args.out or f"screen_{args.universe}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n{len(result['picks'])} picks written to {out}")
    for r in result["picks"]:
        print(f"  {r['symbol']:<14} {r.get('name','')[:28]:<28} score={r['SCORE']}")


if __name__ == "__main__":
    main()
