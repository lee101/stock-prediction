#!/usr/bin/env python3
"""Entry points for gstockagent."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="gstockagent")
    sub = parser.add_subparsers(dest="cmd")

    sim = sub.add_parser("sim", help="Run single simulation")
    sim.add_argument("--model", default="gemini-3.1-lite")
    sim.add_argument("--leverage", type=float, default=1.0)
    sim.add_argument("--max-positions", type=int, default=5)
    sim.add_argument("--start", default="2025-10-01")
    sim.add_argument("--end", default="2026-01-10")
    sim.add_argument("--verbose", action="store_true")

    sw = sub.add_parser("sweep", help="Run hyperparameter sweep")
    sw.add_argument("--start", default="2025-10-01")
    sw.add_argument("--end", default="2026-01-10")
    sw.add_argument("--verbose", action="store_true")

    an = sub.add_parser("analyze", help="Detailed analysis of a run")
    an.add_argument("--model", default="gemini-3.1-lite")
    an.add_argument("--leverage", type=float, default=1.0)
    an.add_argument("--max-positions", type=int, default=5)
    an.add_argument("--start", default="2025-10-01")
    an.add_argument("--end", default="2026-01-10")

    live = sub.add_parser("live", help="Live trading daemon")
    live.add_argument("--model", default="gemini-3.1-lite")
    live.add_argument("--leverage", type=float, default=1.0)
    live.add_argument("--max-positions", type=int, default=5)
    live.add_argument("--interval-hours", type=int, default=24)
    live.add_argument("--dry-run", action="store_true")
    live.add_argument("--daemon", action="store_true")

    args = parser.parse_args()

    if args.cmd == "sim":
        from .config import GStockConfig
        from .simulator import run_simulation
        cfg = GStockConfig(leverage=args.leverage, model=args.model,
                           max_positions=args.max_positions)
        r = run_simulation(cfg, args.start, args.end,
                          use_cache=True, verbose=args.verbose)
        if "error" in r:
            print(f"ERROR: {r['error']}")
            sys.exit(1)
        print(f"ret={r['total_return_pct']:+.1f}% dd={r['max_drawdown_pct']:.1f}% "
              f"sort={r['sortino']:.2f} shrp={r['sharpe']:.2f} "
              f"trades={r['n_trades']} wr={r['win_rate_pct']:.1f}%")

    elif args.cmd == "sweep":
        from .sweep import run_sweep
        run_sweep(args.start, args.end, args.verbose)

    elif args.cmd == "analyze":
        from .analyze import analyze_run
        from .config import GStockConfig
        cfg = GStockConfig(leverage=args.leverage, model=args.model,
                           max_positions=args.max_positions)
        analyze_run(cfg, args.start, args.end)

    elif args.cmd == "live":
        from .trade_live import main as live_main
        sys.argv = ["gstockagent", "--model", args.model,
                    "--leverage", str(args.leverage),
                    "--max-positions", str(args.max_positions),
                    "--interval-hours", str(args.interval_hours)]
        if args.dry_run:
            sys.argv.append("--dry-run")
        if args.daemon:
            sys.argv.append("--daemon")
        live_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
