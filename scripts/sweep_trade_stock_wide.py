from __future__ import annotations

import argparse
from pathlib import Path

from trade_stock_wide.intraday import load_hourly_histories
from trade_stock_wide.planner import build_daily_candidates
from trade_stock_wide.run import discover_symbols, load_backtests
from trade_stock_wide.sweep import run_parameter_sweep


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep trade_stock_wide selection and execution parameters")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbol list. Defaults to trainingdata/*.csv")
    ap.add_argument("--data-root", default="trainingdata")
    ap.add_argument("--limit-symbols", type=int, default=20)
    ap.add_argument("--top-ks", default="4,6,8")
    ap.add_argument("--selection-objectives", default="pnl,sortino,hybrid,tiny_net,torch_mlp")
    ap.add_argument("--watch-activation-pcts", default="0.003,0.005,0.0075")
    ap.add_argument("--steal-protection-pcts", default="0.003,0.004,0.006")
    ap.add_argument("--account-equity", type=float, default=100000.0)
    ap.add_argument("--pair-notional-fraction", type=float, default=0.50)
    ap.add_argument("--max-pair-notional-fraction", type=float, default=0.50)
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--backtest-days", type=int, default=30)
    ap.add_argument("--num-simulations", type=int, default=20)
    ap.add_argument("--model-override", default="chronos2")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--fill-buffer-bps", type=float, default=5.0)
    ap.add_argument("--selection-lookback-days", type=int, default=20)
    ap.add_argument("--tiny-net-hidden-dim", type=int, default=8)
    ap.add_argument("--tiny-net-epochs", type=int, default=120)
    ap.add_argument("--tiny-net-learning-rate", type=float, default=0.03)
    ap.add_argument("--tiny-net-l2", type=float, default=1e-4)
    ap.add_argument("--tiny-net-augment-copies", type=int, default=3)
    ap.add_argument("--tiny-net-noise-scale", type=float, default=0.04)
    ap.add_argument("--tiny-net-min-train-samples", type=int, default=12)
    ap.add_argument("--selection-seed", type=int, default=1337)
    ap.add_argument("--selection-torch-device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--selection-torch-batch-size", type=int, default=256)
    ap.add_argument("--hourly-root", default="trainingdatahourly")
    ap.add_argument("--daily-only-replay", action="store_true")
    ap.add_argument("--output", default=None, help="Optional CSV path for the leaderboard")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if args.symbols:
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    else:
        symbols = discover_symbols(data_root, limit=args.limit_symbols)
    if not symbols:
        raise SystemExit("No symbols resolved for sweep")

    frames = load_backtests(
        symbols,
        data_root=data_root,
        model_override=args.model_override,
        num_simulations=args.num_simulations,
    )
    if not frames:
        raise SystemExit("No backtest frames were produced")

    max_days = min(args.backtest_days, min(len(frame) for frame in frames.values()))
    raw_candidate_days: list[list] = []
    for offset in reversed(range(max_days)):
        raw_candidate_days.append(
            build_daily_candidates(
                frames,
                day_index=offset,
                require_realized_ohlc=True,
            )
        )

    hourly_by_symbol = None
    if not args.daily_only_replay:
        hourly_by_symbol = load_hourly_histories(frames.keys(), Path(args.hourly_root))

    leaderboard = run_parameter_sweep(
        raw_candidate_days,
        starting_equity=args.account_equity,
        selection_objectives=[item.strip() for item in args.selection_objectives.split(",") if item.strip()],
        top_ks=[int(item.strip()) for item in args.top_ks.split(",") if item.strip()],
        watch_activation_pcts=[float(item.strip()) for item in args.watch_activation_pcts.split(",") if item.strip()],
        steal_protection_pcts=[float(item.strip()) for item in args.steal_protection_pcts.split(",") if item.strip()],
        fee_bps=args.fee_bps,
        fill_buffer_bps=args.fill_buffer_bps,
        pair_notional_fraction=args.pair_notional_fraction,
        max_pair_notional_fraction=args.max_pair_notional_fraction,
        max_leverage=args.max_leverage,
        selection_lookback_days=args.selection_lookback_days,
        tiny_net_hidden_dim=args.tiny_net_hidden_dim,
        tiny_net_epochs=args.tiny_net_epochs,
        tiny_net_learning_rate=args.tiny_net_learning_rate,
        tiny_net_l2=args.tiny_net_l2,
        tiny_net_augment_copies=args.tiny_net_augment_copies,
        tiny_net_noise_scale=args.tiny_net_noise_scale,
        tiny_net_min_train_samples=args.tiny_net_min_train_samples,
        selection_seed=args.selection_seed,
        selection_torch_device=args.selection_torch_device,
        selection_torch_batch_size=args.selection_torch_batch_size,
        daily_only=args.daily_only_replay,
        hourly_by_symbol=hourly_by_symbol,
    )
    if leaderboard.empty:
        print("No sweep rows were produced.")
        return 0

    print(leaderboard.head(10).to_string(index=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(output_path, index=False)
        print(f"\nleaderboard_csv={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
