#!/usr/bin/env python3
"""
Daily trading loop using the neural daily model.
Integrates trained model into live trading system.
"""
import time
from datetime import datetime
from pathlib import Path
from neuraldailytraining import DailyTradingRuntime
from neural_trade_stock_e2e import _build_dataset_config
import argparse

def get_latest_final_checkpoint() -> Path:
    """Find the most recent final model checkpoint."""
    checkpoints_dir = Path("neuraldailytraining/checkpoints")
    final_dirs = sorted(checkpoints_dir.glob("final_*/"))

    if not final_dirs:
        raise FileNotFoundError("No final model checkpoints found. Run Phase 2 first.")

    latest_final = final_dirs[-1]
    # Get latest epoch from that directory
    epochs = sorted(latest_final.glob("epoch_*.pt"))

    if not epochs:
        raise FileNotFoundError(f"No epoch checkpoints in {latest_final}")

    return epochs[-1]

def run_daily_trading_loop(checkpoint_path: str, symbols: list, dry_run: bool = True):
    """
    Main daily trading loop.

    Args:
        checkpoint_path: Path to trained model checkpoint
        symbols: List of symbols to trade
        dry_run: If True, only print signals without executing
    """
    print("="*80)
    print("Neural Daily Trading Bot")
    print("="*80)
    print(f"Model: {checkpoint_path}")
    print(f"Symbols: {symbols}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print("="*80)

    # Load model
    dataset_cfg = _build_dataset_config(argparse.Namespace(
        data_root="trainingdata/train",
        forecast_cache="strategytraining/forecast_cache",
        sequence_length=256,
        val_fraction=0.2,
        validation_days=40,
        device=None,
        symbols=symbols,
    ))

    runtime = DailyTradingRuntime(
        checkpoint_path,
        dataset_config=dataset_cfg,
        device="cuda",
    )

    print(f"\n✅ Model loaded successfully")
    print(f"Risk threshold: {runtime.risk_threshold}")

    # Generate today's trading plans
    plans = runtime.generate_plans(symbols)

    print(f"\n📊 Generated {len(plans)} trading plans for {datetime.now().date()}")
    print("\n" + "="*80)

    for plan in plans:
        print(f"\n{plan.symbol}:")
        print(f"  Buy at:  ${plan.buy_price:.2f}")
        print(f"  Sell at: ${plan.sell_price:.2f}")
        print(f"  Amount:  {plan.trade_amount:.4f}")
        print(f"  Reference: ${plan.reference_close:.2f}")

        spread_pct = (plan.sell_price - plan.buy_price) / plan.buy_price * 100
        print(f"  Spread: {spread_pct:.2f}%")

        if not dry_run:
            # TODO: Execute trades via alpaca_wrapper
            # from alpaca_wrapper import place_limit_order
            # place_limit_order(plan.symbol, plan.trade_amount, plan.buy_price, 'buy')
            # place_limit_order(plan.symbol, plan.trade_amount, plan.sell_price, 'sell')
            print(f"  ⚠️ LIVE TRADING NOT IMPLEMENTED YET")

    print("\n" + "="*80)

    return plans

def main():
    parser = argparse.ArgumentParser(description="Daily neural trading bot")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (default: latest final model)")
    parser.add_argument("--symbols", nargs="+", help="Symbols to trade")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: dry run)")
    args = parser.parse_args()

    # Get checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = str(get_latest_final_checkpoint())
        print(f"Using latest final model: {checkpoint_path}")

    # Default symbols (those with Chronos forecasts)
    default_symbols = [
        "BTCUSD", "ETHUSD", "LINKUSD", "UNIUSD",
        "SPY", "QQQ", "GLD",
        "ADBE", "AVGO", "CRM", "INTC",
        "SHOP", "PLTR",
    ]

    symbols = args.symbols or default_symbols

    # Run trading
    plans = run_daily_trading_loop(
        checkpoint_path=checkpoint_path,
        symbols=symbols,
        dry_run=not args.live,
    )

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
