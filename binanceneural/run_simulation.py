from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import DatasetConfig, TrainingConfig
from .data import BinanceHourlyDataModule
from .inference import generate_actions_from_frame
from .marketsimulator import BinanceMarketSimulator, SimulationConfig, save_trade_plot
from .model import BinancePolicyBase, build_policy, policy_config_from_payload


def _load_model(checkpoint_path: Path, input_dim: int, default_cfg: TrainingConfig) -> BinancePolicyBase:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    cfg = payload.get("config", default_cfg)
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Binance hourly market simulation.")
    parser.add_argument("--symbol", default="BTCUSD", help="Symbol to simulate (e.g., BTCUSD)")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--horizon", type=int, default=1, help="Chronos horizon used for actions")
    parser.add_argument("--sequence-length", type=int, default=72, help="Sequence length for model input")
    parser.add_argument("--cache-only", action="store_true", help="Use cached Chronos forecasts only")
    parser.add_argument("--initial-cash", type=float, default=10_000.0, help="Initial cash per symbol")
    parser.add_argument("--probe-after-loss", action="store_true", help="Enable probe trades after a losing sell")
    parser.add_argument("--probe-notional", type=float, default=1.0, help="Notional size for probe trades")
    parser.add_argument("--max-hold-hours", type=int, default=None, help="Force close positions after N hours")
    parser.add_argument(
        "--plot-dir",
        default="binanceneural/plots",
        help="Directory to write PNG trade plots (set --no-plot to disable)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable writing PNG trade plots")
    args = parser.parse_args()

    data_cfg = DatasetConfig(
        symbol=args.symbol,
        sequence_length=args.sequence_length,
        cache_only=args.cache_only,
    )
    data = BinanceHourlyDataModule(data_cfg)
    val_frame = data.val_dataset.frame.copy()
    if "symbol" not in val_frame.columns:
        val_frame["symbol"] = args.symbol

    default_cfg = TrainingConfig(sequence_length=args.sequence_length)
    model = _load_model(Path(args.checkpoint), len(data.feature_columns), default_cfg)

    actions = generate_actions_from_frame(
        model=model,
        frame=val_frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
    )
    if "symbol" not in actions.columns:
        actions["symbol"] = args.symbol

    sim = BinanceMarketSimulator(
        SimulationConfig(
            initial_cash=args.initial_cash,
            enable_probe_mode=args.probe_after_loss,
            probe_notional=args.probe_notional,
            max_hold_hours=args.max_hold_hours,
        )
    )
    result = sim.run(val_frame, actions)
    metrics = result.metrics
    print(f"Combined total_return: {metrics['total_return']:.4f}")
    print(f"Combined sortino: {metrics['sortino']:.4f}")

    if not args.no_plot:
        plot_dir = Path(args.plot_dir)
        for symbol, sym_result in result.per_symbol.items():
            sym_bars = val_frame[val_frame["symbol"].astype(str).str.upper() == symbol]
            sym_actions = actions[actions["symbol"].astype(str).str.upper() == symbol]
            output_path = plot_dir / f"{symbol.lower()}_simulation.png"
            saved = save_trade_plot(symbol, sym_bars, sym_actions, sym_result, output_path)
            print(f"Saved plot: {saved}")


if __name__ == "__main__":
    main()
