from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from hourlycryptomarketsimulator import HourlyCryptoMarketSimulator, SimulationConfig
from hourlycryptotraining import HourlyCryptoDataModule, PolicyHeadConfig, TrainingConfig
from hourlycryptotraining.checkpoints import load_checkpoint
from hourlycryptotraining.model import HourlyCryptoPolicy
from hourlycrypto.trade_stock_crypto_hourly import PriceOffsetParams


def _apply_checkpoint_config(config: TrainingConfig, payload_cfg: dict) -> TrainingConfig:
    """Overlay key hyperparameters from a checkpoint payload onto a fresh TrainingConfig."""
    overrides: Tuple[str, ...] = (
        "price_offset_pct",
        "min_price_gap_pct",
        "max_trade_qty",
        "transformer_dim",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
        "sequence_length",
    )
    for key in overrides:
        if key in payload_cfg:
            setattr(config, key, payload_cfg[key])
    return config


def load_policy_and_data(
    checkpoint_path: Path,
    symbol: str,
    *,
    sequence_length: int,
    refresh_hours: int = 0,
    price_offset_override: float | None = None,
) -> Tuple[TrainingConfig, HourlyCryptoDataModule, HourlyCryptoPolicy]:
    """Load policy + datamodule for offline simulation/visualisation."""
    payload = load_checkpoint(checkpoint_path)
    payload_cfg = payload.get("config") or {}
    feature_columns: Iterable[str] = payload.get("feature_columns") or ()

    config = _apply_checkpoint_config(TrainingConfig(), payload_cfg)
    config.sequence_length = sequence_length
    if price_offset_override is not None:
        config.price_offset_pct = float(price_offset_override)

    dataset_cfg = replace(
        config.dataset,
        symbol=symbol.upper(),
        sequence_length=sequence_length,
        refresh_hours=refresh_hours,
        feature_columns=tuple(feature_columns) if feature_columns else config.dataset.feature_columns,
    )
    config.dataset = dataset_cfg

    data_module = HourlyCryptoDataModule(dataset_cfg)
    data_module.normalizer = payload["normalizer"]

    state_dict = payload["state_dict"]
    max_len = 2048
    if "pos_encoding.pe" in state_dict:
        max_len = int(state_dict["pos_encoding.pe"].shape[0])

    policy = HourlyCryptoPolicy(
        PolicyHeadConfig(
            input_dim=len(feature_columns) if feature_columns else len(data_module.feature_columns),
            hidden_dim=config.transformer_dim,
            dropout=config.transformer_dropout,
            price_offset_pct=config.price_offset_pct,
            max_trade_qty=config.max_trade_qty,
            min_price_gap_pct=config.min_price_gap_pct,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            max_len=max_len,
        )
    )
    upgraded_state = HourlyCryptoPolicy.upgrade_legacy_state_dict(dict(state_dict))
    policy.load_state_dict(upgraded_state, strict=False)
    return config, data_module, policy


def _slice_frame(frame: pd.DataFrame, window_hours: int, seq_len: int) -> pd.DataFrame:
    """Keep only the tail needed for the requested window plus context."""
    if window_hours <= 0:
        return frame.reset_index(drop=True)
    cutoff = frame["timestamp"].max() - pd.Timedelta(hours=window_hours + seq_len + 12)
    trimmed = frame[frame["timestamp"] >= cutoff].reset_index(drop=True)
    if len(trimmed) < seq_len + 1:
        raise ValueError("Not enough history after trimming; reduce window or sequence length.")
    return trimmed


def infer_actions(
    policy: HourlyCryptoPolicy,
    *,
    frame: pd.DataFrame,
    normalizer,
    feature_columns: Iterable[str],
    sequence_length: int,
    price_offset_pct: float,
    offset_params: PriceOffsetParams | None = None,
) -> pd.DataFrame:
    """Run autoregressive decoding to produce hourly trade intents."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device).eval()

    features = frame[list(feature_columns)].to_numpy(dtype="float32")
    norm = normalizer.transform(features)
    closes = frame["close"].to_numpy(dtype="float32")
    ref_close = frame["reference_close"].to_numpy(dtype="float32")
    chronos_high = frame["chronos_high"].to_numpy(dtype="float32")
    chronos_low = frame["chronos_low"].to_numpy(dtype="float32")
    timestamps = frame["timestamp"].to_numpy()

    rows = []
    with torch.no_grad():
        for idx in range(sequence_length, len(frame) + 1):
            window = slice(idx - sequence_length, idx)
            feat = torch.from_numpy(norm[window]).unsqueeze(0).to(device)
            ref_tensor = torch.from_numpy(ref_close[window]).unsqueeze(0).to(device)
            high_tensor = torch.from_numpy(chronos_high[window]).unsqueeze(0).to(device)
            low_tensor = torch.from_numpy(chronos_low[window]).unsqueeze(0).to(device)
            outputs = policy(feat)
            offset_tensor = (
                offset_params.build_tensor(ref_tensor, high_tensor, low_tensor) if offset_params else None
            )
            decoded = policy.decode_actions(
                outputs,
                reference_close=ref_tensor,
                chronos_high=high_tensor,
                chronos_low=low_tensor,
                dynamic_offset_pct=offset_tensor,
            )
            ts = pd.Timestamp(timestamps[idx - 1])
            if offset_tensor is None:
                buy_off = sell_off = float(price_offset_pct)
            else:
                buy_off = float(offset_tensor[0][0, -1].item())
                sell_off = float(offset_tensor[1][0, -1].item())
            rows.append(
                {
                    "timestamp": ts,
                    "buy_price": float(decoded["buy_price"][0, -1].item()),
                    "sell_price": float(decoded["sell_price"][0, -1].item()),
                    "trade_amount": float(decoded["trade_amount"][0, -1].item()),
                    "buy_amount": float(decoded["buy_amount"][0, -1].item()),
                    "sell_amount": float(decoded["sell_amount"][0, -1].item()),
                    "buy_offset_pct": buy_off,
                    "sell_offset_pct": sell_off,
                    "reference_close": float(ref_tensor[0, -1].item()),
                    "chronos_high": float(high_tensor[0, -1].item()),
                    "chronos_low": float(low_tensor[0, -1].item()),
                }
            )
    return pd.DataFrame(rows)


def plot_simulation(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    result,
    *,
    symbol: str,
    checkpoint_label: str,
    output_path: Path,
) -> None:
    """Create a 3-panel PNG summarising price path, actions, and equity."""
    sns.set_theme(style="whitegrid")
    palette = {"buy": "#1f77b4", "sell": "#d62728", "close": "#111"}
    fig, (ax_price, ax_amt, ax_equity) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.1, 1.4]}
    )

    ax_price.fill_between(bars["timestamp"], bars["low"], bars["high"], color="#e5e5e5", alpha=0.4, label="High/Low")
    ax_price.plot(bars["timestamp"], bars["close"], color=palette["close"], linewidth=1.4, label="Close")
    ax_price.plot(actions["timestamp"], actions["buy_price"], color=palette["buy"], linewidth=1.2, label="Pred buy")
    ax_price.plot(actions["timestamp"], actions["sell_price"], color=palette["sell"], linewidth=1.2, label="Pred sell")

    buys = [(t.timestamp, t.price) for t in result.trades if t.side == "buy"]
    sells = [(t.timestamp, t.price) for t in result.trades if t.side == "sell"]
    if buys:
        ax_price.scatter(*zip(*buys), color=palette["buy"], s=28, marker="^", label="Filled buy", zorder=5)
    if sells:
        ax_price.scatter(*zip(*sells), color=palette["sell"], s=28, marker="v", label="Filled sell", zorder=5)

    ax_price.set_ylabel("Price (USD)")
    ax_price.legend(ncol=3, fontsize=9)

    width = 0.015
    ax_amt.bar(actions["timestamp"], actions["buy_amount"], width=width, color=palette["buy"], alpha=0.7, label="buy_amount")
    ax_amt.bar(actions["timestamp"], -actions["sell_amount"], width=width, color=palette["sell"], alpha=0.7, label="sell_amount")
    ax_amt.axhline(0, color="#555", linewidth=0.9)
    ax_amt.set_ylabel("Trade Intensity")
    ax_amt.legend(ncol=2, fontsize=9)

    equity_idx = pd.to_datetime(result.equity_curve.index)
    ax_equity.plot(equity_idx, result.equity_curve.values, color="#2ca02c", linewidth=1.4, label="Equity")
    inv_axis = ax_equity.twinx()
    inv_axis.plot(result.per_hour["timestamp"], result.per_hour["inventory"], color="#ff7f0e", alpha=0.6, linewidth=1.0, label="Inventory")
    ax_equity.set_ylabel("Portfolio Value")
    inv_axis.set_ylabel("Inventory")

    # Combined legend for equity/inventory
    handles1, labels1 = ax_equity.get_legend_handles_labels()
    handles2, labels2 = inv_axis.get_legend_handles_labels()
    ax_equity.legend(handles1 + handles2, labels1 + labels2, fontsize=9, loc="upper left")

    ret_pct = result.metrics.get("total_return", 0.0) * 100
    sortino = result.metrics.get("sortino", 0.0)
    title = f"{symbol.upper()} | {checkpoint_label} | {ret_pct:+.2f}% return | sortino {sortino:.2f}"
    fig.suptitle(title, fontsize=14, y=0.97)

    ax_equity.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax_equity.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_equity.xaxis.get_major_locator()))
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_simulation_to_png(
    checkpoint_path: Path,
    symbol: str,
    *,
    sequence_length: int = 256,
    window_hours: int = 24 * 8,
    refresh_hours: int = 0,
    price_offset_pct: float | None = None,
    output_dir: Path = Path("plots"),
) -> Path:
    config, data_module, policy = load_policy_and_data(
        checkpoint_path,
        symbol,
        sequence_length=sequence_length,
        refresh_hours=refresh_hours,
        price_offset_override=price_offset_pct,
    )
    frame = _slice_frame(data_module.frame, window_hours, sequence_length)
    actions = infer_actions(
        policy,
        frame=frame,
        normalizer=data_module.normalizer,
        feature_columns=data_module.feature_columns,
        sequence_length=sequence_length,
        price_offset_pct=config.price_offset_pct,
        offset_params=None,
    )
    bars = frame.loc[frame["timestamp"].isin(actions["timestamp"]), ["timestamp", "high", "low", "close"]]
    if window_hours:
        cutoff = bars["timestamp"].max() - pd.Timedelta(hours=window_hours)
        bars = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)
        actions = actions[actions["timestamp"] >= cutoff].reset_index(drop=True)

    simulator = HourlyCryptoMarketSimulator(SimulationConfig(symbol=symbol.upper()))
    result = simulator.run(bars, actions)

    ckpt_label = f"{checkpoint_path.parent.name}/{checkpoint_path.stem}"
    output_path = output_dir / f"{symbol.lower()}_{checkpoint_path.parent.name}_{checkpoint_path.stem}_{window_hours}h.png"
    plot_simulation(bars, actions, result, symbol=symbol, checkpoint_label=ckpt_label, output_path=output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PNG visualisations for HourlyCrypto simulations.")
    parser.add_argument("--checkpoint-path", required=True, type=Path, help="Path to .pt checkpoint")
    parser.add_argument("--symbol", required=True, type=str, help="Trading symbol, e.g., UNIUSD")
    parser.add_argument("--sequence-length", type=int, default=256, help="Sequence length used by the model")
    parser.add_argument("--window-hours", type=int, default=24 * 8, help="Lookback window to plot (hours)")
    parser.add_argument("--refresh-hours", type=int, default=0, help="Price refresh window (0 to skip downloads)")
    parser.add_argument("--price-offset-pct", type=float, default=None, help="Override price offset pct")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"), help="Where to write PNGs")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = run_simulation_to_png(
        args.checkpoint_path,
        args.symbol,
        sequence_length=args.sequence_length,
        window_hours=args.window_hours,
        refresh_hours=args.refresh_hours,
        price_offset_pct=args.price_offset_pct,
        output_dir=args.output_dir,
    )
    print(f"Saved plot -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
