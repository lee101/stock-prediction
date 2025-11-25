from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse existing pipeline pieces
from hourlycrypto.trade_stock_crypto_hourly import (  # type: ignore
    PriceOffsetParams,
    _build_training_config,
    _ensure_forecasts,
    _infer_actions,
    _load_pretrained_policy,
)
from hourlycryptotraining import TrainingConfig  # type: ignore


def _build_actions_df(
    symbol: str,
    checkpoint_path: Path,
    sequence_length: int,
    price_offset_pct: Optional[float],
    offset_span_multiplier: float,
    offset_max_pct: Optional[float],
    cache_only: bool,
    use_cpu: bool,
) -> tuple[TrainingConfig, "pd.DataFrame", "pd.DataFrame"]:
    """Load checkpoint, run inference, and return bars/actions frames."""
    from argparse import Namespace
    import pandas as pd

    args = Namespace(
        epochs=1,
        batch_size=32,
        sequence_length=sequence_length,
        learning_rate=None,
        checkpoint_root=None,
        checkpoint_path=str(checkpoint_path),
        preload_checkpoint=None,
        force_retrain=False,
        dry_train_steps=None,
        ema_decay=None,
        no_compile=True,
        use_amp=True,
        amp_dtype="bfloat16",
        dropout=None,
        price_offset_pct=price_offset_pct,
        training_symbols=None,
        price_offset_span_multiplier=offset_span_multiplier,
        price_offset_max_pct=offset_max_pct,
        symbol=symbol,
        cache_only_forecasts=cache_only,
    )
    config = _build_training_config(args)
    if use_cpu:
        config.device = "cpu"

    _ensure_forecasts(config, cache_only=cache_only)
    data_module, policy, _ = _load_pretrained_policy(config, price_offset_override=price_offset_pct)

    max_pct = offset_max_pct if offset_max_pct and offset_max_pct > 0 else None
    offset_params = PriceOffsetParams(
        base_pct=config.price_offset_pct,
        span_multiplier=max(0.0, offset_span_multiplier),
        max_pct=max_pct,
    )
    actions = _infer_actions(policy, data_module, config, offset_params=offset_params)

    # Align bars to action timestamps for plotting
    mask = data_module.frame["timestamp"].isin(actions["timestamp"])
    bars = data_module.frame.loc[mask, ["timestamp", "open", "high", "low", "close"]].copy()
    actions = actions.set_index("timestamp")
    bars = bars.set_index("timestamp")
    bars = bars.sort_index()
    actions = actions.sort_index()
    return config, bars, actions


def _to_mdate(ts):
    # Matplotlib dislikes tz-aware; convert to naive UTC
    if getattr(ts, "tzinfo", None):
        ts = ts.tz_convert("UTC").tz_localize(None)
    return mdates.date2num(ts.to_pydatetime())


def _plot_candles(ax, bars):
    """Lightweight candlestick drawing without extra deps."""
    width = 0.025  # in days (~36 minutes)
    for ts, row in bars.iterrows():
        x = _to_mdate(ts)
        open_, high, low, close = row["open"], row["high"], row["low"], row["close"]
        color = "#2ca02c" if close >= open_ else "#d62728"
        ax.vlines(x, low, high, color=color, linewidth=0.6)
        rect_y = min(open_, close)
        rect_h = abs(close - open_)
        if rect_h == 0:
            rect_h = 1e-6
        ax.add_patch(
            Rectangle(
                (x - width / 2, rect_y),
                width,
                rect_h,
                facecolor=color,
                edgecolor=color,
                alpha=0.6,
                linewidth=0,
            )
        )
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)


def _plot_actions(ax, bars, actions):
    times = [_to_mdate(ts) for ts in actions.index]
    ax.plot(times, actions["trade_amount"], label="trade_amount", color="#444444")
    ax.plot(times, actions["buy_amount"], label="buy_amount", color="#2ca02c", alpha=0.8)
    ax.plot(times, actions["sell_amount"], label="sell_amount", color="#d62728", alpha=0.8)

    # Highlight near-zero intents
    tiny = actions["trade_amount"] < 0.01
    if tiny.any():
        ax.scatter(
            [times[i] for i, flag in enumerate(tiny) if flag],
            actions.loc[tiny, "trade_amount"],
            color="#ff7f0e",
            s=12,
            label="trade_amount < 1%",
            zorder=5,
        )

    ax.set_ylabel("Model trade fractions (0-1)")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    ax.legend()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    # Keep single y-axis for clarity


def run(
    symbol: str,
    checkpoint_path: Path,
    output_path: Path,
    sequence_length: int,
    days: int,
    price_offset_pct: Optional[float],
    offset_span_multiplier: float,
    offset_max_pct: Optional[float],
    cache_only: bool,
    use_cpu: bool,
) -> None:
    import pandas as pd

    config, bars, actions = _build_actions_df(
        symbol,
        checkpoint_path,
        sequence_length,
        price_offset_pct,
        offset_span_multiplier,
        offset_max_pct,
        cache_only,
        use_cpu,
    )

    # Limit to recent window for readability
    if days and days > 0:
        cutoff = bars.index.max() - pd.Timedelta(days=days)
        bars = bars[bars.index >= cutoff]
        actions = actions.loc[bars.index.intersection(actions.index)]

    # Plot
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    _plot_candles(ax_top, bars)
    x_vals = [_to_mdate(ts) for ts in actions.index]
    ax_top.plot(x_vals, bars["close"], color="#1f77b4", alpha=0.25, label="close")
    ax_top.plot(x_vals, actions["buy_price"], color="#1f77b4", linestyle="-", linewidth=1.2, alpha=0.8, label="buy_price")
    ax_top.plot(x_vals, actions["sell_price"], color="#9467bd", linestyle="-", linewidth=1.2, alpha=0.8, label="sell_price")
    ax_top.legend(loc="upper left", fontsize=8)

    _plot_actions(ax_bottom, bars, actions)

    # Consistent tick formatting with full date
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M")
    for ax in (ax_bottom, ax_top):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    # Detect gaps > 1.5h and shade them
    gaps = []
    ts_series = bars.index.to_series().sort_values()
    deltas = ts_series.diff().dt.total_seconds().div(3600)
    gap_mask = deltas > 1.5
    for idx in ts_series[gap_mask].index:
        prev = ts_series.loc[idx - pd.Timedelta(hours=deltas.loc[idx])]
        gaps.append((prev, ts_series.loc[idx]))
        ax_top.axvspan(_to_mdate(prev), _to_mdate(ts_series.loc[idx]), color="orange", alpha=0.12)
        ax_bottom.axvspan(_to_mdate(prev), _to_mdate(ts_series.loc[idx]), color="orange", alpha=0.12)

    tiny_count = int((actions["trade_amount"] < 0.01).sum())
    fig.suptitle(
        f"{symbol} actions (last {days}d) | tiny(<1%)={tiny_count}/{len(actions)} "
        f"| gaps>{1.5}h: {len(gaps)} | checkpoint={checkpoint_path.name} | price_offset={config.price_offset_pct:.6f}",
        fontsize=12,
    )
    fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=140)
    print(f"Saved plot to {output_path}")
    if gaps:
        print("Detected gaps (>1.5h):")
        for start, end in gaps:
            print(f"  gap from {start} to {end} ({(end-start).total_seconds()/3600:.2f}h)")
    else:
        print("No gaps > 1.5h detected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HourlyCrypto actions vs price.")
    parser.add_argument("--symbol", default="UNIUSD")
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--days", type=int, default=20, help="Plot only the most recent N days (default 20).")
    parser.add_argument("--price-offset-pct", type=float, default=None)
    parser.add_argument("--offset-span-multiplier", type=float, default=0.15)
    parser.add_argument("--offset-max-pct", type=float, default=0.003)
    parser.add_argument("--cache-only", action="store_true", help="Only use existing forecast cache (no regeneration).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference to avoid GPU contention.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/hourlycrypto_actions.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        symbol=args.symbol.upper(),
        checkpoint_path=args.checkpoint_path,
        output_path=args.output,
        sequence_length=args.sequence_length,
        days=args.days,
        price_offset_pct=args.price_offset_pct,
        offset_span_multiplier=args.offset_span_multiplier,
        offset_max_pct=args.offset_max_pct,
        cache_only=args.cache_only,
        use_cpu=args.cpu,
    )
