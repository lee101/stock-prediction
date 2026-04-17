from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.marketsim_video import MarketsimTrace, OrderTick
from trainingefficiency.compiled_sim_loss import CompiledSimTrajectory


def _series_for_sample(value: torch.Tensor, sample_index: int) -> np.ndarray:
    if value.ndim == 1:
        flat = value.reshape(1, -1)
    else:
        flat = value.reshape(-1, value.shape[-1])
    if sample_index < 0 or sample_index >= flat.shape[0]:
        raise IndexError(f"sample_index={sample_index} out of range for {flat.shape[0]} samples")
    return flat[sample_index].detach().cpu().numpy().astype(np.float32, copy=False)


def _metric_for_sample(value: torch.Tensor, sample_index: int) -> float:
    flat = value.reshape(-1)
    if sample_index < 0 or sample_index >= flat.shape[0]:
        raise IndexError(f"sample_index={sample_index} out of range for {flat.shape[0]} samples")
    return float(flat[sample_index].detach().cpu().item())


def trajectory_to_marketsim_trace(
    trajectory: CompiledSimTrajectory,
    *,
    sample_index: int = 0,
    symbol: str = "SIM",
) -> MarketsimTrace:
    close = _series_for_sample(trajectory.closes, sample_index)
    high = _series_for_sample(trajectory.highs, sample_index)
    low = _series_for_sample(trajectory.lows, sample_index)
    buy_price = _series_for_sample(trajectory.buy_prices, sample_index)
    sell_price = _series_for_sample(trajectory.sell_prices, sample_index)
    inventory = _series_for_sample(trajectory.inventory_path, sample_index)
    equity = _series_for_sample(trajectory.portfolio_values, sample_index)
    exec_buys = _series_for_sample(trajectory.executed_buys, sample_index)
    exec_sells = _series_for_sample(trajectory.executed_sells, sample_index)

    opens = close.copy()
    if close.size > 1:
        opens[1:] = close[:-1]
    prices = close[:, None]
    prices_ohlc = np.stack([opens, high, low, close], axis=-1)[:, None, :]
    trace = MarketsimTrace(symbols=[symbol], prices=prices, prices_ohlc=prices_ohlc)

    eps = 1e-8
    for step in range(close.shape[0]):
        qty_buy = float(exec_buys[step])
        qty_sell = float(exec_sells[step])
        inv = float(inventory[step])
        action_id = 0
        orders: list[OrderTick] = []
        if qty_buy > eps and qty_buy >= qty_sell:
            action_id = 1
            orders.append(OrderTick(sym=0, price=float(buy_price[step]), is_short=False))
        elif qty_sell > eps:
            action_id = 2
            orders.append(OrderTick(sym=0, price=float(sell_price[step]), is_short=True))
        trace.record(
            step=step,
            action_id=action_id,
            position_sym=(0 if abs(inv) > eps else -1),
            position_is_short=(inv < -eps),
            equity=float(equity[step]),
            orders=orders,
            long_target=np.asarray([float(buy_price[step])], dtype=np.float32),
            short_target=np.asarray([float(sell_price[step])], dtype=np.float32),
        )
    return trace


def write_comparison_html(
    *,
    out_path: str | Path,
    baseline: CompiledSimTrajectory,
    fused: CompiledSimTrajectory,
    sample_index: int = 0,
    title: str = "Compiled Sim Validation",
) -> Path:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    base_close = _series_for_sample(baseline.closes, sample_index)
    base_buy = _series_for_sample(baseline.buy_prices, sample_index)
    base_sell = _series_for_sample(baseline.sell_prices, sample_index)
    base_values = _series_for_sample(baseline.portfolio_values, sample_index)
    fused_values = _series_for_sample(fused.portfolio_values, sample_index)
    base_returns = _series_for_sample(baseline.returns, sample_index)
    fused_returns = _series_for_sample(fused.returns, sample_index)
    base_inventory = _series_for_sample(baseline.inventory_path, sample_index)
    fused_inventory = _series_for_sample(fused.inventory_path, sample_index)
    base_buy_exec = _series_for_sample(baseline.executed_buys, sample_index)
    fused_buy_exec = _series_for_sample(fused.executed_buys, sample_index)
    base_sell_exec = _series_for_sample(baseline.executed_sells, sample_index)
    fused_sell_exec = _series_for_sample(fused.executed_sells, sample_index)

    x = list(range(base_close.shape[0]))
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Price and target levels",
            "Portfolio value",
            "Returns and inventory",
            "Executed quantities and absolute diffs",
        ),
    )
    fig.add_trace(go.Scatter(x=x, y=base_close, name="close", line={"color": "#222"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=base_buy, name="buy_price", line={"color": "#2ca02c", "dash": "dot"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=base_sell, name="sell_price", line={"color": "#d62728", "dash": "dot"}), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=base_values, name="baseline_value", line={"color": "#1f77b4"}), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=fused_values, name="fused_value", line={"color": "#ff7f0e"}), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.abs(base_values - fused_values),
            name="value_abs_diff",
            line={"color": "#9467bd", "dash": "dash"},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(go.Scatter(x=x, y=base_returns, name="baseline_returns", line={"color": "#1f77b4"}), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=fused_returns, name="fused_returns", line={"color": "#ff7f0e"}), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=base_inventory, name="baseline_inventory", line={"color": "#2ca02c"}), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=fused_inventory, name="fused_inventory", line={"color": "#d62728"}), row=3, col=1)

    fig.add_trace(go.Bar(x=x, y=base_buy_exec, name="baseline_buy_exec", marker_color="#2ca02c", opacity=0.55), row=4, col=1)
    fig.add_trace(go.Bar(x=x, y=fused_buy_exec, name="fused_buy_exec", marker_color="#98df8a", opacity=0.55), row=4, col=1)
    fig.add_trace(go.Bar(x=x, y=-base_sell_exec, name="baseline_sell_exec", marker_color="#d62728", opacity=0.55), row=4, col=1)
    fig.add_trace(go.Bar(x=x, y=-fused_sell_exec, name="fused_sell_exec", marker_color="#ff9896", opacity=0.55), row=4, col=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.abs(base_returns - fused_returns),
            name="returns_abs_diff",
            line={"color": "#8c564b", "dash": "dash"},
        ),
        row=4,
        col=1,
    )

    summary = {
        "sample_index": int(sample_index),
        "max_abs_return_diff": float(np.max(np.abs(base_returns - fused_returns))),
        "max_abs_value_diff": float(np.max(np.abs(base_values - fused_values))),
        "max_abs_inventory_diff": float(np.max(np.abs(base_inventory - fused_inventory))),
        "baseline_loss": _metric_for_sample(baseline.loss.reshape(1), 0),
        "fused_loss": _metric_for_sample(fused.loss.reshape(1), 0),
        "baseline_score": _metric_for_sample(baseline.score, sample_index),
        "fused_score": _metric_for_sample(fused.score, sample_index),
        "baseline_sortino": _metric_for_sample(baseline.sortino, sample_index),
        "fused_sortino": _metric_for_sample(fused.sortino, sample_index),
        "baseline_annual_return": _metric_for_sample(baseline.annual_return, sample_index),
        "fused_annual_return": _metric_for_sample(fused.annual_return, sample_index),
    }

    fig.update_layout(
        title=f"{title}<br><sup>{json.dumps(summary, sort_keys=True)}</sup>",
        barmode="overlay",
        height=1100,
        width=1400,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_xaxes(title_text="step", row=4, col=1)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return out
