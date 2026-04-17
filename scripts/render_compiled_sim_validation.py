#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from differentiable_loss_utils import HOURLY_PERIODS_PER_YEAR, compute_loss_by_type
from src.marketsim_video import render_html_plotly, render_mp4
from trainingefficiency.compiled_sim_loss import CompiledSimTrajectory, compiled_sim_trajectory
from trainingefficiency.compiled_sim_visual import trajectory_to_marketsim_trace, write_comparison_html
from trainingefficiency.fast_differentiable_sim import simulate_hourly_trades_fast


def _build_scenario(*, batch_size: int, steps: int, device: torch.device, dtype: torch.dtype, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    t = torch.linspace(0.0, 1.0, steps=steps, device=device, dtype=torch.float32)
    phase = torch.linspace(0.0, 1.2, steps=batch_size, device=device, dtype=torch.float32).unsqueeze(-1)
    trend = 100.0 + 6.0 * t.unsqueeze(0) + 1.8 * torch.sin(2.0 * torch.pi * (3.0 * t.unsqueeze(0) + phase))
    shock = 0.7 * torch.sin(2.0 * torch.pi * (11.0 * t.unsqueeze(0) + 0.5 * phase))
    closes = trend + shock
    highs = closes + 0.6 + 0.15 * (1.0 + torch.sin(2.0 * torch.pi * (7.0 * t.unsqueeze(0) + phase)))
    lows = closes - 0.6 - 0.15 * (1.0 + torch.cos(2.0 * torch.pi * (5.0 * t.unsqueeze(0) + phase)))
    buy_prices = closes * (1.0 - 0.0012 - 0.0007 * torch.sigmoid(2.5 * torch.sin(2.0 * torch.pi * (4.0 * t.unsqueeze(0) + phase))))
    sell_prices = closes * (1.0 + 0.0012 + 0.0007 * torch.sigmoid(2.5 * torch.cos(2.0 * torch.pi * (4.0 * t.unsqueeze(0) + phase))))
    buy_frac = 0.35 * torch.sigmoid(2.8 * torch.sin(2.0 * torch.pi * (2.0 * t.unsqueeze(0) + phase)) - 0.4)
    sell_frac = 0.35 * torch.sigmoid(2.8 * torch.cos(2.0 * torch.pi * (2.0 * t.unsqueeze(0) + phase)) - 0.4)
    max_leverage = torch.full_like(closes, 1.5)
    can_short = torch.ones(batch_size, device=device, dtype=torch.float32)
    can_long = torch.ones(batch_size, device=device, dtype=torch.float32)
    return {
        "closes": closes.to(dtype=dtype),
        "highs": highs.to(dtype=dtype),
        "lows": lows.to(dtype=dtype),
        "buy_prices": buy_prices.to(dtype=dtype),
        "sell_prices": sell_prices.to(dtype=dtype),
        "buy_frac": buy_frac.to(dtype=dtype),
        "sell_frac": sell_frac.to(dtype=dtype),
        "max_leverage": max_leverage.to(dtype=dtype),
        "can_short": can_short,
        "can_long": can_long,
    }


def _baseline_trajectory(
    *,
    highs: torch.Tensor,
    lows: torch.Tensor,
    closes: torch.Tensor,
    buy_prices: torch.Tensor,
    sell_prices: torch.Tensor,
    buy_frac: torch.Tensor,
    sell_frac: torch.Tensor,
    max_leverage: torch.Tensor,
    can_short: torch.Tensor,
    can_long: torch.Tensor,
    initial_cash: float,
    initial_inventory: float,
    maker_fee: float,
    temperature: float,
    fill_buffer_pct: float,
    margin_annual_rate: float,
    periods_per_year: float,
    return_weight: float,
    decision_lag_bars: int,
) -> CompiledSimTrajectory:
    sim = simulate_hourly_trades_fast(
        highs=highs,
        lows=lows,
        closes=closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        trade_intensity=buy_frac,
        buy_trade_intensity=buy_frac,
        sell_trade_intensity=sell_frac,
        maker_fee=maker_fee,
        initial_cash=initial_cash,
        initial_inventory=initial_inventory,
        temperature=temperature,
        max_leverage=max_leverage,
        can_short=can_short,
        can_long=can_long,
        decision_lag_bars=decision_lag_bars,
        fill_buffer_pct=fill_buffer_pct,
        margin_annual_rate=margin_annual_rate,
    )
    if decision_lag_bars > 0:
        lag = decision_lag_bars
        highs = highs[..., lag:]
        lows = lows[..., lag:]
        closes = closes[..., lag:]
        buy_prices = buy_prices[..., :-lag]
        sell_prices = sell_prices[..., :-lag]
    loss, score, sortino, annual_return = compute_loss_by_type(
        sim.returns,
        "sortino",
        periods_per_year=periods_per_year,
        return_weight=return_weight,
    )
    return CompiledSimTrajectory(
        returns=sim.returns,
        portfolio_values=sim.portfolio_values,
        cash_path=sim.portfolio_values - sim.inventory_path * closes,
        inventory_path=sim.inventory_path,
        executed_buys=sim.executed_buys,
        executed_sells=sim.executed_sells,
        buy_fill_probability=sim.buy_fill_probability,
        sell_fill_probability=sim.sell_fill_probability,
        closes=closes,
        highs=highs,
        lows=lows,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        loss=loss,
        score=score,
        sortino=sortino,
        annual_return=annual_return,
    )


def _render_trace_bundle(
    *,
    trajectory: CompiledSimTrajectory,
    out_prefix: Path,
    title: str,
    sample_index: int,
    fps: int,
    frames_per_bar: int,
    skip_video: bool,
) -> None:
    trace = trajectory_to_marketsim_trace(trajectory, sample_index=sample_index, symbol=out_prefix.stem.upper())
    trace.to_json(out_prefix.with_suffix(".json"))
    render_html_plotly(trace, out_prefix.with_suffix(".html"), num_pairs=1, title=title)
    if skip_video:
        return
    render_mp4(
        trace,
        out_prefix.with_suffix(".mp4"),
        num_pairs=1,
        fps=fps,
        frames_per_bar=frames_per_bar,
        title=title,
        periods_per_year=HOURLY_PERIODS_PER_YEAR,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render visual parity artifacts for compiled/eager fused sim validation.")
    parser.add_argument("--outdir", default="models/artifacts/compiled_sim_validation")
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--decision-lag-bars", type=int, default=2)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0625)
    parser.add_argument("--return-weight", type=float, default=0.08)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frames-per-bar", type=int, default=1)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenario = _build_scenario(
        batch_size=int(args.batch_size),
        steps=int(args.steps),
        device=device,
        dtype=dtype,
        seed=int(args.seed),
    )

    baseline = _baseline_trajectory(
        **scenario,
        initial_cash=1.0,
        initial_inventory=0.0,
        maker_fee=float(args.maker_fee),
        temperature=float(args.temperature),
        fill_buffer_pct=float(args.fill_buffer_pct),
        margin_annual_rate=float(args.margin_annual_rate),
        periods_per_year=HOURLY_PERIODS_PER_YEAR,
        return_weight=float(args.return_weight),
        decision_lag_bars=int(args.decision_lag_bars),
    )
    fused = compiled_sim_trajectory(
        **scenario,
        initial_cash=1.0,
        initial_inventory=0.0,
        maker_fee=float(args.maker_fee),
        temperature=float(args.temperature),
        fill_buffer_pct=float(args.fill_buffer_pct),
        margin_annual_rate=float(args.margin_annual_rate),
        periods_per_year=HOURLY_PERIODS_PER_YEAR,
        return_weight=float(args.return_weight),
        decision_lag_bars=int(args.decision_lag_bars),
    )

    summary = {
        "sample_index": int(args.sample_index),
        "steps_after_lag": int(baseline.returns.shape[-1]),
        "max_abs_return_diff": float((baseline.returns - fused.returns).abs().max().item()),
        "max_abs_value_diff": float((baseline.portfolio_values - fused.portfolio_values).abs().max().item()),
        "max_abs_inventory_diff": float((baseline.inventory_path - fused.inventory_path).abs().max().item()),
        "max_abs_buy_fill_prob_diff": float((baseline.buy_fill_probability - fused.buy_fill_probability).abs().max().item()),
        "max_abs_sell_fill_prob_diff": float((baseline.sell_fill_probability - fused.sell_fill_probability).abs().max().item()),
        "loss_diff": float((baseline.loss - fused.loss).abs().item()),
        "score_diff_max": float((baseline.score - fused.score).abs().max().item()),
        "sortino_diff_max": float((baseline.sortino - fused.sortino).abs().max().item()),
        "annual_return_diff_max": float((baseline.annual_return - fused.annual_return).abs().max().item()),
        "device": str(device),
        "dtype": args.dtype,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    title = (
        f"compiled sim validation · steps={args.steps} lag={args.decision_lag_bars} "
        f"return_diff={summary['max_abs_return_diff']:.3e}"
    )
    write_comparison_html(
        out_path=outdir / "comparison.html",
        baseline=baseline,
        fused=fused,
        sample_index=int(args.sample_index),
        title=title,
    )
    _render_trace_bundle(
        trajectory=baseline,
        out_prefix=outdir / "baseline",
        title=f"{title} · baseline",
        sample_index=int(args.sample_index),
        fps=int(args.fps),
        frames_per_bar=int(args.frames_per_bar),
        skip_video=bool(args.skip_video),
    )
    _render_trace_bundle(
        trajectory=fused,
        out_prefix=outdir / "fused",
        title=f"{title} · fused",
        sample_index=int(args.sample_index),
        fps=int(args.fps),
        frames_per_bar=int(args.frames_per_bar),
        skip_video=bool(args.skip_video),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
