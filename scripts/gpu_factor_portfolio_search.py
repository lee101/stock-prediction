#!/usr/bin/env python3
"""GPU search for first-principles factor portfolio packing.

This is a research screen for the market simulator. It does not use the
single-position RL policy. Instead it builds cross-sectional factor scores
directly from screened32 daily features, packs the top-ranked symbols into a
long-only portfolio, and runs many cardinality/risk/turnover cells on CUDA with
lag-2 binary-fill execution.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.gpu_realism_gate import _P_CLOSE, _P_LOW, _stage_windows  # noqa: E402
from pufferlib_market.hourly_replay import INITIAL_CASH, read_mktd  # noqa: E402
from xgbnew.artifacts import write_json_atomic  # noqa: E402

from scripts.gpu_portfolio_pack_screen import (  # noqa: E402
    _monthly_from_total,
    _parse_float_list,
    _parse_int_list,
    _rolling_vol,
    validate_common_pack_args,
)


FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "ma_delta_5d",
    "ma_delta_20d",
    "ma_delta_60d",
    "atr_pct_14d",
    "range_pct_1d",
    "rsi_14",
    "trend_60d",
    "drawdown_20d",
    "drawdown_60d",
    "log_volume_z20d",
    "log_volume_delta_5d",
]


@dataclass(frozen=True)
class FactorRecipe:
    name: str
    weights: tuple[float, ...]


@dataclass(frozen=True)
class FactorConfig:
    factor: str
    pack_size: int
    score_power: float
    vol_power: float
    score_gate: float
    gross_scale: float
    rebalance_every: int
    rebalance_threshold: float


def _weights(**items: float) -> tuple[float, ...]:
    values = [0.0] * len(FEATURE_NAMES)
    for name, value in items.items():
        values[FEATURE_NAMES.index(name)] = float(value)
    return tuple(values)


BASE_RECIPES: tuple[FactorRecipe, ...] = (
    FactorRecipe("ret1_mom", _weights(return_1d=1.0)),
    FactorRecipe("ret1_rev", _weights(return_1d=-1.0)),
    FactorRecipe("ret5_mom", _weights(return_5d=1.0)),
    FactorRecipe("ret5_rev", _weights(return_5d=-1.0)),
    FactorRecipe("ret20_mom", _weights(return_20d=1.0)),
    FactorRecipe("ret20_rev", _weights(return_20d=-1.0)),
    FactorRecipe("trend60_mom", _weights(trend_60d=1.0)),
    FactorRecipe("trend60_rev", _weights(trend_60d=-1.0)),
    FactorRecipe("ma5_mom", _weights(ma_delta_5d=1.0)),
    FactorRecipe("ma5_rev", _weights(ma_delta_5d=-1.0)),
    FactorRecipe("ma20_mom", _weights(ma_delta_20d=1.0)),
    FactorRecipe("ma20_rev", _weights(ma_delta_20d=-1.0)),
    FactorRecipe("ma60_mom", _weights(ma_delta_60d=1.0)),
    FactorRecipe("ma60_rev", _weights(ma_delta_60d=-1.0)),
    FactorRecipe("rsi_mom", _weights(rsi_14=1.0)),
    FactorRecipe("rsi_rev", _weights(rsi_14=-1.0)),
    FactorRecipe("dd20_breakout", _weights(drawdown_20d=1.0)),
    FactorRecipe("dd20_dip", _weights(drawdown_20d=-1.0)),
    FactorRecipe("dd60_breakout", _weights(drawdown_60d=1.0)),
    FactorRecipe("dd60_dip", _weights(drawdown_60d=-1.0)),
    FactorRecipe("vol_z_mom", _weights(log_volume_z20d=1.0)),
    FactorRecipe("vol_z_rev", _weights(log_volume_z20d=-1.0)),
    FactorRecipe("vol_delta_mom", _weights(log_volume_delta_5d=1.0)),
    FactorRecipe("vol_delta_rev", _weights(log_volume_delta_5d=-1.0)),
    FactorRecipe(
        "trend_combo",
        _weights(return_5d=0.5, return_20d=1.0, ma_delta_20d=0.75, ma_delta_60d=0.5, volatility_20d=-0.25),
    ),
    FactorRecipe(
        "fast_breakout",
        _weights(return_1d=0.35, return_5d=1.0, ma_delta_5d=0.75, log_volume_z20d=0.25, range_pct_1d=0.15),
    ),
    FactorRecipe(
        "mean_reversion",
        _weights(return_1d=-0.4, return_5d=-1.0, rsi_14=-0.5, drawdown_20d=-0.5, volatility_5d=-0.15),
    ),
    FactorRecipe(
        "low_vol_trend",
        _weights(return_20d=1.0, ma_delta_20d=0.5, trend_60d=0.5, volatility_20d=-1.0, atr_pct_14d=-0.5),
    ),
    FactorRecipe(
        "dip_in_uptrend",
        _weights(return_5d=-0.75, return_20d=0.75, ma_delta_60d=0.5, drawdown_20d=-0.5, volatility_20d=-0.25),
    ),
    FactorRecipe(
        "volume_confirmed_trend",
        _weights(return_20d=1.0, ma_delta_20d=0.5, rsi_14=0.25, log_volume_z20d=0.35, volatility_20d=-0.25),
    ),
)


def build_factor_configs(
    recipes: Sequence[FactorRecipe],
    *,
    pack_sizes: Sequence[int],
    score_powers: Sequence[float],
    vol_powers: Sequence[float],
    score_gates: Sequence[float],
    gross_scales: Sequence[float],
    rebalance_everys: Sequence[int],
    rebalance_thresholds: Sequence[float],
) -> list[FactorConfig]:
    configs: list[FactorConfig] = []
    for item in itertools.product(
            recipes,
            pack_sizes,
            score_powers,
            vol_powers,
            score_gates,
            gross_scales,
            rebalance_everys,
            rebalance_thresholds,
    ):
        recipe, pack_size, score_power, vol_power, score_gate, gross_scale, rebalance_every, rebalance_threshold = item
        configs.append(
            FactorConfig(
                factor=recipe.name,
                pack_size=int(pack_size),
                score_power=float(score_power),
                vol_power=float(vol_power),
                score_gate=float(score_gate),
                gross_scale=float(gross_scale),
                rebalance_every=int(rebalance_every),
                rebalance_threshold=float(rebalance_threshold),
            )
        )
    return configs


def _recipe_matrix(recipes: Sequence[FactorRecipe], *, device: torch.device) -> torch.Tensor:
    return torch.tensor([recipe.weights for recipe in recipes], device=device, dtype=torch.float32)


def _factor_score_bank(
    *,
    features: torch.Tensor,
    tradable: torch.Tensor,
    recipe_matrix: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Return cross-sectionally z-scored factor scores [R, N, S]."""
    t_obs = max(0, int(step) - 1)
    raw = torch.einsum("nsf,rf->rns", features[:, t_obs, :, : recipe_matrix.shape[1]], recipe_matrix)
    mask = tradable[:, step, :].unsqueeze(0)
    raw_masked = torch.where(mask, raw, torch.zeros_like(raw))
    count = mask.sum(dim=2, keepdim=True).clamp_min(1)
    mean = raw_masked.sum(dim=2, keepdim=True) / count
    centered = torch.where(mask, raw - mean, torch.zeros_like(raw))
    var = (centered * centered).sum(dim=2, keepdim=True) / count
    z = centered / torch.sqrt(var.clamp_min(1e-8))
    return torch.where(mask, z, torch.zeros_like(z))


def _target_weights_from_factor_scores(
    *,
    factor_scores: torch.Tensor,
    factor_ids: torch.Tensor,
    tradable: torch.Tensor,
    vol: torch.Tensor,
    configs: Sequence[FactorConfig],
) -> torch.Tensor:
    device = factor_scores.device
    c = len(configs)
    n_windows, n_symbols = factor_scores.shape[1], factor_scores.shape[2]
    pack_sizes = torch.tensor([cfg.pack_size for cfg in configs], device=device, dtype=torch.long)
    score_power = torch.tensor([cfg.score_power for cfg in configs], device=device, dtype=torch.float32)
    vol_power = torch.tensor([cfg.vol_power for cfg in configs], device=device, dtype=torch.float32)
    score_gate = torch.tensor([cfg.score_gate for cfg in configs], device=device, dtype=torch.float32)
    gross_scale = torch.tensor([cfg.gross_scale for cfg in configs], device=device, dtype=torch.float32)

    score = factor_scores.index_select(0, factor_ids)
    score = torch.clamp(score - score_gate.view(c, 1, 1), min=0.0)
    risk = torch.pow(vol.clamp_min(1e-4).unsqueeze(0), vol_power.view(c, 1, 1))
    score = score / risk.clamp_min(1e-6)
    score = torch.where(tradable.unsqueeze(0), score, torch.zeros_like(score))

    max_k = int(min(max(int(cfg.pack_size) for cfg in configs), int(n_symbols)))
    top_vals, top_idx = torch.topk(score, k=max_k, dim=2)
    selected = (torch.arange(max_k, device=device).view(1, 1, max_k) < pack_sizes.clamp_max(max_k).view(c, 1, 1)) & (
        top_vals > 0.0
    )
    raw_vals = torch.where(
        score_power.view(c, 1, 1) <= 0.0,
        torch.ones_like(top_vals),
        torch.pow(top_vals.clamp_min(1e-12), score_power.view(c, 1, 1)),
    )
    raw_vals = torch.where(selected, raw_vals, torch.zeros_like(raw_vals))
    weights_top = raw_vals / raw_vals.sum(dim=2, keepdim=True).clamp_min(1e-12)
    weights = torch.zeros(c, n_windows, n_symbols, device=device, dtype=torch.float32)
    weights.scatter_add_(2, top_idx, weights_top)
    return weights * gross_scale.view(c, 1, 1)


def run_factor_sim(
    *,
    prices: torch.Tensor,
    features: torch.Tensor,
    tradable: torch.Tensor,
    recipes: Sequence[FactorRecipe],
    configs: Sequence[FactorConfig],
    window_days: int,
    fill_buffer_bps: float,
    max_leverage: float,
    fee_rate: float,
    slippage_bps: float,
    margin_apr: float,
    vol_lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = prices.device
    recipe_id_by_name = {recipe.name: i for i, recipe in enumerate(recipes)}
    factor_ids = torch.tensor([recipe_id_by_name[cfg.factor] for cfg in configs], device=device, dtype=torch.long)
    recipe_matrix = _recipe_matrix(recipes, device=device)
    c = len(configs)
    n = int(prices.shape[0])
    s = int(prices.shape[2])
    init_cash = float(INITIAL_CASH)
    cash = torch.full((c, n), init_cash, device=device, dtype=torch.float32)
    qty = torch.zeros(c, n, s, device=device, dtype=torch.float32)
    peak_equity = torch.full((c, n), init_cash, device=device, dtype=torch.float32)
    max_dd = torch.zeros(c, n, device=device, dtype=torch.float32)
    turnover = torch.zeros(c, n, device=device, dtype=torch.float32)
    target_buf = torch.zeros(c, n, s, 3, device=device, dtype=torch.float32)
    desired_target = torch.zeros(c, n, s, device=device, dtype=torch.float32)
    rebalance_every = torch.tensor(
        [max(1, int(cfg.rebalance_every)) for cfg in configs],
        device=device,
        dtype=torch.long,
    ).view(c, 1, 1)
    rebalance_threshold = torch.tensor(
        [cfg.rebalance_threshold for cfg in configs],
        device=device,
        dtype=torch.float32,
    ).view(c, 1, 1)
    fill_buffer_frac = max(0.0, float(fill_buffer_bps)) / 10_000.0
    effective_fee = float(fee_rate) + max(0.0, float(slippage_bps)) / 10_000.0
    margin_rate = max(0.0, float(margin_apr)) / 365.0

    for step in range(int(window_days)):
        factor_scores = _factor_score_bank(
            features=features,
            tradable=tradable,
            recipe_matrix=recipe_matrix,
            step=step,
        )
        vol = _rolling_vol(prices, int(step), lookback=int(vol_lookback))
        target_now = _target_weights_from_factor_scores(
            factor_scores=factor_scores,
            factor_ids=factor_ids,
            tradable=tradable[:, step, :],
            vol=vol,
            configs=configs,
        )
        should_rebalance = (torch.tensor(step, device=device) % rebalance_every) == 0
        target_now = torch.where(should_rebalance, target_now, desired_target)
        desired_target = target_now
        target_buf[:, :, :, step % 3] = target_now
        if step < 2:
            target = torch.zeros(c, n, s, device=device, dtype=torch.float32)
        else:
            target = target_buf[:, :, :, (step - 2) % 3]

        close_t = prices[:, step, :, _P_CLOSE].unsqueeze(0).expand(c, n, s)
        low_t = prices[:, step, :, _P_LOW].unsqueeze(0).expand(c, n, s)
        trad_t = tradable[:, step, :].unsqueeze(0).expand(c, n, s)
        current_value = qty * close_t
        equity = cash + current_value.sum(dim=2)
        desired_value = equity.clamp_min(0.0).unsqueeze(2) * float(max_leverage) * target

        sell_value = torch.clamp(current_value - desired_value, min=0.0)
        sell_mask = (sell_value > current_value * rebalance_threshold) & trad_t & (close_t > 0.0)
        sell_qty = torch.where(sell_mask, sell_value / close_t.clamp_min(1e-12), torch.zeros_like(sell_value))
        sell_qty = torch.minimum(sell_qty, qty)
        qty = qty - sell_qty
        cash = cash + (sell_qty * close_t * (1.0 - effective_fee)).sum(dim=2)
        turnover = turnover + (sell_qty * close_t).sum(dim=2) / init_cash

        current_value = qty * close_t
        buy_value = torch.clamp(desired_value - current_value, min=0.0)
        buy_mask = buy_value > desired_value * rebalance_threshold
        fillable = low_t <= close_t * (1.0 - fill_buffer_frac)
        can_buy = buy_mask & fillable & trad_t & (close_t > 0.0) & (equity.unsqueeze(2) > 0.0)
        buy_qty = torch.where(
            can_buy,
            buy_value / (close_t * (1.0 + effective_fee)).clamp_min(1e-12),
            torch.zeros_like(buy_value),
        )
        qty = qty + buy_qty
        cash = cash - (buy_qty * close_t * (1.0 + effective_fee)).sum(dim=2)
        turnover = turnover + (buy_qty * close_t).sum(dim=2) / init_cash

        if margin_rate > 0.0:
            cash = cash - torch.clamp(-cash, min=0.0) * margin_rate

        close_new = prices[:, min(step + 1, int(prices.size(1)) - 1), :, _P_CLOSE].unsqueeze(0).expand(c, n, s)
        equity_after = cash + (qty * close_new).sum(dim=2)
        peak_equity = torch.maximum(peak_equity, equity_after)
        dd = torch.where(
            peak_equity > 0,
            (peak_equity - equity_after) / peak_equity.clamp_min(1e-12),
            torch.zeros_like(peak_equity),
        )
        max_dd = torch.maximum(max_dd, dd)

    close_end = prices[:, -1, :, _P_CLOSE].unsqueeze(0).expand(c, n, s)
    cash = cash + (qty * close_end * (1.0 - effective_fee)).sum(dim=2)
    total_return = (cash / init_cash) - 1.0
    return (
        total_return.detach().to(torch.float64).cpu().numpy(),
        max_dd.detach().to(torch.float64).cpu().numpy(),
        turnover.detach().to(torch.float64).cpu().numpy(),
    )


def _summarize(
    configs: Sequence[FactorConfig],
    total_returns: np.ndarray,
    max_drawdowns: np.ndarray,
    turnover: np.ndarray,
    *,
    window_days: int,
    neg_penalty: float,
    dd_penalty: float,
    turnover_penalty: float,
) -> list[dict]:
    med_total = np.percentile(total_returns, 50, axis=1)
    p10_total = np.percentile(total_returns, 10, axis=1)
    p90_total = np.percentile(total_returns, 90, axis=1)
    max_dd = np.max(max_drawdowns, axis=1)
    med_dd = np.percentile(max_drawdowns, 50, axis=1)
    med_turnover = np.percentile(turnover, 50, axis=1)
    n_neg = np.sum(total_returns < 0.0, axis=1).astype(int)
    out: list[dict] = []
    for i, cfg in enumerate(configs):
        med_monthly = _monthly_from_total(float(med_total[i]), int(window_days))
        p10_monthly = _monthly_from_total(float(p10_total[i]), int(window_days))
        score = (
            float(med_monthly)
            + 0.5 * float(p10_monthly)
            - float(neg_penalty) * float(n_neg[i])
            - float(dd_penalty) * float(max_dd[i])
            - float(turnover_penalty) * float(med_turnover[i])
        )
        out.append(
            {
                "config": asdict(cfg),
                "score": float(score),
                "median_total_return": float(med_total[i]),
                "p10_total_return": float(p10_total[i]),
                "p90_total_return": float(p90_total[i]),
                "median_monthly_return": float(med_monthly),
                "p10_monthly_return": float(p10_monthly),
                "max_drawdown": float(max_dd[i]),
                "median_drawdown": float(med_dd[i]),
                "median_turnover_x_initial": float(med_turnover[i]),
                "n_neg": int(n_neg[i]),
                "n_windows": int(total_returns.shape[1]),
            }
        )
    out.sort(key=lambda item: float(item["score"]), reverse=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--window-days", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--pack-sizes", default="1,2,3,4,6,8")
    parser.add_argument("--score-powers", default="0,1")
    parser.add_argument("--vol-powers", default="0,1")
    parser.add_argument("--score-gates", default="0,0.5")
    parser.add_argument("--gross-scales", default="0.25,0.5,1.0")
    parser.add_argument("--rebalance-everys", default="1,5,10")
    parser.add_argument("--rebalance-thresholds", default="0,0.10")
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--margin-apr", type=float, default=0.0625)
    parser.add_argument("--neg-penalty", type=float, default=0.002)
    parser.add_argument("--dd-penalty", type=float, default=0.15)
    parser.add_argument("--turnover-penalty", type=float, default=0.00005)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="analysis/screened32_factor_portfolio/search.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_failures = validate_common_pack_args(args, gate_attr="score_gates")
    if validation_failures:
        for failure in validation_failures:
            print(f"gpu_factor_portfolio_search: {failure}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("gpu_factor_portfolio_search: CUDA is unavailable", file=sys.stderr)
        return 2
    recipes = list(BASE_RECIPES)
    configs = build_factor_configs(
        recipes,
        pack_sizes=_parse_int_list(args.pack_sizes),
        score_powers=_parse_float_list(args.score_powers),
        vol_powers=_parse_float_list(args.vol_powers),
        score_gates=_parse_float_list(args.score_gates),
        gross_scales=_parse_float_list(args.gross_scales),
        rebalance_everys=_parse_int_list(args.rebalance_everys),
        rebalance_thresholds=_parse_float_list(args.rebalance_thresholds),
    )
    val_path = Path(args.val_data)
    if not val_path.is_absolute():
        val_path = REPO / val_path
    data = read_mktd(val_path)
    window_len = int(args.window_days) + 1
    if window_len > int(data.num_timesteps):
        print("gpu_factor_portfolio_search: window is longer than val data", file=sys.stderr)
        return 2
    starts = list(range(int(data.num_timesteps) - window_len + 1))
    if args.max_windows is not None:
        starts = starts[: max(1, int(args.max_windows))]
    device = torch.device(str(args.device))
    prices, features, tradable = _stage_windows(data, starts, int(args.window_days), device)
    print(
        f"[factor-pack] recipes={len(recipes)} configs={len(configs)} windows={len(starts)} "
        f"slip={float(args.slippage_bps):g}bps lev={float(args.leverage):g}x"
    )
    total_returns, max_drawdowns, turnover = run_factor_sim(
        prices=prices,
        features=features,
        tradable=tradable,
        recipes=recipes,
        configs=configs,
        window_days=int(args.window_days),
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        margin_apr=float(args.margin_apr),
        vol_lookback=int(args.vol_lookback),
    )
    results = _summarize(
        configs,
        total_returns,
        max_drawdowns,
        turnover,
        window_days=int(args.window_days),
        neg_penalty=float(args.neg_penalty),
        dd_penalty=float(args.dd_penalty),
        turnover_penalty=float(args.turnover_penalty),
    )
    payload = {
        "val_data": str(val_path),
        "window_days": int(args.window_days),
        "n_windows": len(starts),
        "starts": starts,
        "feature_names": FEATURE_NAMES,
        "recipes": [{"name": r.name, "weights": list(r.weights)} for r in recipes],
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "slippage_bps": float(args.slippage_bps),
        "leverage": float(args.leverage),
        "fee_rate": float(args.fee_rate),
        "margin_apr": float(args.margin_apr),
        "decision_lag": 2,
        "vol_lookback": int(args.vol_lookback),
        "results": results,
        "best": results[0] if results else None,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    write_json_atomic(out_path, payload)
    print("\nTop factor-pack configs:")
    for item in results[: max(1, int(args.top_k))]:
        cfg = item["config"]
        print(
            f"{item['score']:+.4f} med={item['median_monthly_return'] * 100:+6.2f}% "
            f"p10={item['p10_monthly_return'] * 100:+6.2f}% "
            f"neg={item['n_neg']:3d}/{item['n_windows']} "
            f"dd={item['max_drawdown'] * 100:5.1f}% "
            f"turn={item['median_turnover_x_initial']:6.1f}x "
            f"{cfg['factor']} K={cfg['pack_size']} sp={cfg['score_power']:g} "
            f"vp={cfg['vol_power']:g} gate={cfg['score_gate']:g} gross={cfg['gross_scale']:g} "
            f"every={cfg['rebalance_every']} thr={cfg['rebalance_threshold']:g}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
