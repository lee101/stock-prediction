"""Train the signal calibrator on historical data using differentiable sim.

Loads hourly OHLCV + Chronos2 forecast cache, computes 16 features per symbol,
applies SignalCalibrator to generate buy/sell prices and intensity, then
runs through the differentiable sim to optimize Sortino via AdamW.

Usage:
    source .venv313/bin/activate
    python rl_trading_agent_binance/train_calibrator.py \
        --symbols BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD \
        --epochs 200 --lr 1e-3 --save-dir rl_trading_agent_binance/calibrator_checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


sys.modules.setdefault("train_calibrator", sys.modules[__name__])
sys.modules.setdefault("rl_trading_agent_binance.train_calibrator", sys.modules[__name__])

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from differentiable_loss_utils import (
    DEFAULT_MAKER_FEE_RATE,
    combined_sortino_pnl_loss,
    compute_hourly_objective,
    simulate_hourly_trades,
    simulate_hourly_trades_binary,
)
from rl_signal import _load_forecast_parquet, compute_symbol_features
from signal_calibrator import CalibrationConfig, SignalCalibrator, save_calibrator


DEPLOYED_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD", "LINKUSD")
DATA_ROOT = REPO / "trainingdatahourly" / "crypto"
FORECAST_ROOT = REPO / "binanceneural" / "forecast_cache"


def load_symbol_data(symbol: str, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    path = data_root / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def prepare_symbol_tensors(
    symbol: str,
    data_root: Path = DATA_ROOT,
    forecast_root: Path = FORECAST_ROOT,
    device: str = "cpu",
) -> dict:
    df = load_symbol_data(symbol, data_root)
    fc_h1 = _load_forecast_parquet(forecast_root, 1, symbol)
    fc_h24 = _load_forecast_parquet(forecast_root, 24, symbol)
    features_np = compute_symbol_features(df, fc_h1, fc_h24)

    n = len(df)
    if features_np.shape[0] != n:
        n = min(features_np.shape[0], n)
        features_np = features_np[:n]
        df = df.iloc[:n]

    return {
        "symbol": symbol,
        "features": torch.tensor(features_np, dtype=torch.float32, device=device),
        "opens": torch.tensor(df["open"].values[:n], dtype=torch.float32, device=device),
        "highs": torch.tensor(df["high"].values[:n], dtype=torch.float32, device=device),
        "lows": torch.tensor(df["low"].values[:n], dtype=torch.float32, device=device),
        "closes": torch.tensor(df["close"].values[:n], dtype=torch.float32, device=device),
        "timestamps": df.index[:n],
        "n_bars": n,
    }


def time_split(n: int, train_frac: float = 0.70, val_frac: float = 0.15):
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def run_sim(
    calibrator: SignalCalibrator,
    features: torch.Tensor,
    closes: torch.Tensor,
    highs: torch.Tensor,
    lows: torch.Tensor,
    opens: torch.Tensor,
    maker_fee: float = DEFAULT_MAKER_FEE_RATE,
    fill_buffer_pct: float = 0.0005,
    temperature: float = 0.01,
    decision_lag: int = 2,
    binary: bool = False,
) -> tuple[torch.Tensor, dict]:
    result_tuple = calibrator.to_prices(features, closes)
    directional = len(result_tuple) == 4
    if directional:
        buy_price, sell_price, buy_int, sell_int = result_tuple
    else:
        buy_price, sell_price, intensity = result_tuple
    sim_fn = simulate_hourly_trades_binary if binary else simulate_hourly_trades
    kwargs = {
        "highs": highs.unsqueeze(0),
        "lows": lows.unsqueeze(0),
        "closes": closes.unsqueeze(0),
        "opens": opens.unsqueeze(0),
        "buy_prices": buy_price.unsqueeze(0),
        "sell_prices": sell_price.unsqueeze(0),
        "maker_fee": maker_fee,
        "initial_cash": 1.0,
        "decision_lag_bars": decision_lag,
        "fill_buffer_pct": fill_buffer_pct,
        "can_short": False,
    }
    if directional:
        kwargs["trade_intensity"] = buy_int.unsqueeze(0)
        kwargs["buy_trade_intensity"] = buy_int.unsqueeze(0)
        kwargs["sell_trade_intensity"] = sell_int.unsqueeze(0)
    else:
        kwargs["trade_intensity"] = intensity.unsqueeze(0)
    if not binary:
        kwargs["temperature"] = temperature
    result = sim_fn(**kwargs)
    returns = result.returns.squeeze(0)
    final_value = result.portfolio_values.squeeze(0)[-1].item() if result.portfolio_values.numel() > 0 else 1.0
    total_return = final_value - 1.0
    score, sortino, ann_return = compute_hourly_objective(returns.unsqueeze(0))
    metrics = {
        "sortino": sortino.item(),
        "return": total_return,
        "ann_return": ann_return.item(),
        "score": score.item(),
        "final_value": final_value,
        "n_bars": returns.shape[-1],
    }
    return returns, metrics


def train_one_symbol(
    symbol: str,
    data: dict,
    config: CalibrationConfig,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    maker_fee: float = DEFAULT_MAKER_FEE_RATE,
    fill_buffer_pct: float = 0.0005,
    temperature: float = 0.01,
    decision_lag: int = 2,
    device: str = "cpu",
    return_weight: float = 0.1,
    save_dir: Path | None = None,
) -> dict:
    n = data["n_bars"]
    train_sl, val_sl, test_sl = time_split(n)

    calibrator = SignalCalibrator(config).to(device)
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_sortino = -float("inf")
    best_state = None
    best_epoch = -1
    history = []

    for epoch in range(epochs):
        calibrator.train()
        optimizer.zero_grad()

        train_returns, train_metrics = run_sim(
            calibrator,
            data["features"][train_sl],
            data["closes"][train_sl],
            data["highs"][train_sl],
            data["lows"][train_sl],
            data["opens"][train_sl],
            maker_fee=maker_fee,
            fill_buffer_pct=fill_buffer_pct,
            temperature=temperature,
            decision_lag=decision_lag,
            binary=False,
        )
        loss = combined_sortino_pnl_loss(
            train_returns.unsqueeze(0),
            return_weight=return_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        calibrator.eval()
        with torch.no_grad():
            _val_returns, val_metrics = run_sim(
                calibrator,
                data["features"][val_sl],
                data["closes"][val_sl],
                data["highs"][val_sl],
                data["lows"][val_sl],
                data["opens"][val_sl],
                maker_fee=maker_fee,
                fill_buffer_pct=fill_buffer_pct,
                decision_lag=decision_lag,
                binary=True,
            )

        record = {
            "epoch": epoch,
            "train_sortino": train_metrics["sortino"],
            "train_return": train_metrics["return"],
            "val_sortino": val_metrics["sortino"],
            "val_return": val_metrics["return"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)

        if val_metrics["sortino"] > best_val_sortino:
            best_val_sortino = val_metrics["sortino"]
            best_state = {k: v.clone() for k, v in calibrator.state_dict().items()}
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"  [{symbol}] ep={epoch:3d} "
                f"train_sort={train_metrics['sortino']:+.2f} train_ret={train_metrics['return']:+.4f} | "
                f"val_sort={val_metrics['sortino']:+.2f} val_ret={val_metrics['return']:+.4f} | "
                f"best_val={best_val_sortino:+.2f}@ep{best_epoch}"
            )

    if best_state is not None:
        calibrator.load_state_dict(best_state)

    calibrator.eval()
    with torch.no_grad():
        _test_returns, test_metrics = run_sim(
            calibrator,
            data["features"][test_sl],
            data["closes"][test_sl],
            data["highs"][test_sl],
            data["lows"][test_sl],
            data["opens"][test_sl],
            maker_fee=maker_fee,
            fill_buffer_pct=fill_buffer_pct,
            decision_lag=decision_lag,
            binary=True,
        )

    print(
        f"  [{symbol}] TEST sort={test_metrics['sortino']:+.2f} "
        f"ret={test_metrics['return']:+.4f} (binary fills, lag={decision_lag})"
    )

    result = {
        "symbol": symbol,
        "best_epoch": best_epoch,
        "best_val_sortino": best_val_sortino,
        "test_metrics": test_metrics,
        "history": history,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"{symbol}_calibrator.pt"
        save_calibrator(
            calibrator,
            ckpt_path,
            config,
            metadata={
                "symbol": symbol,
                "best_epoch": best_epoch,
                "test_sortino": test_metrics["sortino"],
                "test_return": test_metrics["return"],
                "maker_fee": maker_fee,
                "fill_buffer_pct": fill_buffer_pct,
                "decision_lag": decision_lag,
            },
        )
        print(f"  [{symbol}] saved -> {ckpt_path}")

    return result


def compute_baseline(
    data: dict,
    config: CalibrationConfig,
    maker_fee: float = DEFAULT_MAKER_FEE_RATE,
    fill_buffer_pct: float = 0.0005,
    decision_lag: int = 2,
    device: str = "cpu",
) -> dict:
    """Evaluate uncalibrated baseline (identity calibrator, zero-init)."""
    calibrator = SignalCalibrator(config).to(device)
    calibrator.eval()
    n = data["n_bars"]
    _, _, test_sl = time_split(n)
    with torch.no_grad():
        _, metrics = run_sim(
            calibrator,
            data["features"][test_sl],
            data["closes"][test_sl],
            data["highs"][test_sl],
            data["lows"][test_sl],
            data["opens"][test_sl],
            maker_fee=maker_fee,
            fill_buffer_pct=fill_buffer_pct,
            decision_lag=decision_lag,
            binary=True,
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train signal calibrator per symbol")
    parser.add_argument("--symbols", type=str, default=",".join(DEPLOYED_SYMBOLS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-price-bps", type=float, default=25.0)
    parser.add_argument("--max-amount-adj", type=float, default=0.30)
    parser.add_argument("--base-buy-offset", type=float, default=-0.001)
    parser.add_argument("--base-sell-offset", type=float, default=0.008)
    parser.add_argument("--base-intensity", type=float, default=0.5)
    parser.add_argument("--maker-fee", type=float, default=DEFAULT_MAKER_FEE_RATE)
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0005)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--return-weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="rl_trading_agent_binance/calibrator_checkpoints")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--forecast-root", type=str, default=str(FORECAST_ROOT))
    parser.add_argument("--directional", action="store_true", help="Use separate buy/sell intensities")
    parser.add_argument("--window-hours", type=int, default=0, help="Train on last N hours only (0=full split)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    config = CalibrationConfig(
        hidden=args.hidden,
        max_price_adj_bps=args.max_price_bps,
        max_amount_adj=args.max_amount_adj,
        base_buy_offset=args.base_buy_offset,
        base_sell_offset=args.base_sell_offset,
        base_intensity=args.base_intensity,
        directional=args.directional,
    )
    save_dir = Path(args.save_dir)

    print(f"Config: {config}")
    print(f"Symbols: {symbols}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, WD: {args.weight_decay}")
    print(f"Sim: fee={args.maker_fee}, buffer={args.fill_buffer_pct}, lag={args.decision_lag}, temp={args.temperature}")
    print()

    results = []
    for symbol in symbols:
        print(f"--- {symbol} ---")
        try:
            data = prepare_symbol_tensors(
                symbol,
                data_root=Path(args.data_root),
                forecast_root=Path(args.forecast_root),
                device=args.device,
            )
            print(f"  loaded {data['n_bars']} bars")

            baseline = compute_baseline(
                data,
                config,
                maker_fee=args.maker_fee,
                fill_buffer_pct=args.fill_buffer_pct,
                decision_lag=args.decision_lag,
                device=args.device,
            )
            print(f"  baseline sort={baseline['sortino']:+.2f} ret={baseline['return']:+.4f}")

            result = train_one_symbol(
                symbol,
                data,
                config,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                maker_fee=args.maker_fee,
                fill_buffer_pct=args.fill_buffer_pct,
                temperature=args.temperature,
                decision_lag=args.decision_lag,
                device=args.device,
                return_weight=args.return_weight,
                save_dir=save_dir,
            )
            result["baseline"] = baseline
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Symbol':<10} {'Base Sort':>10} {'Base Ret':>10} {'Cal Sort':>10} {'Cal Ret':>10} {'Ep':>4} {'Improve':>10}"
    )
    print("-" * 70)
    for r in results:
        bs = r["baseline"]["sortino"]
        br = r["baseline"]["return"]
        cs = r["test_metrics"]["sortino"]
        cr = r["test_metrics"]["return"]
        improvement = cs - bs
        print(
            f"{r['symbol']:<10} {bs:+10.2f} {br:+10.4f} {cs:+10.2f} {cr:+10.4f} {r['best_epoch']:4d} {improvement:+10.2f}"
        )

    summary_path = save_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in results:
        summary.append(
            {
                "symbol": r["symbol"],
                "best_epoch": r["best_epoch"],
                "baseline_sortino": r["baseline"]["sortino"],
                "baseline_return": r["baseline"]["return"],
                "test_sortino": r["test_metrics"]["sortino"],
                "test_return": r["test_metrics"]["return"],
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
