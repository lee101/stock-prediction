#!/usr/bin/env python3
"""
Compare HF (profit-tracking) vs Pufferlib PPO trainings on the same symbol.

This orchestrator optionally trains both, then evaluates and prints a PnL comparison.

Usage examples:
  PYTHONPATH=$(pwd) python scripts/compare_trainers.py --symbol AAPL --data-dir data \
      --hf-steps 800 --puffer-steps 100000 --skip-puffer

Notes:
  - By default this runs light/truncated training so you can verify pipeline.
  - For a full run, increase --hf-steps and --puffer-steps.
  - Stable-Baselines3 is required for Pufferlib PPO; if not available, skip.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def run_hf_training(symbol: str, data_dir: str, hf_steps: int):
    """Train HF model with profit tracking and evaluate PnL on test split."""
    # Lazy imports to keep CLI light
    from hftraining.train_with_profit import train_single_stock_with_profit
    from hftraining.config import create_config
    from hftraining.hf_trainer import TransformerTradingModel, HFTrainingConfig
    from hftraining.data_utils import StockDataProcessor, split_data, create_sequences, load_local_stock_data
    from hftraining.profit_tracker import ProfitTracker
    import torch
    from pathlib import Path

    cfg = create_config("quick_test")
    # Keep it light but not trivial
    cfg.model.hidden_size = max(128, cfg.model.hidden_size)
    cfg.model.num_layers = max(4, cfg.model.num_layers)
    cfg.training.max_steps = int(hf_steps)
    cfg.training.batch_size = 8

    result = train_single_stock_with_profit(symbol, config=cfg, data_dir=data_dir)
    if not result:
        raise SystemExit(f"HF training failed or no data for {symbol}")
    trained_model, model_path = result

    # Re-load saved checkpoint artifacts
    ckpt = torch.load(model_path, map_location="cpu")
    hf_cfg: HFTrainingConfig = ckpt["config"]
    input_dim = int(ckpt["input_dim"]) if "input_dim" in ckpt else None

    # Prepare test data using same local CSV
    local = load_local_stock_data([symbol], data_dir=data_dir)
    if symbol not in local:
        raise SystemExit(f"No CSV found for {symbol} in {data_dir}")
    df = local[symbol]
    proc = StockDataProcessor(
        sequence_length=hf_cfg.sequence_length,
        prediction_horizon=hf_cfg.prediction_horizon,
    )
    feats = proc.prepare_features(df)
    # For a fair eval, just standardize by training portion stats
    split_idx = int(len(feats) * 0.7)
    proc.fit_scalers(feats[:split_idx])
    feats_norm = proc.transform(feats)
    _, _, test_data = split_data(feats_norm, 0.7, 0.15, 0.15)

    # Build model and load weights
    model = TransformerTradingModel(hf_cfg, input_dim=input_dim or feats.shape[1])
    model.load_state_dict(ckpt["model_state_dict"])  # type: ignore[arg-type]
    model.eval()

    # Create sequences for test evaluation
    try:
        _, test_targets, _ = create_sequences(
            test_data, hf_cfg.sequence_length, hf_cfg.prediction_horizon
        )
        # For evaluation we use the last window repeatedly (fast path)
        # A more complete pass would iterate all windows.
        from torch.utils.data import DataLoader
        class _Dataset:
            def __init__(self, data, sl, ph):
                self.data = data
                self.sl = sl
                self.ph = ph
            def __len__(self):
                return max(1, len(self.data) - self.sl - self.ph)
            def __getitem__(self, i):
                x = self.data[i:i+self.sl]
                y = self.data[i+self.sl:i+self.sl+self.ph]
                return x, y
        ds = _Dataset(test_data, hf_cfg.sequence_length, hf_cfg.prediction_horizon)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        preds = []
        labels = []
        import torch
        with torch.no_grad():
            for x, y in dl:
                out = model(x.float())
                price_pred = out.get("price_predictions")
                preds.append(price_pred.cpu())
                labels.append(y[:, :, :price_pred.shape[-1]].cpu())
        predictions = torch.cat(preds, dim=0)
        actuals = torch.cat(labels, dim=0)
    except Exception:
        # Fallback: single window eval if sequence creation fails
        import torch
        x = torch.tensor(test_data[-hf_cfg.sequence_length:]).unsqueeze(0).float()
        y = torch.tensor(test_data[-hf_cfg.prediction_horizon:]).unsqueeze(0).float()
        out = model(x)
        predictions = out["price_predictions"].cpu()
        actuals = y[:, :, :predictions.shape[-1]]

    # Compute profit-style metrics
    tracker = ProfitTracker(initial_capital=10_000, commission=0.001)
    metrics = tracker.calculate_metrics_from_predictions(predictions, actuals)
    return {
        "framework": "hf",
        "symbol": symbol,
        **metrics.to_dict(),
        "model_path": model_path,
    }


def maybe_run_puffer(symbol: str, data_dir: str, puffer_steps: int):
    """Train + eval PPO model via pufferlibtraining; returns metrics or None if unavailable."""
    try:
        import stable_baselines3  # noqa: F401
    except Exception:
        print("[pufferlib] stable-baselines3 not available; skipping puffer run.")
        return None

    import subprocess, sys as _sys
    base = Path(__file__).resolve().parent.parent
    train = base / "pufferlibtraining" / "train_ppo.py"
    evalp = base / "pufferlibtraining" / "eval_model.py"
    models_dir = base / "pufferlibtraining" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train
    cmd_train = [
        _sys.executable, str(train),
        "--symbol", symbol,
        "--data-dir", data_dir,
        "--total-timesteps", str(int(puffer_steps)),
        "--n-envs", "4",
        "--device", "cpu",
    ]
    print("Running:", " ".join(cmd_train))
    subprocess.run(cmd_train, check=True)

    # Find best model
    model_path = None
    for p in sorted(models_dir.glob("*.zip")):
        if symbol.lower() in p.stem:
            model_path = p
            break
    if model_path is None:
        # Try the default best model path from EvalCallback
        candidates = list(models_dir.glob("best_model.zip"))
        model_path = candidates[0] if candidates else None
    if model_path is None:
        print("No PPO model found after training; skipping puffer eval.")
        return None

    # Eval
    out_dir = base / "pufferlibtraining" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_eval = [
        _sys.executable, str(evalp),
        "--symbol", symbol,
        "--data-dir", data_dir,
        "--model-path", str(model_path),
        "--output-dir", str(out_dir),
    ]
    print("Running:", " ".join(cmd_eval))
    subprocess.run(cmd_eval, check=True)

    # Read metrics json
    metrics_file = out_dir / f"metrics_{symbol.lower()}.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        metrics.update({"framework": "puffer", "model_path": str(model_path)})
        return metrics
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--hf-steps", type=int, default=800)
    ap.add_argument("--puffer-steps", type=int, default=100000)
    ap.add_argument("--skip-puffer", action="store_true")
    ap.add_argument("--output", default="reports/trainer_comparison.json")
    args = ap.parse_args()

    out = {"symbol": args.symbol, "data_dir": args.data_dir}

    print("\n=== HF Training + Eval ===")
    hf_metrics = run_hf_training(args.symbol, args.data_dir, args.hf_steps)
    out["hf"] = hf_metrics
    print("HF metrics:", json.dumps({k: v for k, v in hf_metrics.items() if k not in {"equity_curve", "model_path"}}, indent=2))

    puff_metrics = None
    if not args.skip_puffer:
        try:
            print("\n=== Pufferlib PPO Training + Eval ===")
            puff_metrics = maybe_run_puffer(args.symbol, args.data_dir, args.puffer_steps)
            if puff_metrics:
                out["puffer"] = puff_metrics
                print("Puffer metrics:", json.dumps(puff_metrics, indent=2))
        except Exception as e:
            print("Pufferlib run failed:", e)

    # Simple comparison summary
    def _ret(m):
        if not m:
            return None
        return float(m.get("total_return") or m.get("total_return", 0))
    summary = {
        "hf_total_return": _ret(hf_metrics),
        "puffer_total_return": _ret(puff_metrics) if puff_metrics else None,
        "winner": None,
    }
    if summary["puffer_total_return"] is not None:
        summary["winner"] = "hf" if summary["hf_total_return"] >= summary["puffer_total_return"] else "puffer"
    out["summary"] = summary

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("\nSaved comparison to:", out_path)


if __name__ == "__main__":
    sys.exit(main())

