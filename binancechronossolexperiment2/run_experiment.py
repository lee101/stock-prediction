from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame

from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binancechronossolexperiment.marketsimulator import (
    BinanceMarketSimulator,
    SimulationConfig,
    save_trade_plot,
)
from binancechronossolexperiment.metrics import annualized_return

DEFAULT_MODEL_ID = "chronos2_finetuned/SOLUSDT_lora_20260202_030749/finetuned-ckpt"


def _package_checkpoint(
    *,
    artifacts,
    training_config: TrainingConfig,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
        checkpoint = torch.load(artifacts.best_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        state_dict = artifacts.state_dict
    payload = {
        "state_dict": state_dict,
        "config": asdict(training_config),
        "feature_columns": list(artifacts.feature_columns),
        "normalizer": artifacts.normalizer.to_dict(),
        "best_checkpoint": str(artifacts.best_checkpoint) if artifacts.best_checkpoint else None,
    }
    output_path = output_dir / "policy_checkpoint.pt"
    torch.save(payload, output_path)
    return output_path


def _stringify_paths(payload: object) -> object:
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _stringify_paths(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        converted = [_stringify_paths(value) for value in payload]
        return converted if isinstance(payload, list) else tuple(converted)
    return payload


def _simulate_window(
    *,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    initial_cash: float,
) -> dict:
    sim = BinanceMarketSimulator(SimulationConfig(initial_cash=initial_cash))
    result = sim.run(bars, actions)
    metrics = dict(result.metrics)
    metrics["annualized_return"] = annualized_return(result.combined_equity)
    metrics["final_equity"] = float(result.combined_equity.iloc[-1])
    return {"result": result, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train + simulate SOLUSDT Chronos2 neural hourly policy (nanochat-style upgrades)."
    )
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--forecast-cache-root", default="binancechronossolexperiment/forecast_cache")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--context-hours", type=int, default=3072)
    parser.add_argument("--chronos-batch-size", type=int, default=32)
    parser.add_argument("--horizons", default="1,4,24", help="Comma-separated forecast horizons")
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--val-days", type=int, default=20)
    parser.add_argument("--test-days", type=int, default=10)
    parser.add_argument("--max-history-days", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--weight-decay-schedule", default="linear_to_zero")
    parser.add_argument("--weight-decay-end", type=float, default=0.0)
    parser.add_argument("--optimizer", default="muon_mix")
    parser.add_argument("--model-arch", default="nano", help="classic or nano")
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--logits-softcap", type=float, default=12.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-momentum-start", type=float, default=0.85)
    parser.add_argument("--muon-momentum-warmup-steps", type=int, default=300)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--causal-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--attention-window", type=int, default=64)
    parser.add_argument("--use-residual-scalars", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual-scale-init", type=float, default=1.0)
    parser.add_argument("--skip-scale-init", type=float, default=0.0)
    parser.add_argument("--use-value-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value-embedding-every", type=int, default=2)
    parser.add_argument("--value-embedding-scale", type=float, default=1.0)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", default="bfloat16")
    parser.add_argument("--use-flash-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())
    if not horizons:
        raise ValueError("At least one horizon must be provided")

    attention_window = None
    if args.attention_window and args.attention_window > 0:
        attention_window = int(args.attention_window)

    run_name = args.run_name or time.strftime("chronos_sol_v2_%Y%m%d_%H%M%S")
    checkpoint_root = Path("binancechronossolexperiment2/checkpoints") / run_name
    results_root = Path("binancechronossolexperiment2/results") / run_name
    plot_root = Path("binancechronossolexperiment2/plots") / run_name

    data_module = ChronosSolDataModule(
        symbol=args.symbol,
        data_root=Path(args.data_root),
        forecast_cache_root=Path(args.forecast_cache_root),
        forecast_horizons=horizons,
        context_hours=args.context_hours,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=args.chronos_batch_size,
        model_id=args.model_id,
        sequence_length=args.sequence_length,
        split_config=SplitConfig(val_days=args.val_days, test_days=args.test_days),
        max_history_days=args.max_history_days,
        cache_only=args.cache_only,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        model_arch=args.model_arch,
        num_kv_heads=args.num_kv_heads,
        mlp_ratio=args.mlp_ratio,
        logits_softcap=args.logits_softcap,
        rope_base=args.rope_base,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_momentum_start=args.muon_momentum_start,
        muon_momentum_warmup_steps=args.muon_momentum_warmup_steps,
        muon_ns_steps=args.muon_ns_steps,
        muon_nesterov=args.muon_nesterov,
        use_qk_norm=args.qk_norm,
        use_causal_attention=args.causal_attn,
        rms_norm_eps=args.rms_norm_eps,
        attention_window=attention_window,
        use_residual_scalars=args.use_residual_scalars,
        residual_scale_init=args.residual_scale_init,
        skip_scale_init=args.skip_scale_init,
        use_value_embedding=args.use_value_embedding,
        value_embedding_every=args.value_embedding_every,
        value_embedding_scale=args.value_embedding_scale,
        weight_decay_schedule=args.weight_decay_schedule,
        weight_decay_end=args.weight_decay_end,
        use_compile=not args.no_compile,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        use_flash_attention=args.use_flash_attention,
        checkpoint_root=checkpoint_root,
        log_dir=Path("tensorboard_logs") / "binancechronossolexperiment2",
    )

    trainer = BinanceHourlyTrainer(training_config, data_module)
    artifacts = trainer.train()
    packaged_ckpt = _package_checkpoint(
        artifacts=artifacts,
        training_config=training_config,
        output_dir=checkpoint_root,
    )

    model, normalizer, feature_columns, _ = load_policy_checkpoint(str(packaged_ckpt))
    test_frame = data_module.test_frame.copy()
    actions = generate_actions_from_frame(
        model=model,
        frame=test_frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=args.sequence_length,
        horizon=horizons[0],
    )

    test_start = data_module.test_window_start
    bars = test_frame[test_frame["timestamp"] >= test_start].copy()
    actions = actions[actions["timestamp"] >= test_start].copy()
    metrics_payload = {}

    test_result = _simulate_window(
        bars=bars,
        actions=actions,
        initial_cash=args.initial_cash,
    )
    metrics_payload["test"] = test_result["metrics"]

    if not bars.empty:
        two_day_start = bars["timestamp"].max() - pd.Timedelta(days=2)
        bars_2d = bars[bars["timestamp"] >= two_day_start]
        actions_2d = actions[actions["timestamp"] >= two_day_start]
        if not bars_2d.empty and not actions_2d.empty:
            short_result = _simulate_window(
                bars=bars_2d,
                actions=actions_2d,
                initial_cash=args.initial_cash,
            )
            metrics_payload["last_2_days"] = short_result["metrics"]

    print("Test metrics:")
    for key, value in metrics_payload.get("test", {}).items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    if "last_2_days" in metrics_payload:
        print("Last 2 days metrics:")
        for key, value in metrics_payload["last_2_days"].items():
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    results_root.mkdir(parents=True, exist_ok=True)
    results_path = results_root / "simulation_metrics.json"
    results_payload = {
        "run_name": run_name,
        "symbol": args.symbol,
        "test_window_start": str(test_start),
        "horizons": horizons,
        "chronos_model_id": args.model_id,
        "training_config": _stringify_paths(asdict(training_config)),
        "metrics": metrics_payload,
    }
    results_path.write_text(json.dumps(results_payload, indent=2))
    print(f"Saved metrics: {results_path}")

    if not args.no_plot:
        plot_root.mkdir(parents=True, exist_ok=True)
        output_path = plot_root / f"{args.symbol.lower()}_test.png"
        saved = save_trade_plot(
            args.symbol.upper(),
            bars,
            actions,
            test_result["result"].per_symbol[args.symbol.upper()],
            output_path,
        )
        print(f"Saved plot: {saved}")


if __name__ == "__main__":
    main()
