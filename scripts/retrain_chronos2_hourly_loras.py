#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chronos2_trainer import TrainerConfig, run_finetune
from src.binance_symbol_utils import normalize_compact_symbol, proxy_symbol_to_usd


DEFAULT_SYMBOLS = "BTCFDUSD,ETHFDUSD,SOLFDUSD,BNBFDUSD"
DEFAULT_DATA_ROOT = Path("trainingdatahourlybinance")
DEFAULT_HPARAM_DIR = Path("hyperparams") / "chronos2" / "hourly"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _base_hourly_config(symbol: str, model_id: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "model": "chronos2",
        "config": {
            "name": "hourly_ctx512_skip1_single",
            "model_id": model_id,
            "device_map": "cuda",
            "context_length": 512,
            "batch_size": 32,
            "quantile_levels": [0.1, 0.5, 0.9],
            "aggregation": "median",
            "sample_count": 0,
            "scaler": "none",
            "predict_kwargs": {},
            "skip_rates": [1],
            "aggregation_method": "single",
            "use_multivariate": False,
        },
        "metadata": {
            "source": "retrain_chronos2_hourly_loras",
            "generated_at": _utc_now_iso(),
            "frequency": "hourly",
        },
    }


def _resolve_template_payload(symbol: str, *, hyperparam_dir: Path) -> Optional[Dict[str, Any]]:
    proxy = proxy_symbol_to_usd(symbol)
    candidate_paths = [
        hyperparam_dir / f"{symbol}.json",
    ]
    if proxy and proxy != symbol:
        candidate_paths.append(hyperparam_dir / f"{proxy}.json")
    for path in candidate_paths:
        if path.exists():
            payload = _load_json(path)
            if isinstance(payload, dict):
                return payload
    return None


def update_hourly_hparams(
    *,
    symbol: str,
    finetuned_model_id: str,
    hyperparam_dir: Path = DEFAULT_HPARAM_DIR,
) -> Path:
    symbol_norm = normalize_compact_symbol(symbol)
    model_id = str(finetuned_model_id)
    template = _resolve_template_payload(symbol_norm, hyperparam_dir=hyperparam_dir)
    if template is None:
        payload = _base_hourly_config(symbol_norm, model_id)
    else:
        payload = copy.deepcopy(template)
        payload["symbol"] = symbol_norm
        payload.setdefault("config", {})
        if isinstance(payload.get("config"), dict):
            payload["config"]["model_id"] = model_id
        payload.setdefault("metadata", {})
        if isinstance(payload.get("metadata"), dict):
            payload["metadata"]["generated_at"] = _utc_now_iso()
            payload["metadata"]["source"] = "retrain_chronos2_hourly_loras"
            payload["metadata"]["finetuned_model_id"] = model_id
    output_path = Path(hyperparam_dir) / f"{symbol_norm}.json"
    _write_json(output_path, payload)
    return output_path


def _parse_symbols(raw: Optional[Sequence[str]], default: str) -> List[str]:
    if raw:
        tokens = []
        for entry in raw:
            for part in str(entry).split(","):
                part = part.strip()
                if part:
                    tokens.append(normalize_compact_symbol(part))
        return [t for t in tokens if t]
    return [normalize_compact_symbol(t) for t in default.split(",") if t.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain Chronos2 hourly LoRAs and update hyperparams/chronos2/hourly.")
    parser.add_argument("--symbol", action="append", dest="symbols", help="Symbol to retrain (repeatable or comma-separated).")
    parser.add_argument("--default-symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("chronos2_finetuned"))
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--test-hours", type=int, default=168)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-name-suffix", default=None, help="Optional suffix appended to the run folder name.")
    parser.add_argument("--no-update-hparams", action="store_true")
    parser.add_argument("--hyperparam-dir", type=Path, default=DEFAULT_HPARAM_DIR)
    args = parser.parse_args(argv)

    symbols = _parse_symbols(args.symbols, args.default_symbols)
    if not symbols:
        raise SystemExit("No symbols provided.")

    logger.info("Retraining {} Chronos2 LoRA(s): {}", len(symbols), symbols)

    for symbol in symbols:
        save_name = None
        if args.save_name_suffix:
            save_name = f"{symbol}_lora_{args.save_name_suffix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        cfg = TrainerConfig(
            symbol=symbol,
            data_root=Path(args.data_root),
            output_root=Path(args.output_root),
            model_id=str(args.model_id),
            device_map=str(args.device_map),
            torch_dtype=args.torch_dtype,
            prediction_length=int(args.prediction_length),
            context_length=int(args.context_length),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            num_steps=int(args.num_steps),
            val_hours=int(args.val_hours),
            test_hours=int(args.test_hours),
            finetune_mode="lora",
            seed=int(args.seed),
            save_name=save_name,
        )

        logger.info("Fine-tuning LoRA for {} (steps={}, lr={:.2e})", symbol, cfg.num_steps, cfg.learning_rate)
        report = run_finetune(cfg)
        finetuned_ckpt = Path(report.output_dir) / "finetuned-ckpt"
        logger.info(
            "{} LoRA ready: {} | val_mae%={:.4f} test_mae%={:.4f}",
            symbol,
            finetuned_ckpt,
            float(report.val_metrics.mae_percent),
            float(report.test_metrics.mae_percent),
        )

        if not args.no_update_hparams:
            updated_path = update_hourly_hparams(
                symbol=symbol,
                finetuned_model_id=str(finetuned_ckpt),
                hyperparam_dir=Path(args.hyperparam_dir),
            )
            logger.info("Updated hourly hyperparams: {}", updated_path)

    logger.info("All LoRA retrains complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
