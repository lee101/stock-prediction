"""CuteChronos2 wrapper for worksteal strategy forecasting.

Singleton-cached pipeline that produces p10/p50/p90 forecasts for close/high/low.
Falls back to original ChronosPipeline if Triton/cutechronos unavailable.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

CUTEDSL_PATH = str(Path(__file__).resolve().parents[1] / "cutedsl")
HF_CACHE = "/vfast/data/stock/hf_cache"
DEFAULT_MODEL = "amazon/chronos-bolt-base"
LORA_ROOT = Path(__file__).resolve().parents[1] / "binanceneural" / "chronos2_finetuned"

LORA_MAP: Dict[str, str] = {
    "BTCUSDT": "BTCUSD_lora_20260203_051412",
    "ETHUSDT": "ETHUSD_lora_20260203_051846",
    "SOLUSDT": "SOLUSD_lora_20260203_052715",
    "LINKUSDT": "LINKUSD_lora_20260203_052258",
}

_pipeline_cache: Dict[str, object] = {}
_cute_available: Optional[bool] = None


def _ensure_cutedsl_path():
    if CUTEDSL_PATH not in sys.path:
        sys.path.insert(0, CUTEDSL_PATH)


def _ensure_hf_env():
    os.environ.setdefault("HF_HOME", HF_CACHE)


def _pad_to_horizon(arr: np.ndarray, horizon: int) -> np.ndarray:
    if len(arr) >= horizon:
        return arr[:horizon]
    pad_val = arr[-1] if len(arr) > 0 else 0.0
    return np.concatenate([arr, np.full(horizon - len(arr), pad_val)])


def _extract_quantiles(q_tensor: torch.Tensor, horizon: int) -> Dict[str, np.ndarray]:
    """Extract p10/p50/p90 from (1, H, 3) tensor, padded to horizon."""
    q = q_tensor.squeeze(0).numpy()
    return {
        "p10": _pad_to_horizon(q[:, 0], horizon),
        "p50": _pad_to_horizon(q[:, 1], horizon),
        "p90": _pad_to_horizon(q[:, 2], horizon),
    }


def _naive_fallback(last_val: float, horizon: int) -> Dict[str, np.ndarray]:
    arr = np.full(horizon, last_val)
    return {"p10": arr * 0.98, "p50": arr.copy(), "p90": arr * 1.02}


def _check_cute_available() -> bool:
    global _cute_available
    if _cute_available is not None:
        return _cute_available
    try:
        _ensure_cutedsl_path()
        from cutechronos.pipeline import CuteChronos2Pipeline  # noqa: F401
        _cute_available = True
    except Exception:
        _cute_available = False
    return _cute_available


def _load_cute_pipeline(model_path: str = DEFAULT_MODEL, device: str = "cuda"):
    _ensure_cutedsl_path()
    _ensure_hf_env()
    from cutechronos.pipeline import CuteChronos2Pipeline
    return CuteChronos2Pipeline.from_pretrained(
        model_path, device=device, use_cute=True,
    )


def _load_original_pipeline(model_path: str = DEFAULT_MODEL, device: str = "cuda"):
    _ensure_hf_env()
    try:
        _ensure_cutedsl_path()
        from cutechronos.pipeline import CuteChronos2Pipeline
        return CuteChronos2Pipeline.from_pretrained(
            model_path, device=device, use_cute=False,
        )
    except Exception:
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(model_path)


def get_pipeline(model_path: str = DEFAULT_MODEL, device: str = "cuda", use_cute: bool = True):
    """Get or create singleton pipeline instance."""
    key = f"{model_path}:{device}:{use_cute}"
    if key in _pipeline_cache:
        return _pipeline_cache[key]

    if use_cute and _check_cute_available():
        try:
            pipe = _load_cute_pipeline(model_path, device)
            logger.info("Loaded CuteChronos2Pipeline: %s", model_path)
            _pipeline_cache[key] = pipe
            return pipe
        except Exception as e:
            logger.warning("CuteChronos2 load failed, falling back: %s", e)

    pipe = _load_original_pipeline(model_path, device)
    logger.info("Loaded original pipeline: %s", model_path)
    _pipeline_cache[key] = pipe
    return pipe


def _find_lora_path(symbol: str) -> Optional[Path]:
    """Find finetuned-ckpt for a symbol if available."""
    sym_key = symbol.upper().replace("/", "")
    if sym_key in LORA_MAP:
        ckpt = LORA_ROOT / LORA_MAP[sym_key] / "finetuned-ckpt"
        if ckpt.exists() and (ckpt / "config.json").exists():
            return ckpt
    try:
        candidates = sorted(LORA_ROOT.glob(f"{sym_key}_lora_*/finetuned-ckpt"))
    except OSError:
        return None
    for c in reversed(candidates):
        if (c / "config.json").exists():
            return c
    return None


def _prepare_ohlc_contexts(bars_df: pd.DataFrame, max_len: int = 512) -> Dict[str, torch.Tensor]:
    """Extract close/high/low series as tensors from OHLC bars."""
    df = bars_df.tail(max_len)
    close_vals = df["close"].astype(float).values
    close_tensor = torch.tensor(close_vals, dtype=torch.float32)
    result: Dict[str, torch.Tensor] = {"close": close_tensor}
    for col in ("high", "low"):
        if col in df.columns:
            result[col] = torch.tensor(df[col].astype(float).values, dtype=torch.float32)
        else:
            result[col] = close_tensor.clone()
    return result


def forecast_symbol(
    symbol: str,
    bars_df: pd.DataFrame,
    horizon: int = 24,
    model_path: str = DEFAULT_MODEL,
    device: str = "cuda",
    use_cute: bool = True,
    context_length: int = 512,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Forecast p10/p50/p90 for close/high/low.

    Returns dict like:
        {"close": {"p10": array, "p50": array, "p90": array},
         "high": {"p10": ..., "p50": ..., "p90": ...},
         "low": {"p10": ..., "p50": ..., "p90": ...}}
    Each array has shape (horizon,).
    """
    lora_path = _find_lora_path(symbol)
    actual_model = str(lora_path) if lora_path else model_path
    pipe = get_pipeline(actual_model, device, use_cute)

    contexts = _prepare_ohlc_contexts(bars_df, max_len=context_length)
    pred_len = min(horizon, getattr(pipe, "model_prediction_length", 64))

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for col in ("close", "high", "low"):
        try:
            quantiles, _ = pipe.predict_quantiles(
                contexts[col],
                prediction_length=pred_len,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            result[col] = _extract_quantiles(quantiles[0], horizon)
        except Exception as e:
            logger.warning("forecast_symbol %s/%s failed: %s, using naive", symbol, col, e)
            result[col] = _naive_fallback(float(contexts[col][-1]), horizon)

    return result


def forecast_batch(
    symbols: List[str],
    bars_dict: Dict[str, pd.DataFrame],
    horizon: int = 24,
    model_path: str = DEFAULT_MODEL,
    device: str = "cuda",
    use_cute: bool = True,
    batch_size: int = 8,
    context_length: int = 512,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Batch forecast multiple symbols. Returns {symbol: forecast_dict}."""
    pipe = get_pipeline(model_path, device, use_cute)
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    pred_len = min(horizon, getattr(pipe, "model_prediction_length", 64))

    # prepare contexts once per symbol (avoids 3x recomputation)
    prepared: Dict[str, Dict[str, torch.Tensor]] = {}
    sym_order: List[str] = []
    for sym in symbols:
        df = bars_dict.get(sym)
        if df is None or df.empty:
            continue
        prepared[sym] = _prepare_ohlc_contexts(df, max_len=context_length)
        sym_order.append(sym)

    if not sym_order:
        return results

    for col in ("close", "high", "low"):
        all_contexts = [prepared[sym][col] for sym in sym_order]

        for batch_start in range(0, len(all_contexts), batch_size):
            batch_ctx = all_contexts[batch_start:batch_start + batch_size]
            batch_syms = sym_order[batch_start:batch_start + batch_size]
            try:
                quantiles, _ = pipe.predict_quantiles(
                    batch_ctx,
                    prediction_length=pred_len,
                    quantile_levels=[0.1, 0.5, 0.9],
                )
                for i, sym in enumerate(batch_syms):
                    if sym not in results:
                        results[sym] = {}
                    results[sym][col] = _extract_quantiles(quantiles[i], horizon)
            except Exception as e:
                logger.warning("batch forecast failed for %s batch %d: %s", col, batch_start, e)
                for sym in batch_syms:
                    last = float(prepared[sym][col][-1])
                    if sym not in results:
                        results[sym] = {}
                    results[sym][col] = _naive_fallback(last, horizon)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path(__file__).resolve().parents[1] / "trainingdata" / "train"
    test_sym = "BTCUSDT"
    csv_path = data_dir / f"{test_sym}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        result = forecast_symbol(test_sym, df, horizon=24, use_cute=True)
        for col, qdict in result.items():
            print(f"{col}: p10={qdict['p10'][:3]}, p50={qdict['p50'][:3]}, p90={qdict['p90'][:3]}")
    else:
        print(f"No data at {csv_path}")
