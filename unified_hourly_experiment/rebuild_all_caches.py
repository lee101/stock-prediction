#!/usr/bin/env python3
"""Rebuild forecast caches for stocks using best LoRA models."""
import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger
from src.chronos2_params import resolve_chronos2_params


BEST_MODELS = {
    "NVDA": "NVDA_lora_nonreg_20260304_213434_ctx512_lr0p00005_st400_r16_a32_d0",
    "MSFT": "MSFT_lora_percent_change_ctx128_lr5e-05_r16_20260213_171918",
    "META": "META_lora_differencing_ctx128_lr5e-05_r16_20260213_172640",
    "GOOG": "GOOG_lora_nonreg_20260304_213434_ctx512_lr0p00005_st400_r16_a32_d0p05",
    "NET": "NET_lora_differencing_ctx128_lr5e-5_r16_20260221_080636",
    "PLTR": "PLTR_lora_nonreg_20260304_213434_ctx512_lr0p00007_st400_r32_a32_d0p05",
    "NYT": "NYT_lora_differencing_ctx128_lr5e-5_r16_20260221_084237",
    "YELP": "YELP_lora_percent_change_ctx128_lr5e-05_r16_20260213_175101",
    "DBX": "DBX_lora_nonreg_20260304_213434_ctx512_lr0p00007_st400_r16_a32_d0",
    "TRIP": "TRIP_lora_nonreg_20260304_213434_ctx512_lr0p00005_st400_r32_a32_d0",
    "KIND": "KIND_lora_differencing_ctx128_lr5e-05_r16_20260215_225721",
    "EBAY": "EBAY_lora_robust_scaling_ctx128_lr5e-5_r16_20260221_083308",
    "MTCH": "MTCH_lora_nonreg_20260304_213434_ctx512_lr0p00007_st400_r32_a32_d0p05",
    "ANGI": "ANGI_lora_differencing_ctx128_lr5e-05_r16_20260215_224308",
    "Z": "Z_lora_differencing_ctx128_lr5e-05_r16_20260215_224604",
    "EXPE": "EXPE_lora_differencing_ctx128_lr5e-05_r16_20260215_224859",
    "BKNG": "BKNG_lora_differencing_ctx128_lr5e-05_r16_20260215_225148",
    "NWSA": "NWSA_lora_differencing_ctx128_lr5e-05_r16_20260215_225432",
    "TSLA": "TSLA_lora_differencing_ctx128_lr5e-05_r16_20260218_102822",
    "AAPL": "AAPL_lora_differencing_ctx128_lr5e-05_r16_20260218_102824",
    "QUBT": "QUBT_lora_percent_change_ctx128_lr5e-05_r16_20260218_224547",
}

CACHE_ROOT = Path("unified_hourly_experiment/forecast_cache")
DATA_ROOT = Path("trainingdatahourly/stocks")
MODEL_ROOT = Path("chronos2_finetuned")
DEFAULT_HORIZONS = "1,24"
DEFAULT_LOOKBACK_HOURS = 8000


def _parse_symbols(raw: str | None) -> list[str]:
    if raw is None:
        return list(BEST_MODELS.keys())
    symbols = [token.strip().upper() for token in str(raw).split(",") if token.strip()]
    if not symbols:
        raise ValueError("Expected at least one symbol in --symbols.")
    return symbols


def _parse_horizons(raw: str) -> str:
    horizons = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        horizons.append(str(int(token)))
    if not horizons:
        raise ValueError("Expected at least one forecast horizon.")
    return ",".join(horizons)

def _resolve_model_path(symbol: str, model_name: str) -> str | Path | None:
    """Resolve a usable Chronos2 model reference for a symbol.

    Prefers the configured model name, but falls back to the most recently
    updated local checkpoint for the symbol when mappings become stale. If no
    local checkpoint exists, falls back to the configured hourly Chronos2
    model_id so cache generation can still proceed from the base model.
    """
    preferred = MODEL_ROOT / model_name / "finetuned-ckpt"
    if preferred.exists():
        return preferred

    fallback_candidates: list[tuple[float, Path]] = []
    for candidate_dir in MODEL_ROOT.glob(f"{symbol}_*"):
        ckpt = candidate_dir / "finetuned-ckpt"
        if ckpt.exists():
            try:
                mtime = ckpt.stat().st_mtime
            except OSError:
                continue
            fallback_candidates.append((mtime, ckpt))

    if not fallback_candidates:
        try:
            params = resolve_chronos2_params(symbol, frequency="hourly")
            fallback_model_id = str(params.get("model_id") or "").strip()
        except Exception as exc:
            logger.warning("Failed to resolve Chronos2 params for {}: {}", symbol, exc)
            fallback_model_id = ""
        if fallback_model_id:
            logger.warning(
                "Model not found for {} -> {}. Falling back to configured Chronos2 model_id: {}",
                symbol,
                preferred,
                fallback_model_id,
            )
            return fallback_model_id
        logger.warning("Model not found and no fallback candidates: {}", preferred)
        return None

    fallback_candidates.sort(key=lambda row: row[0], reverse=True)
    chosen = fallback_candidates[0][1]
    logger.warning(
        "Model not found for {} -> {}. Falling back to latest available checkpoint: {}",
        symbol,
        preferred,
        chosen,
    )
    return chosen

def build_cache(
    symbol: str,
    model_name: str,
    *,
    horizons: str = DEFAULT_HORIZONS,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
):
    """Build forecast cache for a symbol."""
    model_path = _resolve_model_path(symbol, model_name)
    if model_path is None:
        return False

    cmd = [
        sys.executable, "-m", "alpacanewccrosslearning.build_forecasts",
        "--symbols", symbol,
        "--finetuned-model", str(model_path),
        "--forecast-cache-root", str(cache_root),
        "--stock-data-root", str(data_root),
        "--horizons", _parse_horizons(horizons),
        "--lookback-hours", str(int(lookback_hours)),
    ]

    logger.info("Building cache for {} horizons={} lookback={}h...", symbol, _parse_horizons(horizons), int(lookback_hours))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        logger.success(f"Built cache for {symbol}")
        return True
    else:
        logger.error(f"Failed {symbol}: {result.stderr[-200:]}")
        return False

def build_selected_caches(
    *,
    symbols: list[str],
    horizons: str = DEFAULT_HORIZONS,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    cache_root: Path = CACHE_ROOT,
    data_root: Path = DATA_ROOT,
) -> tuple[int, list[str]]:
    success = 0
    failed: list[str] = []

    for symbol in symbols:
        model = BEST_MODELS.get(symbol)
        if not model:
            logger.warning("No best Chronos2 model mapping configured for {}", symbol)
            failed.append(symbol)
            continue
        if build_cache(
            symbol,
            model,
            horizons=horizons,
            lookback_hours=lookback_hours,
            cache_root=cache_root,
            data_root=data_root,
        ):
            success += 1
        else:
            failed.append(symbol)
    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Rebuild stock Chronos2 forecast caches.")
    parser.add_argument("--symbols", default=None, help="Comma-separated subset of stock symbols. Default: all BEST_MODELS.")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS, help="Comma-separated forecast horizons to rebuild.")
    parser.add_argument("--lookback-hours", type=int, default=DEFAULT_LOOKBACK_HOURS)
    parser.add_argument("--forecast-cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--stock-data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    horizons = _parse_horizons(args.horizons)

    logger.info("=" * 60)
    logger.info("Rebuilding forecast caches for {} symbols (horizons={})", len(symbols), horizons)
    logger.info("=" * 60)

    success, failed = build_selected_caches(
        symbols=symbols,
        horizons=horizons,
        lookback_hours=int(args.lookback_hours),
        cache_root=args.forecast_cache_root,
        data_root=args.stock_data_root,
    )

    logger.info("=" * 60)
    logger.info("Complete: {}/{} successful", success, len(symbols))
    if failed:
        logger.warning("Failed: {}", failed)

if __name__ == "__main__":
    main()
