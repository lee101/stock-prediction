from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from loguru import logger

from chronos2_trainer import DEFAULT_TARGET_COLS, _evaluate_pipeline, _load_hourly_frame, _load_pipeline
from src.hourly_data_utils import resolve_hourly_symbol_path
from src.symbol_utils import is_crypto_symbol

from .symbols import build_longable_symbols, build_shortable_symbols, normalize_symbols


@dataclass
class SymbolScore:
    symbol: str
    mae_percent: float
    rows: int
    val_hours: int
    error: Optional[str] = None


def _parse_symbols(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return normalize_symbols([token for token in raw.split(",") if token.strip()])


def _resolve_symbol_path(
    symbol: str,
    *,
    data_root: Optional[Path],
    crypto_root: Optional[Path],
    stock_root: Optional[Path],
) -> Path:
    symbol = symbol.upper()
    if data_root is not None:
        candidate = resolve_hourly_symbol_path(symbol, data_root)
        if candidate is not None:
            return candidate
        raise FileNotFoundError(f"Hourly data for {symbol} not found under {data_root}")
    if is_crypto_symbol(symbol) and crypto_root is not None:
        candidate = resolve_hourly_symbol_path(symbol, crypto_root)
        if candidate is not None:
            return candidate
        raise FileNotFoundError(f"Hourly crypto data for {symbol} not found under {crypto_root}")
    if not is_crypto_symbol(symbol) and stock_root is not None:
        candidate = resolve_hourly_symbol_path(symbol, stock_root)
        if candidate is not None:
            return candidate
        raise FileNotFoundError(f"Hourly stock data for {symbol} not found under {stock_root}")
    fallback_root = Path("trainingdatahourly") / ("crypto" if is_crypto_symbol(symbol) else "stocks")
    candidate = resolve_hourly_symbol_path(symbol, fallback_root)
    if candidate is None:
        raise FileNotFoundError(f"Hourly data for {symbol} not found under {fallback_root}")
    return candidate


def _score_symbol(
    symbol: str,
    *,
    pipeline: object,
    context_length: int,
    prediction_length: int,
    val_hours: int,
    data_root: Optional[Path],
    crypto_root: Optional[Path],
    stock_root: Optional[Path],
) -> SymbolScore:
    symbol = symbol.upper()
    path = _resolve_symbol_path(
        symbol,
        data_root=data_root,
        crypto_root=crypto_root,
        stock_root=stock_root,
    )
    df = _load_hourly_frame(path, DEFAULT_TARGET_COLS)
    rows = len(df)
    if rows <= context_length + 4:
        return SymbolScore(
            symbol=symbol,
            mae_percent=float("inf"),
            rows=rows,
            val_hours=val_hours,
            error=f"Insufficient rows ({rows}) for context_length={context_length}",
        )
    start_idx = max(context_length, rows - val_hours)
    end_idx = rows
    metrics = _evaluate_pipeline(
        pipeline=pipeline,
        df=df,
        target_cols=DEFAULT_TARGET_COLS,
        context_length=context_length,
        prediction_length=prediction_length,
        start_idx=start_idx,
        end_idx=end_idx,
        preaug_choice=None,
    )
    return SymbolScore(
        symbol=symbol,
        mae_percent=float(metrics.mae_percent),
        rows=rows,
        val_hours=val_hours,
    )


def _select_top(symbols: Sequence[str], scores: Dict[str, SymbolScore], top_n: int) -> List[str]:
    ranked = [scores[s] for s in symbols if s in scores and scores[s].error is None]
    ranked.sort(key=lambda item: item.mae_percent)
    if top_n <= 0 or top_n >= len(ranked):
        return [item.symbol for item in ranked]
    return [item.symbol for item in ranked[:top_n]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank constrained symbols by Chronos2 MAE% and suggest groups for LoRA fine-tuning."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols to score.")
    parser.add_argument(
        "--long-stocks",
        default=None,
        help="Comma-separated longable stock symbols (default: NVDA,GOOG,MSFT).",
    )
    parser.add_argument(
        "--short-stocks",
        default=None,
        help="Comma-separated shortable stock symbols (default: YELP,EBAY,TRIP,MTCH,KIND,ANGI,Z,EXPE,BKNG,NWSA,NYT).",
    )
    parser.add_argument(
        "--crypto",
        default=None,
        help="Comma-separated crypto symbols to include (default: BTCUSD,ETHUSD,SOLUSD).",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--crypto-data-root", default=None)
    parser.add_argument("--stock-data-root", default=None)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--top-long", type=int, default=0, help="Top-N long symbols to suggest (0=all).")
    parser.add_argument("--top-short", type=int, default=5, help="Top-N short symbols to suggest (0=all).")
    parser.add_argument("--output-dir", default="alpacaconstrainedexp/outputs")
    args = parser.parse_args()

    long_symbols = build_longable_symbols(
        crypto_symbols=_parse_symbols(args.crypto) if args.crypto else None,
        stock_symbols=_parse_symbols(args.long_stocks) if args.long_stocks else None,
    )
    short_symbols = build_shortable_symbols(
        stock_symbols=_parse_symbols(args.short_stocks) if args.short_stocks else None,
    )

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        symbols = normalize_symbols(long_symbols + short_symbols)
    if not symbols:
        raise ValueError("At least one symbol is required.")

    pipeline = _load_pipeline(args.model_id, args.device_map, args.torch_dtype)

    scores: Dict[str, SymbolScore] = {}
    for symbol in symbols:
        try:
            score = _score_symbol(
                symbol,
                pipeline=pipeline,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
                val_hours=args.val_hours,
                data_root=Path(args.data_root) if args.data_root else None,
                crypto_root=Path(args.crypto_data_root) if args.crypto_data_root else None,
                stock_root=Path(args.stock_data_root) if args.stock_data_root else None,
            )
        except Exception as exc:
            logger.warning("Failed to score %s: %s", symbol, exc)
            score = SymbolScore(
                symbol=symbol,
                mae_percent=float("inf"),
                rows=0,
                val_hours=args.val_hours,
                error=str(exc),
            )
        scores[symbol] = score

    suggested_long = _select_top(long_symbols, scores, args.top_long)
    suggested_short = _select_top(short_symbols, scores, args.top_short)
    suggested_symbols = normalize_symbols(suggested_long + suggested_short)

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "val_hours": args.val_hours,
        "scores": [asdict(score) for score in scores.values()],
        "suggested": {
            "long_symbols": suggested_long,
            "short_symbols": suggested_short,
            "all_symbols": suggested_symbols,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"symbol_groups_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(payload, indent=2))

    print("Suggested long symbols:", ",".join(suggested_long))
    print("Suggested short symbols:", ",".join(suggested_short))
    print("Suggested all symbols:", ",".join(suggested_symbols))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
