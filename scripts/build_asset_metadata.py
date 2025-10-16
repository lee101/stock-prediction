#!/usr/bin/env python3
"""
Build asset metadata for the training pipeline.

The script scans the consolidated ``trainingdata/data_summary.csv`` file,
classifies each symbol as equity or crypto, attaches the default trading fee,
and writes the result to ``trainingdata/asset_metadata.json`` (configurable).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

# Ensure repository root is on the import path so we can reuse the fee constants.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from loss_utils import CRYPTO_TRADING_FEE, TRADING_FEE
except ImportError as exc:  # pragma: no cover - should not happen in repo context
    raise ImportError("Unable to import trading fee constants from loss_utils.py") from exc

try:
    from src.fixtures import crypto_symbols as FIXTURE_CRYPTO_SYMBOLS
except ImportError:
    FIXTURE_CRYPTO_SYMBOLS: List[str] = []


@dataclass
class AssetRecord:
    symbol: str
    asset_class: str
    default_trading_fee: float
    latest_date: Optional[str]
    total_rows: Optional[int]
    train_rows: Optional[int]
    test_rows: Optional[int]
    source_file: Optional[str]


def _normalise_symbol(symbol: str) -> str:
    return symbol.replace("-", "").replace("/", "").upper()


def _build_crypto_reference(symbols: Iterable[str]) -> Set[str]:
    normalised: Set[str] = set()
    for sym in symbols:
        normalised.add(_normalise_symbol(sym))
    return normalised


def classify_symbol(symbol: str, fixture_lookup: Set[str]) -> str:
    sym_upper = symbol.upper()
    norm = _normalise_symbol(sym_upper)
    if norm in fixture_lookup:
        return "crypto"
    if sym_upper.endswith("-USD") or sym_upper.endswith("/USD"):
        return "crypto"
    if sym_upper.endswith("USD") and not sym_upper.isalpha():
        return "crypto"
    return "equity"


def build_metadata(summary_path: Path) -> Dict[str, AssetRecord]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file '{summary_path}' does not exist")

    df = pd.read_csv(summary_path)
    fixture_lookup = _build_crypto_reference(FIXTURE_CRYPTO_SYMBOLS)

    metadata: Dict[str, AssetRecord] = {}
    for _, row in df.iterrows():
        symbol = str(row["symbol"]).upper()
        asset_class = classify_symbol(symbol, fixture_lookup)
        fee = CRYPTO_TRADING_FEE if asset_class == "crypto" else TRADING_FEE

        latest_date_raw = row.get("latest_date")
        latest_date = None
        if isinstance(latest_date_raw, str) and latest_date_raw:
            try:
                latest_date = datetime.fromisoformat(latest_date_raw).date().isoformat()
            except ValueError:
                latest_date = latest_date_raw

        source_file = None
        for candidate in ("source_file", "train_file"):
            value = row.get(candidate)
            if isinstance(value, str) and value:
                source_file = value
                break

        record = AssetRecord(
            symbol=symbol,
            asset_class=asset_class,
            default_trading_fee=float(fee),
            latest_date=latest_date,
            total_rows=int(row["total_rows"]) if not pd.isna(row.get("total_rows")) else None,
            train_rows=int(row["train_rows"]) if not pd.isna(row.get("train_rows")) else None,
            test_rows=int(row["test_rows"]) if not pd.isna(row.get("test_rows")) else None,
            source_file=source_file,
        )
        metadata[symbol] = record

    return metadata


def write_metadata(metadata: Dict[str, AssetRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {symbol: asdict(record) for symbol, record in sorted(metadata.items())}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build asset metadata for training.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "trainingdata" / "data_summary.csv",
        help="Path to training data summary CSV (default: trainingdata/data_summary.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "trainingdata" / "asset_metadata.json",
        help="Where to write the metadata JSON (default: trainingdata/asset_metadata.json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    metadata = build_metadata(args.summary)
    write_metadata(metadata, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
