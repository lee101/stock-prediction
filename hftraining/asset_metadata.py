"""
Utility helpers for loading asset metadata generated from trainingdata.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from loss_utils import CRYPTO_TRADING_FEE, TRADING_FEE


DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[1] / "trainingdata" / "asset_metadata.json"
ASSET_CLASS_TO_ID = {"equity": 0, "crypto": 1}


@lru_cache(maxsize=1)
def load_asset_metadata(path: Optional[Path] = None) -> Dict[str, Dict[str, object]]:
    metadata_path = Path(path) if path else DEFAULT_METADATA_PATH
    if not metadata_path.exists():
        return {}
    data = json.loads(metadata_path.read_text())
    normalised: Dict[str, Dict[str, object]] = {}
    for symbol, record in data.items():
        normalised[symbol.upper()] = record
    return normalised


def get_asset_record(symbol: str, path: Optional[Path] = None) -> Dict[str, object]:
    metadata = load_asset_metadata(path)
    return metadata.get(symbol.upper(), {})


def get_asset_class(symbol: str, default: str = "equity") -> str:
    record = get_asset_record(symbol)
    return str(record.get("asset_class", default))


def get_trading_fee(symbol: str) -> float:
    record = get_asset_record(symbol)
    if "default_trading_fee" in record:
        return float(record["default_trading_fee"])
    asset_class = get_asset_class(symbol)
    return float(CRYPTO_TRADING_FEE if asset_class == "crypto" else TRADING_FEE)


def get_asset_class_id(symbol: str) -> int:
    asset_class = get_asset_class(symbol)
    return ASSET_CLASS_TO_ID.get(asset_class.lower(), 0)
