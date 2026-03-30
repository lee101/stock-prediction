"""Utilities for assembling OHLC percent-change data (stateless agent)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from stockagent.agentsimulator.local_market_data import (
    USE_DEFAULT,
    build_market_data_bundle as _build_market_data_bundle,
    resolve_local_data_dirs as _shared_resolve_local_data_dirs,
)
from stockagent.agentsimulator.market_data_bundle import MarketDataBundle

from ..constants import DEFAULT_SYMBOLS

FALLBACK_DATA_DIRS = [
    Path("trainingdata/stockagent/marketdata"),
    Path("stockagentindependant_market_data"),
    Path("stockagent_market_data"),
    Path("trainingdata/marketdata"),
]

def fetch_latest_ohlc(
    symbols: Optional[Iterable[str]] = None,
    lookback_days: int = 60,
    as_of: Optional[datetime] = None,
    local_data_dir: Optional[Path] | object = USE_DEFAULT,
    allow_remote_download: bool = False,
    use_fallback_data_dirs: bool | object = USE_DEFAULT,
) -> MarketDataBundle:
    as_of = as_of or datetime.now(timezone.utc)

    return _build_market_data_bundle(
        symbols=symbols,
        default_symbols=DEFAULT_SYMBOLS,
        lookback_days=lookback_days,
        as_of=as_of,
        local_data_dir=local_data_dir,
        fallback_data_dirs=FALLBACK_DATA_DIRS,
        use_fallback_data_dirs=use_fallback_data_dirs,
        remote_loader=None,
    )


def resolve_local_data_dirs(
    *,
    local_data_dir: Optional[Path] | object = USE_DEFAULT,
    use_fallback_data_dirs: bool | object = USE_DEFAULT,
) -> List[Path]:
    return _shared_resolve_local_data_dirs(
        local_data_dir=local_data_dir,
        fallback_data_dirs=FALLBACK_DATA_DIRS,
        use_fallback_data_dirs=use_fallback_data_dirs,
    )
