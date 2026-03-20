"""Symbol ownership config for multi-service conflict prevention.

Three services share one Alpaca account.  Rather than file-level locking
(which risks deadlocks), we use a static ownership config that assigns each
symbol exclusively to one service.  Each service reads its own symbol set
from this module; overlaps are caught at startup and at test time.

Usage in a service::

    from unified_orchestrator.symbol_lock import load_service_symbols, warn_position_conflicts

    crypto, stocks = load_service_symbols("unified-orchestrator", alpaca_api)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

_CONFIG_PATH = Path(__file__).resolve().parent / "service_config.json"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[Path] = None) -> dict:
    """Load the service symbol config from JSON.  Returns empty dict on error."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        logger.warning("service_config.json not found at {}; using empty config", path)
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data
    except Exception as exc:
        logger.error("Failed to load service_config.json ({}): {}", path, exc)
        return {}


def load_service_symbols(
    service_name: str,
    config_path: Optional[Path] = None,
) -> tuple[list[str], list[str]]:
    """Return (crypto_symbols, stock_symbols) assigned to *service_name*.

    Falls back to empty lists when the config is missing or the service key is
    absent, so callers can safely fall through to their hardcoded defaults.
    """
    cfg = load_config(config_path)
    svc = cfg.get(service_name, {})
    crypto = [s.strip().upper() for s in svc.get("crypto_symbols", [])]
    stocks = [s.strip().upper() for s in svc.get("stock_symbols", [])]
    return crypto, stocks


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def get_all_symbols_by_service(config_path: Optional[Path] = None) -> dict[str, set[str]]:
    """Return {service_name: set_of_all_symbols} for all services in config."""
    cfg = load_config(config_path)
    result: dict[str, set[str]] = {}
    for svc_name, svc_cfg in cfg.items():
        all_syms = set()
        all_syms.update(s.strip().upper() for s in svc_cfg.get("crypto_symbols", []))
        all_syms.update(s.strip().upper() for s in svc_cfg.get("stock_symbols", []))
        result[svc_name] = all_syms
    return result


def find_symbol_overlaps(config_path: Optional[Path] = None) -> dict[str, list[str]]:
    """Return {symbol: [service1, service2, ...]} for any overlapping symbols.

    An empty dict means no overlaps — all symbol sets are disjoint.
    """
    by_service = get_all_symbols_by_service(config_path)
    services = list(by_service.keys())
    overlaps: dict[str, list[str]] = {}
    for i, svc_a in enumerate(services):
        for svc_b in services[i + 1:]:
            shared = by_service[svc_a] & by_service[svc_b]
            for sym in sorted(shared):
                overlaps.setdefault(sym, [])
                if svc_a not in overlaps[sym]:
                    overlaps[sym].append(svc_a)
                if svc_b not in overlaps[sym]:
                    overlaps[sym].append(svc_b)
    return overlaps


def assert_no_overlaps(config_path: Optional[Path] = None) -> None:
    """Raise ValueError if any symbol is assigned to more than one service."""
    overlaps = find_symbol_overlaps(config_path)
    if overlaps:
        details = "; ".join(
            f"{sym} owned by {', '.join(svcs)}" for sym, svcs in sorted(overlaps.items())
        )
        raise ValueError(f"Symbol ownership conflict in service_config.json: {details}")


# ---------------------------------------------------------------------------
# Live position conflict check
# ---------------------------------------------------------------------------

def warn_position_conflicts(service_name: str, alpaca_api, config_path: Optional[Path] = None) -> None:
    """Log a WARNING for any open Alpaca position that belongs to another service.

    Call this at service startup so operators are alerted immediately if two
    services have accidentally traded the same symbol.

    Args:
        service_name: The name key of this service in service_config.json.
        alpaca_api:   An object with a ``get_all_positions()`` method (or
                      ``list_positions()``); typically ``alpaca_wrapper`` or
                      an ``alpaca.trading.TradingClient`` instance.
        config_path:  Override path to service_config.json (for testing).
    """
    # Resolve our own symbols and the other services' symbols.
    by_service = get_all_symbols_by_service(config_path)
    my_symbols = by_service.get(service_name, set())
    other_symbols: dict[str, set[str]] = {
        name: syms for name, syms in by_service.items() if name != service_name
    }

    # Fetch open positions.
    try:
        positions = (
            alpaca_api.get_all_positions()
            if hasattr(alpaca_api, "get_all_positions")
            else alpaca_api.list_positions()
        )
    except Exception as exc:
        logger.warning("warn_position_conflicts: could not fetch positions: {}", exc)
        return

    for pos in positions:
        sym = str(getattr(pos, "symbol", "") or "").upper()
        if not sym:
            continue
        for other_svc, other_syms in other_symbols.items():
            if sym in other_syms:
                qty = getattr(pos, "qty", "?")
                mv = getattr(pos, "market_value", "?")
                logger.warning(
                    "CONFLICT: open position {} (qty={}, mv={}) belongs to {} not {}",
                    sym, qty, mv, other_svc, service_name,
                )
