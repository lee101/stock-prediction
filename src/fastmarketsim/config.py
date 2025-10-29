from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, MutableMapping

from .module import load_extension

DEFAULTS = {
    "context_len": 128,
    "horizon": 1,
    "mode": "open_close",
    "normalize_returns": True,
    "seed": 1337,
    "trading_fee": 0.0005,
    "crypto_trading_fee": 0.0015,
    "slip_bps": 1.5,
    "annual_leverage_rate": 0.0675,
    "intraday_leverage_max": 4.0,
    "overnight_leverage_max": 2.0,
}


def _as_mapping(cfg: Any) -> MutableMapping[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if cfg is None:
        return dict(DEFAULTS)
    raise TypeError(f"Unsupported config type {type(cfg)!r}; expected dataclass or mapping.")


def build_sim_config(cfg: Any) -> Any:
    """Convert a Python configuration object into a native simulator config."""

    data = _as_mapping(cfg)
    merged = {**DEFAULTS, **data}

    fees = {
        "stock_fee": float(merged.get("trading_fee", DEFAULTS["trading_fee"])),
        "crypto_fee": float(merged.get("crypto_trading_fee", DEFAULTS["crypto_trading_fee"])),
        "slip_bps": float(merged.get("slip_bps", DEFAULTS["slip_bps"])),
        "annual_leverage": float(merged.get("annual_leverage_rate", DEFAULTS["annual_leverage_rate"])),
        "intraday_max": float(merged.get("intraday_leverage_max", DEFAULTS["intraday_leverage_max"])),
        "overnight_max": float(merged.get("overnight_leverage_max", DEFAULTS["overnight_leverage_max"])),
    }

    sim_dict = {
        "context_len": int(merged.get("context_len", DEFAULTS["context_len"])),
        "horizon": int(merged.get("horizon", DEFAULTS["horizon"])),
        "mode": merged.get("mode", DEFAULTS["mode"]),
        "normalize_returns": bool(merged.get("normalize_returns", DEFAULTS["normalize_returns"])),
        "seed": int(merged.get("seed", DEFAULTS["seed"])),
        "fees": fees,
    }

    extension = load_extension()
    return extension.sim_config_from_dict(sim_dict)
