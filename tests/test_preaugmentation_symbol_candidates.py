from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.preaug.runtime import PreAugmentationSelector, candidate_preaug_symbols


def test_candidate_preaug_symbols_strips_suffix_and_maps_stable_quotes() -> None:
    candidates = candidate_preaug_symbols("BTCFDUSD__20260206T090000Z__0")
    assert candidates[0] == "BTCFDUSD"
    assert "BTCUSD" in candidates
    assert "BTCUSDT" in candidates


def test_candidate_preaug_symbols_adds_hyphenated_usd_variant() -> None:
    candidates = candidate_preaug_symbols("ADAFDUSD")
    assert "ADAUSD" in candidates
    assert "ADA-USD" in candidates


def test_candidate_preaug_symbols_does_not_map_short_u_suffix_stock() -> None:
    # Guardrail: do not treat every trailing "U" as a quote (MU -> MUSD would be wrong).
    assert "MUSD" not in candidate_preaug_symbols("MU")


def _write_preaug_config(path: Path, *, best_strategy: str) -> None:
    payload = {
        "symbol": path.stem,
        "best_strategy": best_strategy,
        "config": {"name": best_strategy, "params": {}},
        "timestamp": "2026-02-06T00:00:00Z",
        "selection_metric": "mae_percent",
        "selection_value": 1.0,
        "comparison": {
            best_strategy: {"mae_percent": 1.0},
            "baseline": {"mae_percent": 2.0},
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_preaug_selector_prefers_exact_symbol_before_proxy(tmp_path: Path) -> None:
    # Simulate a BTCFDUSD-specific config + a BTCUSD proxy config.
    _write_preaug_config(tmp_path / "BTCFDUSD.json", best_strategy="rolling_norm")
    _write_preaug_config(tmp_path / "BTCUSD.json", best_strategy="differencing")

    selector = PreAugmentationSelector(best_dirs=[tmp_path])
    chosen = None
    for candidate in candidate_preaug_symbols("BTCFDUSD__anything__0"):
        chosen = selector.get_choice(candidate)
        if chosen is not None:
            break

    assert chosen is not None
    assert chosen.symbol == "BTCFDUSD"
    assert chosen.strategy == "rolling_norm"


def test_preaug_selector_uses_proxy_when_exact_missing(tmp_path: Path) -> None:
    _write_preaug_config(tmp_path / "BTCUSD.json", best_strategy="differencing")

    selector = PreAugmentationSelector(best_dirs=[tmp_path])
    chosen = None
    for candidate in candidate_preaug_symbols("BTCFDUSD__anything__0"):
        chosen = selector.get_choice(candidate)
        if chosen is not None:
            break

    assert chosen is not None
    assert chosen.symbol == "BTCUSD"
    assert chosen.strategy == "differencing"

