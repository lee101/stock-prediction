from __future__ import annotations

import json
from pathlib import Path

from chronos2_trainer import _resolve_preaug_choice


def _write_choice(path: Path, *, symbol: str, best_strategy: str) -> None:
    payload = {
        "symbol": symbol,
        "best_strategy": best_strategy,
        "selection_metric": "mae_percent",
        "selection_value": 0.1,
        "comparison": {
            best_strategy: {"mae_percent": 0.1},
        },
        "config": {"name": best_strategy, "params": {}},
    }
    path.write_text(json.dumps(payload))


def test_resolve_preaug_choice_prefers_first_dir(tmp_path: Path) -> None:
    hourly_dir = tmp_path / "hourly"
    base_dir = tmp_path / "base"
    hourly_dir.mkdir()
    base_dir.mkdir()

    symbol = "FOOUSD"
    _write_choice(hourly_dir / f"{symbol}.json", symbol=symbol, best_strategy="differencing")
    _write_choice(base_dir / f"{symbol}.json", symbol=symbol, best_strategy="rolling_norm")

    choice, source = _resolve_preaug_choice(symbol, None, (hourly_dir, base_dir))
    assert choice is not None
    assert choice.strategy == "differencing"
    assert source is not None
    assert source.endswith(f"{hourly_dir}/{symbol}.json")


def test_resolve_preaug_choice_maps_usdt_to_usd(tmp_path: Path) -> None:
    hourly_dir = tmp_path / "hourly"
    hourly_dir.mkdir()

    _write_choice(hourly_dir / "BARUSD.json", symbol="BARUSD", best_strategy="detrending")

    choice, source = _resolve_preaug_choice("BARUSDT", None, (hourly_dir,))
    assert choice is not None
    assert choice.symbol == "BARUSD"
    assert choice.strategy == "detrending"
    assert source is not None
    assert source.endswith("BARUSD.json")

