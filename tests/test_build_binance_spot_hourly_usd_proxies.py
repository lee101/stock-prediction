from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import build_binance_spot_hourly_usd_proxies as module


def test_main_writes_hourly_usd_proxy_and_updates_symbol_column(tmp_path: Path) -> None:
    hourly_root = tmp_path / "hourly"
    hourly_root.mkdir()
    source_path = hourly_root / "SUIUSDT.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-03-17T00:00:00Z",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 1000.0,
                "symbol": "SUIUSDT",
            }
        ]
    ).to_csv(source_path, index=False)

    rc = module.main(
        [
            "--hourly-root",
            str(hourly_root),
            "--symbols",
            "SUIUSDT",
        ]
    )

    assert rc == 0
    proxy_path = hourly_root / "SUIUSD.csv"
    assert proxy_path.exists()
    proxy_frame = pd.read_csv(proxy_path)
    assert list(proxy_frame["symbol"]) == ["SUIUSD"]


def test_main_skips_when_symbol_is_already_usd_proxy(tmp_path: Path) -> None:
    hourly_root = tmp_path / "hourly"
    hourly_root.mkdir()
    source_path = hourly_root / "BTCUSD.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-03-17T00:00:00Z",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 1000.0,
                "symbol": "BTCUSD",
            }
        ]
    ).to_csv(source_path, index=False)

    rc = module.main(
        [
            "--hourly-root",
            str(hourly_root),
            "--symbols",
            "BTCUSD",
        ]
    )

    assert rc == 0
    assert source_path.exists()
    assert len(list(hourly_root.glob("*.csv"))) == 1
