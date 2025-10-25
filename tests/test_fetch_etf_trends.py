from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import scripts.fetch_etf_trends as trends


class _DummyResponse:
    def __init__(self, *, text: str | None = None, payload: Dict[str, Any] | None = None):
        self.text = text or ""
        self._payload = payload or {}

    def raise_for_status(self) -> None:  # noqa: D401 - simple stub
        """Do nothing."""

    def json(self) -> Dict[str, Any]:
        if self._payload is None:
            raise ValueError("No payload provided")
        return json.loads(json.dumps(self._payload))


def test_fetch_prices_prefers_fallback(monkeypatch):
    def faux_stooq(symbol: str, days: int):  # noqa: ARG001 - signature for compatibility
        raise trends.PriceSourceError("Not enough price data")

    sample_rows = [
        (datetime(2025, 1, 1, tzinfo=timezone.utc), 100.0),
        (datetime(2025, 1, 2, tzinfo=timezone.utc), 101.5),
    ]

    def faux_yahoo(symbol: str, days: int):  # noqa: ARG001 - signature for compatibility
        return sample_rows

    monkeypatch.setattr(trends, "fetch_prices_stooq", faux_stooq)
    monkeypatch.setattr(trends, "fetch_prices_yahoo", faux_yahoo)

    provider, rows, latency = trends.fetch_prices("QQQ", 5, ["stooq", "yahoo"])

    assert provider == "yahoo"
    assert rows == sample_rows
    assert latency >= 0.0


def test_fetch_prices_yahoo_parses_response(monkeypatch):
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1730419200, 1730505600, 1730592000],
                    "indicators": {
                        "quote": [
                            {
                                "close": [410.0, None, 412.5],
                            }
                        ]
                    },
                }
            ]
        }
    }

    def faux_get(url: str, *args: Any, **kwargs: Any):  # noqa: ANN001 - match requests.get
        assert "QQQ" in url
        return _DummyResponse(payload=payload)

    monkeypatch.setattr(trends.requests, "get", faux_get)

    rows = trends.fetch_prices_yahoo("QQQ", 3)

    assert len(rows) == 2
    dates = [row[0] for row in rows]
    assert dates[0] == datetime.fromtimestamp(1730419200, tz=timezone.utc)
    closes = [row[1] for row in rows]
    assert closes == [410.0, 412.5]


def test_update_summary_records_provider(tmp_path):
    summary_path = tmp_path / "summary.json"
    metrics = {"QQQ": {"latest": 10.0, "pnl": 1.0, "sma": 9.0, "std": 0.0, "observations": 2, "pct_change": 0.1}}
    providers = {"QQQ": "yahoo"}

    trends.update_summary(summary_path, metrics, providers)

    payload = json.loads(summary_path.read_text())
    assert payload["QQQ"]["provider"] == "yahoo"


def test_main_appends_provider_log(monkeypatch, tmp_path):
    summary_path = tmp_path / "trend_summary.json"
    provider_log = tmp_path / "providers.csv"
    latency_log = tmp_path / "latency.csv"
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("QQQ\n", encoding="utf-8")

    sample_rows = [
        (datetime(2025, 1, 1, tzinfo=timezone.utc), 100.0),
        (datetime(2025, 1, 2, tzinfo=timezone.utc), 101.0),
    ]

    def faux_fetch(symbol: str, days: int, providers):  # noqa: ANN001 - match signature
        assert providers == ["yahoo"]
        return "yahoo", sample_rows, 0.05

    monkeypatch.setattr(trends, "fetch_prices", faux_fetch)

    argv = [
        "fetch_etf_trends.py",
        "--symbols-file",
        str(symbols_file),
        "--days",
        "10",
        "--summary-path",
        str(summary_path),
        "--providers",
        "yahoo",
        "--provider-log",
        str(provider_log),
        "--latency-log",
        str(latency_log),
    ]

    monkeypatch.setattr(sys, "argv", argv)

    trends.main()

    content = provider_log.read_text(encoding="utf-8").splitlines()
    assert content[0] == "timestamp,provider,count"
    assert content[1].endswith(",yahoo,1")

    latency_lines = latency_log.read_text(encoding="utf-8").splitlines()
    assert latency_lines[0] == "timestamp,symbol,provider,latency_ms"
    assert latency_lines[1].split(",")[2] == "yahoo"
