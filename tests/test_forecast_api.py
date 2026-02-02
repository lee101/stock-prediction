from __future__ import annotations

import csv
from typing import Dict, Iterable

from fastapi.testclient import TestClient

from src.crypto_loop import crypto_order_loop_server as server


def _write_predictions(path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "instrument": "AAPL",
            "close_predicted_price": 202.9,
            "high_predicted_price": 210.24,
            "low_predicted_price": 200.72,
            "entry_takeprofit_profit": 0.0362,
            "entry_takeprofit_high_price": 210.5,
            "entry_takeprofit_low_price": 199.1,
            "maxdiffprofit_profit": 0.0469,
            "maxdiffprofit_high_price": 211.3,
            "maxdiffprofit_low_price": 198.2,
            "takeprofit_profit": 0.0109,
            "takeprofit_high_price": 206.2,
            "takeprofit_low_price": 201.5,
            "generated_at": "2025-02-21T02:53:14.669196+00:00",
        },
        {
            "instrument": "MSFT",
            "close_predicted_price": 391.0,
            "high_predicted_price": 395.0,
            "low_predicted_price": 388.0,
            "entry_takeprofit_profit": 0.005,
            "maxdiffprofit_profit": 0.006,
            "takeprofit_profit": 0.004,
            "generated_at": "2025-02-21T02:53:14.669196+00:00",
        },
    ]


def test_forecast_endpoints_latest_and_prices(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"
    _write_predictions(predictions_file, _sample_rows())

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))

    client = TestClient(server.app)

    latest_resp = client.get("/api/v1/forecasts/latest")
    assert latest_resp.status_code == 200
    latest_payload = latest_resp.json()
    assert latest_payload["count"] == 2
    assert latest_payload["generated_at"] == "2025-02-21T02:53:14.669196+00:00"
    symbols = {entry["symbol"] for entry in latest_payload["forecasts"]}
    assert symbols == {"AAPL", "MSFT"}

    prices_resp = client.get("/api/v1/forecasts/prices?symbols=MSFT")
    assert prices_resp.status_code == 200
    prices_payload = prices_resp.json()
    assert prices_payload["count"] == 1
    assert prices_payload["prices"][0]["symbol"] == "MSFT"
    assert prices_payload["prices"][0]["predicted"]["close"] == 391.0


def test_bot_forecasts_recommendations(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"
    _write_predictions(predictions_file, _sample_rows())

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))

    client = TestClient(server.app)
    resp = client.get("/api/v1/bot/forecasts?min_profit=0.02")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["buy_list"] == ["AAPL"]
    assert payload["count"] == 2

    aapl = next(item for item in payload["forecasts"] if item["symbol"] == "AAPL")
    assert aapl["recommendation"] == "BUY"
    assert aapl["strategy"] == "maxdiffprofit"
    assert aapl["price_targets"]["low"] == 198.2
    assert aapl["price_targets"]["high"] == 211.3
