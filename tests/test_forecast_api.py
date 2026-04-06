from __future__ import annotations

import csv
import importlib
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable

from fastapi.testclient import TestClient

from src.crypto_loop import crypto_order_loop_server as server


def test_root_crypto_order_loop_server_delegates_to_src_module() -> None:
    import sys

    sys.modules.pop("crypto_loop.crypto_order_loop_server", None)
    sys.modules.pop("src.crypto_loop.crypto_order_loop_server", None)

    root_module = importlib.import_module("crypto_loop.crypto_order_loop_server")
    src_module = importlib.import_module("src.crypto_loop.crypto_order_loop_server")

    assert root_module is src_module
    assert root_module.app is src_module.app
    assert root_module.forecasts_latest is src_module.forecasts_latest
    assert root_module.bot_forecasts is src_module.bot_forecasts


def test_stock_order_rejects_unknown_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ensure_crypto_order_loop_started", lambda: None)
    monkeypatch.setattr(server, "stop_crypto_order_loop", lambda timeout=1.0: None)
    monkeypatch.setattr(server, "crypto_symbol_to_order", {})

    with TestClient(server.app) as client:
        response = client.post(
            "/api/v1/stock_order",
            json={"symbol": "DOGEUSD", "side": "buy", "price": 1.0, "qty": 1.0},
        )

    assert response.status_code == 422
    assert "unsupported symbol" in response.text


def test_stock_order_rejects_invalid_side_and_non_positive_values(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ensure_crypto_order_loop_started", lambda: None)
    monkeypatch.setattr(server, "stop_crypto_order_loop", lambda timeout=1.0: None)
    monkeypatch.setattr(server, "crypto_symbol_to_order", {})

    with TestClient(server.app) as client:
        invalid_side = client.post(
            "/api/v1/stock_order",
            json={"symbol": "BTCUSD", "side": "hold", "price": 1.0, "qty": 1.0},
        )
        invalid_qty = client.post(
            "/api/v1/stock_order",
            json={"symbol": "BTCUSD", "side": "buy", "price": 1.0, "qty": 0.0},
        )

    assert invalid_side.status_code == 422
    assert invalid_qty.status_code == 422


def test_stock_order_normalizes_valid_input_and_mirrors_bitcoin_order(tmp_path, monkeypatch):
    queued_orders: dict[str, object] = {}
    mirrored: list[tuple[str, str, float]] = []
    cancel_calls: list[str] = []

    monkeypatch.setattr(server, "ensure_crypto_order_loop_started", lambda: None)
    monkeypatch.setattr(server, "stop_crypto_order_loop", lambda timeout=1.0: None)
    monkeypatch.setattr(server, "crypto_symbol_to_order", queued_orders)
    monkeypatch.setattr(
        server.binance_wrapper,
        "cancel_all_orders",
        lambda: (_ for _ in ()).throw(AssertionError("stock_order should use broker adapter")),
    )
    monkeypatch.setattr(
        server.binance_wrapper,
        "create_all_in_order",
        lambda *_args: (_ for _ in ()).throw(AssertionError("stock_order should use broker adapter")),
    )
    monkeypatch.setattr(
        server,
        "_load_crypto_order_broker_api",
        lambda: server.CryptoOrderBrokerApi(
            submit_price_order=lambda *_args: None,
            cancel_all_binance_orders=lambda: cancel_calls.append("cancelled"),
            mirror_all_in_order=lambda symbol, side, price: mirrored.append((symbol, side, price)),
        ),
    )

    with TestClient(server.app) as client:
        response = client.post(
            "/api/v1/stock_order",
            json={"symbol": "btc/usd", "side": "BUY", "price": 70000.0, "qty": 0.05},
        )

    assert response.status_code == 200
    assert queued_orders["BTCUSD"]["symbol"] == "BTCUSD"
    assert queued_orders["BTCUSD"]["side"] == "buy"
    assert queued_orders["BTCUSD"]["price"] == 70000.0
    assert queued_orders["BTCUSD"]["qty"] == 0.05
    assert queued_orders["BTCUSD"]["created_at"].endswith("+00:00")
    assert cancel_calls == ["cancelled"]
    assert mirrored == [("BTCUSDT", "BUY", 70000.0)]


def test_stock_orders_endpoint_returns_snapshot_of_queued_orders(tmp_path, monkeypatch):
    queued_orders = {
        "BTCUSD": {"symbol": "BTCUSD", "side": "buy", "price": 70000.0, "qty": 0.05},
        "ETHUSD": None,
    }

    monkeypatch.setattr(server, "ensure_crypto_order_loop_started", lambda: None)
    monkeypatch.setattr(server, "stop_crypto_order_loop", lambda timeout=1.0: None)
    monkeypatch.setattr(server, "crypto_symbol_to_order", queued_orders)

    with TestClient(server.app) as client:
        response = client.get("/api/v1/stock_orders")

    assert response.status_code == 200
    assert response.json() == {
        "BTCUSD": {"symbol": "BTCUSD", "side": "buy", "price": 70000.0, "qty": 0.05}
    }


def test_stock_order_emits_structured_log(tmp_path, monkeypatch):
    queued_orders: dict[str, object] = {}
    order_logs: list[dict[str, object]] = []

    monkeypatch.setattr(server, "ensure_crypto_order_loop_started", lambda: None)
    monkeypatch.setattr(server, "stop_crypto_order_loop", lambda timeout=1.0: None)
    monkeypatch.setattr(server, "crypto_symbol_to_order", queued_orders)
    monkeypatch.setattr(server, "logger", SimpleNamespace(info=lambda _message, payload: order_logs.append(payload)))
    monkeypatch.setattr(
        server,
        "_load_crypto_order_broker_api",
        lambda: server.CryptoOrderBrokerApi(
            submit_price_order=lambda *_args: None,
            cancel_all_binance_orders=lambda: None,
            mirror_all_in_order=lambda *_args: None,
        ),
    )

    with TestClient(server.app) as client:
        response = client.post(
            "/api/v1/stock_order",
            json={"symbol": "BTCUSD", "side": "buy", "price": 70000.0, "qty": 0.05},
        )

    assert response.status_code == 200
    assert order_logs == [
        {
            "symbol": "BTCUSD",
            "side": "buy",
            "price": 70000.0,
            "qty": 0.05,
            "created_at": queued_orders["BTCUSD"]["created_at"],
            "mirrored_to_binance": True,
            "mirror_symbol": "BTCUSDT",
        }
    ]


def test_stock_order_does_not_mirror_non_configured_symbol(tmp_path, monkeypatch):
    queued_orders: dict[str, object] = {}
    mirrored: list[tuple[str, str, float]] = []
    cancel_calls: list[str] = []

    monkeypatch.setattr(server, "ensure_crypto_order_loop_started", lambda: None)
    monkeypatch.setattr(server, "stop_crypto_order_loop", lambda timeout=1.0: None)
    monkeypatch.setattr(server, "crypto_symbol_to_order", queued_orders)
    monkeypatch.setattr(
        server,
        "_load_crypto_order_broker_api",
        lambda: server.CryptoOrderBrokerApi(
            submit_price_order=lambda *_args: None,
            cancel_all_binance_orders=lambda: cancel_calls.append("cancelled"),
            mirror_all_in_order=lambda symbol, side, price: mirrored.append((symbol, side, price)),
        ),
    )

    with TestClient(server.app) as client:
        response = client.post(
            "/api/v1/stock_order",
            json={"symbol": "ETHUSD", "side": "buy", "price": 3000.0, "qty": 0.5},
        )

    assert response.status_code == 200
    assert queued_orders["ETHUSD"]["symbol"] == "ETHUSD"
    assert cancel_calls == []
    assert mirrored == []


def test_pop_queued_order_clears_claimed_order_without_touching_replacement(tmp_path, monkeypatch):
    queued_orders: dict[str, object] = {
        "BTCUSD": {"symbol": "BTCUSD", "side": "buy", "price": 70000.0, "qty": 0.05},
    }

    monkeypatch.setattr(server, "crypto_symbol_to_order", queued_orders)

    claimed_order = server._pop_queued_order("BTCUSD")
    server._set_queued_order("BTCUSD", {"symbol": "BTCUSD", "side": "sell", "price": 71000.0, "qty": 0.04})

    assert claimed_order == {"symbol": "BTCUSD", "side": "buy", "price": 70000.0, "qty": 0.05}
    assert server._get_queued_order("BTCUSD") == {
        "symbol": "BTCUSD",
        "side": "sell",
        "price": 71000.0,
        "qty": 0.04,
    }


def test_pop_queued_order_discards_malformed_payload(monkeypatch):
    queued_orders: dict[str, object] = {
        "BTCUSD": {"symbol": "BTCUSD", "side": "hold", "price": 70000.0, "qty": 0.05},
    }

    monkeypatch.setattr(server, "crypto_symbol_to_order", queued_orders)

    assert server._pop_queued_order("BTCUSD") is None
    assert "BTCUSD" not in queued_orders


def test_process_queued_order_requeues_failed_submission_when_slot_is_empty(monkeypatch):
    queued_orders: dict[str, object] = {
        "BTCUSD": {"symbol": "BTCUSD", "side": "buy", "price": 70000.0, "qty": 0.05},
    }

    monkeypatch.setattr(server, "crypto_symbol_to_order", queued_orders)
    broker_api = server.CryptoOrderBrokerApi(
        submit_price_order=lambda *_args: (_ for _ in ()).throw(RuntimeError("broker down")),
        cancel_all_binance_orders=lambda: None,
        mirror_all_in_order=lambda *_args: None,
    )

    server._process_queued_order("BTCUSD", broker_api)

    assert queued_orders["BTCUSD"] == {
        "symbol": "BTCUSD",
        "side": "buy",
        "price": 70000.0,
        "qty": 0.05,
    }


def test_process_queued_order_keeps_newer_replacement_on_failed_submission(monkeypatch):
    queued_orders: dict[str, object] = {
        "BTCUSD": {"symbol": "BTCUSD", "side": "buy", "price": 70000.0, "qty": 0.05},
    }
    replacement_order = {"symbol": "BTCUSD", "side": "sell", "price": 71000.0, "qty": 0.04}

    def _failing_submit(*_args):
        server._set_queued_order("BTCUSD", replacement_order)
        raise RuntimeError("broker down")

    monkeypatch.setattr(server, "crypto_symbol_to_order", queued_orders)
    broker_api = server.CryptoOrderBrokerApi(
        submit_price_order=_failing_submit,
        cancel_all_binance_orders=lambda: None,
        mirror_all_in_order=lambda *_args: None,
    )

    server._process_queued_order("BTCUSD", broker_api)

    assert queued_orders["BTCUSD"] == replacement_order


def test_ensure_crypto_order_loop_reuses_unwinding_thread_without_returning_doomed_worker(
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(server, "symbols", ["BTCUSD"])
    monkeypatch.setattr(
        server,
        "_load_crypto_order_broker_api",
        lambda: server.CryptoOrderBrokerApi(
            submit_price_order=lambda *_args: None,
            cancel_all_binance_orders=lambda: None,
            mirror_all_in_order=lambda *_args: None,
        ),
    )

    def _blocking_process(_symbol: str, _broker_api: server.CryptoOrderBrokerApi) -> None:
        entered.set()
        release.wait(timeout=5.0)

    monkeypatch.setattr(server, "_process_queued_order", _blocking_process)

    server.stop_crypto_order_loop(timeout=1.0)
    try:
        thread_one = server.ensure_crypto_order_loop_started()
        assert thread_one is not None
        assert entered.wait(timeout=1.0)

        server.stop_crypto_order_loop(timeout=0.01)
        assert thread_one.is_alive()
        assert server._thread_stop_event.is_set()

        thread_two = server.ensure_crypto_order_loop_started()

        assert thread_two is thread_one
        assert not server._thread_stop_event.is_set()

        release.set()
        assert thread_one.is_alive()
    finally:
        release.set()
        server.stop_crypto_order_loop(timeout=1.0)


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
    os.utime(predictions_file, (100, 100))
    forecast_logs: list[dict[str, object]] = []

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))
    monkeypatch.setattr(
        server,
        "_forecast_metadata_now",
        lambda: datetime.fromtimestamp(200, tz=timezone.utc),
    )
    monkeypatch.setattr(
        server,
        "logger",
        SimpleNamespace(info=lambda _message, payload: forecast_logs.append(payload), warning=lambda *_args: None),
    )
    server.stop_crypto_order_loop()

    with TestClient(server.app) as client:
        latest_resp = client.get("/api/v1/forecasts/latest")
        assert latest_resp.status_code == 200
        latest_payload = latest_resp.json()
        assert latest_payload["count"] == 2
        assert latest_payload["generated_at"] == "2025-02-21T02:53:14.669196+00:00"
        assert latest_payload["source_file"] == str(predictions_file)
        assert latest_payload["source_filename"] == "predictions.csv"
        assert latest_payload["source_file_updated_at"] == "1970-01-01T00:01:40+00:00"
        assert latest_payload["source_file_age_seconds"] == 100
        assert latest_payload["forecast_source_status"] == "ready"
        assert latest_payload["symbol_query"] == {"applied": False}
        symbols = {entry["symbol"] for entry in latest_payload["forecasts"]}
        assert symbols == {"AAPL", "MSFT"}

        prices_resp = client.get("/api/v1/forecasts/prices?symbols=MSFT")
        assert prices_resp.status_code == 200
        prices_payload = prices_resp.json()
        assert prices_payload["count"] == 1
        assert prices_payload["source_file"] == str(predictions_file)
        assert prices_payload["source_filename"] == "predictions.csv"
        assert prices_payload["source_file_updated_at"] == "1970-01-01T00:01:40+00:00"
        assert prices_payload["source_file_age_seconds"] == 100
        assert prices_payload["forecast_source_status"] == "ready"
        assert prices_payload["symbol_query"] == {
            "applied": True,
            "requested": ["MSFT"],
            "matched": ["MSFT"],
            "missing": [],
        }
        assert prices_payload["prices"][0]["symbol"] == "MSFT"
        assert prices_payload["prices"][0]["predicted"]["close"] == 391.0

    assert forecast_logs[0]["endpoint"] == "/api/v1/forecasts/latest"
    assert forecast_logs[0]["status_code"] == 200
    assert forecast_logs[0]["forecast_source_status"] == "ready"
    assert forecast_logs[0]["count"] == 2
    assert forecast_logs[1] == {
        "endpoint": "/api/v1/forecasts/prices",
        "status_code": 200,
        "forecast_source_status": "ready",
        "source_file": str(predictions_file),
        "source_filename": "predictions.csv",
        "generated_at": "2025-02-21T02:53:14.669196+00:00",
        "source_file_age_seconds": 100,
        "symbol_query": {
            "applied": True,
            "requested": ["MSFT"],
            "matched": ["MSFT"],
            "missing": [],
        },
        "count": 1,
        "error": None,
        "error_detail": None,
        "source_search_paths": None,
    }

    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_bot_forecasts_recommendations(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"
    _write_predictions(predictions_file, _sample_rows())
    os.utime(predictions_file, (100, 100))

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))
    monkeypatch.setattr(
        server,
        "_forecast_metadata_now",
        lambda: datetime.fromtimestamp(250, tz=timezone.utc),
    )
    server.stop_crypto_order_loop()

    with TestClient(server.app) as client:
        resp = client.get("/api/v1/bot/forecasts?min_profit=0.02")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["buy_list"] == ["AAPL"]
        assert payload["count"] == 2
        assert payload["source_filename"] == "predictions.csv"
        assert payload["source_file_updated_at"] == "1970-01-01T00:01:40+00:00"
        assert payload["source_file_age_seconds"] == 150
        assert payload["forecast_source_status"] == "ready"
        assert payload["symbol_query"] == {"applied": False}

        aapl = next(item for item in payload["forecasts"] if item["symbol"] == "AAPL")
        assert aapl["recommendation"] == "BUY"
        assert aapl["strategy"] == "maxdiffprofit"
        assert aapl["price_targets"]["low"] == 198.2
        assert aapl["price_targets"]["high"] == 211.3

    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_forecast_generated_at_falls_back_to_file_mtime_in_utc(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"
    rows = _sample_rows()
    for row in rows:
        row.pop("generated_at", None)
    _write_predictions(predictions_file, rows)
    os.utime(predictions_file, (120, 120))

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))
    monkeypatch.setattr(
        server,
        "_forecast_metadata_now",
        lambda: datetime.fromtimestamp(180, tz=timezone.utc),
    )
    monkeypatch.setattr(server, "_forecast_cache_entry", None)
    server.stop_crypto_order_loop()

    with TestClient(server.app) as client:
        payload = client.get("/api/v1/forecasts/latest").json()
        assert payload["generated_at"] == "1970-01-01T00:02:00+00:00"
        assert payload["source_file_updated_at"] == "1970-01-01T00:02:00+00:00"
        assert payload["source_file_age_seconds"] == 60
        assert payload["forecast_source_status"] == "ready"
        assert payload["symbol_query"] == {"applied": False}

    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_forecast_metadata_tolerates_missing_file_after_cache_load(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"
    _write_predictions(predictions_file, _sample_rows())
    os.utime(predictions_file, (120, 120))

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))
    monkeypatch.setattr(server, "_forecast_cache_entry", None)
    server.stop_crypto_order_loop()

    original_builder = server._build_forecast_response_metadata

    def _vanishing_metadata(source_file: str | None, generated_at: str | None) -> dict[str, object]:
        predictions_file.unlink(missing_ok=True)
        return original_builder(source_file, generated_at)

    monkeypatch.setattr(server, "_build_forecast_response_metadata", _vanishing_metadata)

    with TestClient(server.app) as client:
        payload = client.get("/api/v1/forecasts/latest").json()
        assert payload["source_file"] == str(predictions_file)
        assert payload["source_filename"] == "predictions.csv"
        assert payload["source_file_updated_at"] is None
        assert payload["source_file_age_seconds"] is None
        assert payload["forecast_source_status"] == "ready"
        assert payload["symbol_query"] == {"applied": False}
        assert payload["count"] == 2

    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_forecast_endpoints_reuse_cached_predictions_file(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"
    _write_predictions(predictions_file, _sample_rows())

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))
    monkeypatch.setattr(server, "_forecast_cache_entry", None)
    server.stop_crypto_order_loop()

    open_count = 0
    real_open = Path.open

    def _counting_open(self: Path, *args, **kwargs):
        nonlocal open_count
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == predictions_file and "r" in str(mode):
            open_count += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)

    with TestClient(server.app) as client:
        assert client.get("/api/v1/forecasts/latest").status_code == 200
        assert client.get("/api/v1/forecasts/prices?symbols=AAPL").status_code == 200
        assert client.get("/api/v1/bot/forecasts?min_profit=0.02").status_code == 200

    assert open_count == 1
    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_forecast_cache_reloads_when_predictions_file_changes(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"
    _write_predictions(predictions_file, _sample_rows())

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))
    monkeypatch.setattr(server, "_forecast_cache_entry", None)
    server.stop_crypto_order_loop()

    open_count = 0
    real_open = Path.open

    def _counting_open(self: Path, *args, **kwargs):
        nonlocal open_count
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == predictions_file and "r" in str(mode):
            open_count += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)

    with TestClient(server.app) as client:
        first_payload = client.get("/api/v1/forecasts/prices?symbols=AAPL").json()
        assert first_payload["prices"][0]["predicted"]["close"] == 202.9

        updated_rows = _sample_rows()
        updated_rows[0]["close_predicted_price"] = 303.7
        updated_rows.append(
            {
                "instrument": "NVDA",
                "close_predicted_price": 999.0,
                "high_predicted_price": 1005.0,
                "low_predicted_price": 990.0,
                "generated_at": "2025-02-22T02:53:14.669196+00:00",
            }
        )
        _write_predictions(predictions_file, updated_rows)

        second_payload = client.get("/api/v1/forecasts/prices?symbols=AAPL,NVDA").json()
        returned_symbols = {entry["symbol"] for entry in second_payload["prices"]}
        assert returned_symbols == {"AAPL", "NVDA"}
        updated_aapl = next(entry for entry in second_payload["prices"] if entry["symbol"] == "AAPL")
        assert updated_aapl["predicted"]["close"] == 303.7

    assert open_count == 2
    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_forecast_cache_switches_when_newer_prediction_file_appears(tmp_path, monkeypatch):
    older_dir = tmp_path / "older"
    newer_dir = tmp_path / "newer"
    older_dir.mkdir()
    newer_dir.mkdir()

    older_file = older_dir / "predictions-old.csv"
    newer_file = newer_dir / "predictions.csv"
    _write_predictions(older_file, _sample_rows())
    older_rows = _sample_rows()
    older_rows[0]["generated_at"] = "2025-02-20T02:53:14.669196+00:00"
    _write_predictions(older_file, older_rows)

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (older_dir, newer_dir))
    monkeypatch.setattr(server, "_forecast_cache_entry", None)
    monkeypatch.setattr(
        server,
        "_forecast_metadata_now",
        lambda: datetime.fromtimestamp(250, tz=timezone.utc),
    )
    server.stop_crypto_order_loop()

    open_count = 0
    real_open = Path.open

    def _counting_open(self: Path, *args, **kwargs):
        nonlocal open_count
        mode = args[0] if args else kwargs.get("mode", "r")
        if self in {older_file, newer_file} and "r" in str(mode):
            open_count += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _counting_open)

    os.utime(older_file, (100, 100))

    with TestClient(server.app) as client:
        first_payload = client.get("/api/v1/forecasts/latest").json()
        assert first_payload["source_file"] == str(older_file)
        assert first_payload["source_filename"] == "predictions-old.csv"
        assert first_payload["generated_at"] == "2025-02-20T02:53:14.669196+00:00"
        assert first_payload["source_file_updated_at"] == "1970-01-01T00:01:40+00:00"
        assert first_payload["source_file_age_seconds"] == 150

        newer_rows = [
            {
                "instrument": "NVDA",
                "close_predicted_price": 999.0,
                "high_predicted_price": 1005.0,
                "low_predicted_price": 990.0,
                "generated_at": "2025-02-22T02:53:14.669196+00:00",
            }
        ]
        _write_predictions(newer_file, newer_rows)
        os.utime(newer_file, (200, 200))

        second_payload = client.get("/api/v1/forecasts/latest").json()
        assert second_payload["source_file"] == str(newer_file)
        assert second_payload["source_filename"] == "predictions.csv"
        assert second_payload["generated_at"] == "2025-02-22T02:53:14.669196+00:00"
        assert second_payload["source_file_updated_at"] == "1970-01-01T00:03:20+00:00"
        assert second_payload["source_file_age_seconds"] == 50
        assert second_payload["forecast_source_status"] == "ready"
        assert second_payload["symbol_query"] == {"applied": False}
        assert second_payload["count"] == 1
        assert second_payload["forecasts"][0]["symbol"] == "NVDA"

    assert open_count == 2
    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_forecast_symbol_query_reports_missing_symbols_and_preserves_requested_order(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"
    _write_predictions(predictions_file, _sample_rows())

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))
    monkeypatch.setattr(server, "_forecast_cache_entry", None)
    server.stop_crypto_order_loop()

    with TestClient(server.app) as client:
        payload = client.get("/api/v1/forecasts/prices?symbols=MSFT,NVDA,AAPL,MSFT").json()
        assert payload["symbol_query"] == {
            "applied": True,
            "requested": ["MSFT", "NVDA", "AAPL"],
            "matched": ["MSFT", "AAPL"],
            "missing": ["NVDA"],
        }
        returned_symbols = [entry["symbol"] for entry in payload["prices"]]
        assert returned_symbols == ["MSFT", "AAPL"]

    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_forecast_endpoint_returns_structured_503_when_source_file_cannot_be_read(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"
    _write_predictions(predictions_file, _sample_rows())
    os.utime(predictions_file, (100, 100))
    warning_logs: list[dict[str, object]] = []

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))
    monkeypatch.setattr(
        server,
        "_forecast_metadata_now",
        lambda: datetime.fromtimestamp(200, tz=timezone.utc),
    )
    monkeypatch.setattr(server, "_forecast_cache_entry", None)
    monkeypatch.setattr(
        server,
        "logger",
        SimpleNamespace(info=lambda *_args: None, warning=lambda _message, payload: warning_logs.append(payload)),
    )
    server.stop_crypto_order_loop()

    real_open = Path.open

    def _failing_open(self: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == predictions_file and "r" in str(mode):
            raise OSError("permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _failing_open)

    with TestClient(server.app) as client:
        response = client.get("/api/v1/forecasts/latest?symbols=AAPL,NVDA")
        assert response.status_code == 503
        payload = response.json()
        assert payload["error"] == "forecast_source_unavailable"
        assert payload["forecast_source_status"] == "error"
        assert "permission denied" in payload["error_detail"]
        assert payload["source_file"] == str(predictions_file)
        assert payload["source_filename"] == "predictions.csv"
        assert payload["source_file_updated_at"] == "1970-01-01T00:01:40+00:00"
        assert payload["source_file_age_seconds"] == 100
        assert payload["symbol_query"] == {
            "applied": True,
            "requested": ["AAPL", "NVDA"],
            "matched": [],
            "missing": ["AAPL", "NVDA"],
        }
        assert payload["count"] == 0
        assert payload["forecasts"] == []

    assert warning_logs == [
        {
            "endpoint": "/api/v1/forecasts/latest",
            "status_code": 503,
            "forecast_source_status": "error",
            "source_file": str(predictions_file),
            "source_filename": "predictions.csv",
            "generated_at": None,
            "source_file_age_seconds": 100,
            "symbol_query": {
                "applied": True,
                "requested": ["AAPL", "NVDA"],
                "matched": [],
                "missing": ["AAPL", "NVDA"],
            },
            "count": 0,
            "error": "forecast_source_unavailable",
            "error_detail": "failed to load forecast source: permission denied",
            "source_search_paths": None,
        }
    ]

    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_forecast_discovery_skips_candidates_that_vanish_before_stat(tmp_path, monkeypatch):
    usable_file = tmp_path / "predictions.csv"
    _write_predictions(usable_file, _sample_rows())

    missing_file = tmp_path / "predictions-stale.csv"

    monkeypatch.setattr(server, "_forecast_cache_entry", None)
    monkeypatch.setattr(server, "_candidate_prediction_files", lambda _results_dir: [missing_file, usable_file])
    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path,))
    server.stop_crypto_order_loop()

    with TestClient(server.app) as client:
        response = client.get("/api/v1/forecasts/latest")
        assert response.status_code == 200
        payload = response.json()
        assert payload["source_file"] == str(usable_file)
        assert payload["forecast_source_status"] == "ready"
        assert payload["count"] == 2

    assert server.thread_loop is None or not server.thread_loop.is_alive()


def test_forecast_endpoint_returns_actionable_missing_source_payload(tmp_path, monkeypatch):
    warning_logs: list[dict[str, object]] = []

    monkeypatch.setattr(server, "_FORECAST_RESULTS_DIRS", (tmp_path / "missing-a", tmp_path / "missing-b"))
    monkeypatch.setattr(server, "_forecast_cache_entry", None)
    monkeypatch.setattr(
        server,
        "logger",
        SimpleNamespace(info=lambda _message, payload: warning_logs.append(payload), warning=lambda *_args: None),
    )
    server.stop_crypto_order_loop()

    with TestClient(server.app) as client:
        response = client.get("/api/v1/forecasts/latest?symbols=AAPL,NVDA")

    assert response.status_code == 200
    payload = response.json()
    assert payload["forecast_source_status"] == "missing"
    assert payload["error"] == "forecast_source_missing"
    assert payload["error_detail"] == "no prediction files found in configured results directories"
    assert payload["source_file"] is None
    assert payload["source_search_paths"] == [
        str(tmp_path / "missing-a"),
        str(tmp_path / "missing-b"),
    ]
    assert payload["source_search_filenames"] == ["predictions.csv", "predictions-sim.csv"]
    assert payload["next_steps"] == [
        "write one of predictions.csv, predictions-sim.csv under a configured results directory",
        "rerun the forecast request after the predictions file is generated",
    ]
    assert payload["symbol_query"] == {
        "applied": True,
        "requested": ["AAPL", "NVDA"],
        "matched": [],
        "missing": ["AAPL", "NVDA"],
    }
    assert payload["count"] == 0
    assert payload["forecasts"] == []
    assert warning_logs == [
        {
            "endpoint": "/api/v1/forecasts/latest",
            "status_code": 200,
            "forecast_source_status": "missing",
            "source_file": None,
            "source_filename": None,
            "generated_at": None,
            "source_file_age_seconds": None,
            "symbol_query": {
                "applied": True,
                "requested": ["AAPL", "NVDA"],
                "matched": [],
                "missing": ["AAPL", "NVDA"],
            },
            "count": 0,
            "error": "forecast_source_missing",
            "error_detail": "no prediction files found in configured results directories",
            "source_search_paths": [
                str(tmp_path / "missing-a"),
                str(tmp_path / "missing-b"),
            ],
        }
    ]
