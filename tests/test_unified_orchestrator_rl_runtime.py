from __future__ import annotations

import json
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from llm_hourly_trader.gemini_wrapper import TradePlan
from unified_orchestrator import orchestrator
from unified_orchestrator.rl_gemini_bridge import RLSignal


def _make_history_frame(symbol: str, *, periods: int = 72, base_price: float = 100.0, freq: str = "h") -> pd.DataFrame:
    index = pd.date_range("2026-03-10 00:00:00+00:00", periods=periods, freq=freq, tz="UTC")
    closes = [base_price + i * 0.5 for i in range(periods)]
    return pd.DataFrame(
        {
            "timestamp": index,
            "symbol": symbol,
            "open": [price - 0.2 for price in closes],
            "high": [price + 0.4 for price in closes],
            "low": [price - 0.6 for price in closes],
            "close": closes,
            "volume": [1_000 + i for i in range(periods)],
        }
    )


def _make_forecast_frame(frame: pd.DataFrame, *, offset: float) -> pd.DataFrame:
    index = pd.to_datetime(frame["timestamp"], utc=True)
    close = frame["close"].astype(float).to_numpy()
    return pd.DataFrame(
        {
            "predicted_close_p50": close + offset,
            "predicted_close_p10": close + offset - 1.0,
            "predicted_close_p90": close + offset + 1.0,
            "predicted_high_p50": close + offset + 0.5,
            "predicted_low_p50": close + offset - 0.5,
        },
        index=index,
    )


def _make_close_frame(symbol: str, closes: list[float], *, freq: str = "h") -> pd.DataFrame:
    index = pd.date_range("2026-03-10 00:00:00+00:00", periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": index,
            "symbol": symbol,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000] * len(closes),
        }
    )


class _FakeBridge:
    def __init__(self, *, checkpoint_path: str, obs_size: int):
        self.checkpoint_path = Path(checkpoint_path)
        self._spec = SimpleNamespace(obs_size=obs_size, alloc_bins=1, level_bins=1)
        self.calls: list[dict[str, object]] = []

    def get_checkpoint_spec(self):
        return self._spec

    def get_rl_signals(self, obs, num_symbols, symbol_names, top_k):
        self.calls.append(
            {
                "obs_shape": tuple(obs.shape),
                "num_symbols": num_symbols,
                "symbol_names": list(symbol_names),
                "top_k": top_k,
            }
        )
        return [
            RLSignal(
                symbol_idx=idx,
                symbol_name=symbol,
                direction="long",
                confidence=0.6 + idx * 0.01,
                logit_gap=1.5,
                allocation_pct=0.5,
            )
            for idx, symbol in enumerate(symbol_names)
        ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_build_crypto_rl_signal_map_uses_full_trained_universe(monkeypatch):
    trained_symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
    monkeypatch.setattr(
        orchestrator,
        "_read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: trained_symbols,
    )

    history_frames = {
        symbol: _make_history_frame(symbol, base_price=100.0 + idx * 10.0)
        for idx, symbol in enumerate(trained_symbols)
    }
    bridge = _FakeBridge(
        checkpoint_path="pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt",
        obs_size=90,
    )

    signal_map = orchestrator._build_crypto_rl_signal_map(history_frames, bridge)

    assert set(signal_map) == set(trained_symbols)
    assert bridge.calls == [
        {
            "obs_shape": (90,),
            "num_symbols": 5,
            "symbol_names": trained_symbols,
            "top_k": 5,
        }
    ]


def test_maybe_use_rl_only_fallback_plan_uses_rl_signal_when_llm_exhausted():
    plan = TradePlan(direction="hold", buy_price=0.0, sell_price=0.0, confidence=0.0, reasoning="API exhausted")
    signal = RLSignal(
        symbol_idx=0,
        symbol_name="ETHUSD",
        direction="long",
        confidence=0.75,
        logit_gap=1.2,
        allocation_pct=0.5,
    )

    out = orchestrator._maybe_use_rl_only_fallback_plan("ETHUSD", plan, signal, 2000.0)

    assert out.direction == "long"
    assert out.buy_price == pytest.approx(1996.0)
    assert out.sell_price == pytest.approx(2020.0)
    assert out.confidence == pytest.approx(0.6)
    assert "llm_unavailable" in out.reasoning


def test_maybe_use_rl_only_fallback_plan_keeps_non_exhausted_hold():
    plan = TradePlan(direction="hold", buy_price=0.0, sell_price=0.0, confidence=0.0, reasoning="model says hold")
    signal = RLSignal(
        symbol_idx=0,
        symbol_name="ETHUSD",
        direction="long",
        confidence=0.75,
        logit_gap=1.2,
        allocation_pct=0.5,
    )

    out = orchestrator._maybe_use_rl_only_fallback_plan("ETHUSD", plan, signal, 2000.0)

    assert out == plan


def test_load_recent_stock_edges_from_meta_log_prefers_latest_fresh_positive_edges(tmp_path: Path):
    event_log = tmp_path / "stock_event_log.jsonl"
    _write_jsonl(
        event_log,
        [
            {
                "event_type": "meta_signal_ready",
                "symbol": "AAPL",
                "edge": 0.01,
                "logged_at": "2026-03-28T07:00:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "symbol": "AAPL",
                "edge": 0.03,
                "logged_at": "2026-03-28T08:50:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "symbol": "NVDA",
                "edge": 0.025,
                "logged_at": "2026-03-28T08:40:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "symbol": "TSLA",
                "edge": -0.02,
                "logged_at": "2026-03-28T08:55:00Z",
            },
            {
                "event_type": "meta_signal_skipped",
                "symbol": "MSFT",
                "edge": 0.04,
                "logged_at": "2026-03-28T08:56:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "symbol": "AAPL",
                "edge": 0.05,
                "logged_at": "2026-03-27T10:00:00Z",
            },
        ],
    )

    out = orchestrator._load_recent_stock_edges_from_meta_log(
        ["AAPL", "NVDA", "TSLA"],
        as_of=datetime(2026, 3, 28, 9, 0, tzinfo=timezone.utc),
        event_log=event_log,
    )

    assert out == {"AAPL": 0.03, "NVDA": 0.025}


@pytest.mark.parametrize(
    ("discount_pct", "expected"),
    [
        (0.01, "mild_caution"),
        (0.02, "mild_caution"),
        (0.03, "soft_reduce"),
        (0.05, "soft_reduce"),
        (0.06, "hard_block"),
    ],
)
def test_classify_crypto_sma_discount_respects_named_thresholds(discount_pct: float, expected: str):
    assert orchestrator._classify_crypto_sma_discount(discount_pct) == expected


def test_crypto_sma_discount_context_returns_none_without_valid_below_sma_signal():
    assert orchestrator._crypto_sma_discount_context(_make_close_frame("BTCUSD", [100.0] * 23)) is None
    assert orchestrator._crypto_sma_discount_context(_make_close_frame("BTCUSD", [100.0] * 24)) is None
    assert orchestrator._crypto_sma_discount_context(_make_close_frame("BTCUSD", [0.0] * 24)) is None


def test_crypto_sma_discount_context_returns_price_sma_and_discount():
    frame = _make_close_frame("BTCUSD", [100.0] * 23 + [94.0])

    current_price, sma_value, discount_pct = orchestrator._crypto_sma_discount_context(frame)

    expected_sma = ((23 * 100.0) + 94.0) / 24.0
    assert current_price == pytest.approx(94.0)
    assert sma_value == pytest.approx(expected_sma)
    assert discount_pct == pytest.approx((expected_sma - 94.0) / expected_sma)


def test_fetch_history_frames_parallel_preserves_input_order_and_skips_missing():
    delays = {"BTCUSD": 0.03, "ETHUSD": 0.0, "SOLUSD": 0.01}

    def _fetch_one(symbol: str):
        time.sleep(delays[symbol])
        if symbol == "ETHUSD":
            return None
        return {"symbol": symbol}

    out = orchestrator._fetch_history_frames_parallel(
        ["BTCUSD", "ETHUSD", "SOLUSD"],
        _fetch_one,
        max_workers=3,
    )

    assert list(out) == ["BTCUSD", "SOLUSD"]
    assert out["BTCUSD"] == {"symbol": "BTCUSD"}
    assert out["SOLUSD"] == {"symbol": "SOLUSD"}


def test_fetch_history_frames_parallel_skips_worker_exceptions_without_aborting_batch():
    delays = {"BTCUSD": 0.03, "ETHUSD": 0.0, "SOLUSD": 0.01}

    def _fetch_one(symbol: str):
        time.sleep(delays[symbol])
        if symbol == "ETHUSD":
            raise RuntimeError("unexpected provider failure")
        return {"symbol": symbol}

    out = orchestrator._fetch_history_frames_parallel(
        ["BTCUSD", "ETHUSD", "SOLUSD"],
        _fetch_one,
        max_workers=3,
    )

    assert list(out) == ["BTCUSD", "SOLUSD"]
    assert out["BTCUSD"] == {"symbol": "BTCUSD"}
    assert out["SOLUSD"] == {"symbol": "SOLUSD"}


def test_fetch_history_frames_parallel_serial_mode_still_isolates_worker_exceptions():
    def _fetch_one(symbol: str):
        if symbol == "ETHUSD":
            raise RuntimeError("unexpected provider failure")
        return {"symbol": symbol}

    out = orchestrator._fetch_history_frames_parallel(
        ["BTCUSD", "ETHUSD", "SOLUSD"],
        _fetch_one,
        max_workers=1,
    )

    assert list(out) == ["BTCUSD", "SOLUSD"]
    assert out["BTCUSD"] == {"symbol": "BTCUSD"}
    assert out["SOLUSD"] == {"symbol": "SOLUSD"}


def test_fetch_crypto_bars_preserves_symbol_order_when_requests_complete_out_of_order(monkeypatch):
    class _FakeCryptoBarsRequest:
        def __init__(self, **kwargs):
            self.symbol_or_symbols = kwargs["symbol_or_symbols"]

    class _FakeClient:
        _delays = {"BTC/USD": 0.03, "ETH/USD": 0.0, "SOL/USD": 0.01}

        def get_crypto_bars(self, request):
            time.sleep(self._delays[request.symbol_or_symbols])
            return SimpleNamespace(df={"request_symbol": request.symbol_or_symbols})

    fake_alpaca = ModuleType("alpaca")
    fake_data = ModuleType("alpaca.data")
    fake_requests = ModuleType("alpaca.data.requests")
    fake_requests.CryptoBarsRequest = _FakeCryptoBarsRequest
    fake_data.requests = fake_requests
    fake_alpaca.data = fake_data
    monkeypatch.setitem(sys.modules, "alpaca", fake_alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.data", fake_data)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setattr(orchestrator, "_select_symbol_frame", lambda raw_df, request_sym: raw_df)

    out = orchestrator._fetch_crypto_bars(
        _FakeClient(),
        ["BTCUSD", "ETHUSD", "SOLUSD"],
        datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc),
        timeframe="1H",
        lookback=timedelta(hours=4),
        limit=4,
    )

    assert list(out) == ["BTCUSD", "ETHUSD", "SOLUSD"]
    assert out["BTCUSD"] == {"request_symbol": "BTC/USD"}
    assert out["ETHUSD"] == {"request_symbol": "ETH/USD"}
    assert out["SOLUSD"] == {"request_symbol": "SOL/USD"}


def test_fetch_stock_history_frames_skips_failed_symbol_without_losing_others(monkeypatch):
    class _FakeStockBarsRequest:
        def __init__(self, **kwargs):
            self.symbol_or_symbols = kwargs["symbol_or_symbols"]
            self.feed = kwargs["feed"]
            self.timeframe = kwargs["timeframe"]

    class _FakeClient:
        def get_stock_bars(self, request):
            if request.symbol_or_symbols == "NVDA":
                raise RuntimeError("rate limit")
            return SimpleNamespace(df={"request_symbol": request.symbol_or_symbols})

    fake_alpaca = ModuleType("alpaca")
    fake_data = ModuleType("alpaca.data")
    fake_requests = ModuleType("alpaca.data.requests")
    fake_requests.StockBarsRequest = _FakeStockBarsRequest
    fake_enums = ModuleType("alpaca.data.enums")
    fake_enums.DataFeed = SimpleNamespace(IEX="IEX")
    fake_timeframe = ModuleType("alpaca.data.timeframe")
    fake_timeframe.TimeFrame = SimpleNamespace(Hour="1H")
    fake_data.requests = fake_requests
    fake_data.enums = fake_enums
    fake_data.timeframe = fake_timeframe
    fake_alpaca.data = fake_data
    monkeypatch.setitem(sys.modules, "alpaca", fake_alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.data", fake_data)
    monkeypatch.setitem(sys.modules, "alpaca.data.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.data.enums", fake_enums)
    monkeypatch.setitem(sys.modules, "alpaca.data.timeframe", fake_timeframe)
    monkeypatch.setattr(orchestrator, "_select_symbol_frame", lambda raw_df, symbol: raw_df)

    out = orchestrator._fetch_stock_history_frames(
        _FakeClient(),
        ["AAPL", "NVDA", "MSFT"],
        datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc),
        lookback_hours=4,
    )

    assert list(out) == ["AAPL", "MSFT"]
    assert out["AAPL"] == {"request_symbol": "AAPL"}
    assert out["MSFT"] == {"request_symbol": "MSFT"}


def test_append_cycle_event_writes_structured_jsonl(tmp_path: Path):
    event_log = tmp_path / "orchestrator_cycle_events.jsonl"

    orchestrator._append_cycle_event(
        "cycle_start",
        event_log=event_log,
        logged_at=datetime(2026, 3, 29, 1, 2, 3, tzinfo=timezone.utc),
        cycle_id="cycle-123",
        dry_run=True,
        crypto_symbols=["BTCUSD"],
    )

    rows = _read_jsonl(event_log)

    assert rows == [
        {
            "event_type": "cycle_start",
            "logged_at": "2026-03-29T01:02:03Z",
            "cycle_id": "cycle-123",
            "dry_run": True,
            "crypto_symbols": ["BTCUSD"],
        }
    ]


def test_run_cycle_with_runtime_logging_writes_error_event(tmp_path: Path, monkeypatch):
    event_log = tmp_path / "orchestrator_cycle_events.jsonl"

    def _boom(**kwargs):
        raise RuntimeError(f"boom for {kwargs['crypto_symbols'][0]}")

    monkeypatch.setattr(orchestrator, "run_cycle", _boom)

    ok = orchestrator._run_cycle_with_runtime_logging(
        crypto_symbols=["BTCUSD"],
        stock_symbols=["NVDA", "MSFT"],
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        review_thinking_level="LOW",
        reprompt_passes=2,
        reprompt_policy="actionable",
        review_max_confidence=0.6,
        review_model="gemini-3.1-pro-preview",
        dry_run=True,
        event_log=event_log,
    )

    rows = _read_jsonl(event_log)

    assert ok is False
    assert [row["event_type"] for row in rows] == ["cycle_start", "cycle_error"]
    start_row, error_row = rows
    assert start_row["cycle_id"] == error_row["cycle_id"]
    assert error_row["dry_run"] is True
    assert error_row["crypto_symbol_count"] == 1
    assert error_row["stock_symbol_count"] == 2
    assert error_row["review_thinking_level"] == "LOW"
    assert error_row["error_type"] == "RuntimeError"
    assert error_row["error"] == "boom for BTCUSD"
    assert "RuntimeError: boom for BTCUSD" in error_row["traceback"]
    assert float(error_row["duration_seconds"]) >= 0.0


def test_run_cycle_with_runtime_logging_writes_success_event(tmp_path: Path, monkeypatch):
    event_log = tmp_path / "orchestrator_cycle_events.jsonl"
    captured: dict[str, object] = {}

    def _ok(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(orchestrator, "run_cycle", _ok)

    ok = orchestrator._run_cycle_with_runtime_logging(
        crypto_symbols=["BTCUSD", "ETHUSD"],
        stock_symbols=["NVDA"],
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        review_thinking_level=None,
        reprompt_passes=1,
        reprompt_policy="always",
        review_max_confidence=None,
        review_model=None,
        dry_run=False,
        event_log=event_log,
    )

    rows = _read_jsonl(event_log)

    assert ok is True
    assert captured["crypto_symbols"] == ["BTCUSD", "ETHUSD"]
    assert captured["stock_symbols"] == ["NVDA"]
    assert [row["event_type"] for row in rows] == ["cycle_start", "cycle_success"]
    start_row, success_row = rows
    assert start_row["cycle_id"] == success_row["cycle_id"]
    assert success_row["dry_run"] is False
    assert success_row["crypto_symbol_count"] == 2
    assert success_row["stock_symbol_count"] == 1
    assert float(success_row["duration_seconds"]) >= 0.0


def test_load_recent_stock_edges_from_meta_log_prefers_latest_real_cycle_over_isolated_signal_rows(tmp_path: Path):
    event_log = tmp_path / "stock_event_log.jsonl"
    _write_jsonl(
        event_log,
        [
            {
                "event_type": "meta_cycle_start",
                "pid": 153050,
                "logged_at": "2026-03-28T09:15:00Z",
            },
            {
                "event_type": "meta_signal_ready",
                "pid": 153050,
                "symbol": "AAPL",
                "edge": 0.04,
                "logged_at": "2026-03-28T09:15:10Z",
            },
            {
                "event_type": "meta_signal_ready",
                "pid": 999999,
                "symbol": "NVDA",
                "edge": 0.50,
                "logged_at": "2026-03-28T09:20:00Z",
            },
        ],
    )

    out = orchestrator._load_recent_stock_edges_from_meta_log(
        ["AAPL", "NVDA"],
        as_of=datetime(2026, 3, 28, 9, 30, tzinfo=timezone.utc),
        event_log=event_log,
    )

    assert out == {"AAPL": 0.04}


def test_run_cycle_pre_market_passes_recent_stock_edges_to_backout(monkeypatch):
    snapshot = SimpleNamespace(
        regime="PRE_MARKET",
        total_stock_value=10_000.0,
        alpaca_positions={},
        binance_positions={},
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(orchestrator, "build_snapshot", lambda now=None: snapshot)
    monkeypatch.setattr(orchestrator, "save_snapshot", lambda snapshot: None)
    monkeypatch.setattr(orchestrator, "read_pending_fills", lambda since_minutes=65: [])
    monkeypatch.setattr(orchestrator, "get_crypto_signals", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        orchestrator,
        "_load_recent_stock_edges_from_meta_log",
        lambda stock_symbols, as_of=None: {"AAPL": 0.04, "NVDA": 0.02},
    )

    def _fake_select_backout_candidates(snapshot_arg, best_stock_edges, min_edge_ratio=2.0, only_profitable=True):
        captured["snapshot"] = snapshot_arg
        captured["best_stock_edges"] = best_stock_edges
        return []

    monkeypatch.setattr(orchestrator, "select_backout_candidates", _fake_select_backout_candidates)

    results = orchestrator.run_cycle(
        crypto_symbols=["BTCUSD"],
        stock_symbols=["AAPL", "NVDA"],
        dry_run=True,
    )

    assert captured["snapshot"] is snapshot
    assert captured["best_stock_edges"] == {"AAPL": 0.04, "NVDA": 0.02}
    assert results["best_stock_edges"] == {"AAPL": 0.04, "NVDA": 0.02}


def test_build_stock_rl_signal_map_uses_forecast_features(monkeypatch):
    trained_symbols = ["AAPL", "NVDA"]
    monkeypatch.setattr(
        orchestrator,
        "_read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: trained_symbols,
    )

    history_frames = {
        symbol: _make_history_frame(symbol, base_price=150.0 + idx * 25.0)
        for idx, symbol in enumerate(trained_symbols)
    }
    forecasts_1h = {
        symbol: _make_forecast_frame(frame, offset=1.0)
        for symbol, frame in history_frames.items()
    }
    forecasts_24h = {
        symbol: _make_forecast_frame(frame, offset=3.0)
        for symbol, frame in history_frames.items()
    }
    bridge = _FakeBridge(
        checkpoint_path="pufferlib_market/checkpoints/stocks13_featlag1_fee5bps_longonly_run4/best.pt",
        obs_size=39,
    )

    signal_map = orchestrator._build_stock_rl_signal_map(
        history_frames,
        bridge,
        forecasts_1h,
        forecasts_24h,
    )

    assert set(signal_map) == set(trained_symbols)
    assert bridge.calls == [
        {
            "obs_shape": (39,),
            "num_symbols": 2,
            "symbol_names": trained_symbols,
            "top_k": 2,
        }
    ]


# ---------------------------------------------------------------------------
# _is_daily_checkpoint tests
# ---------------------------------------------------------------------------

def _write_mktd_header(path: Path, symbols: list[str]) -> None:
    """Write a minimal MKTD header for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD", 2, len(symbols), 1, 16, 5, b"\x00" * 40,
    )
    with path.open("wb") as handle:
        handle.write(header)
        for sym in symbols:
            handle.write(sym.encode("ascii").ljust(16, b"\x00"))


def test_is_daily_checkpoint_detects_daily_data_hint(monkeypatch, tmp_path):
    """When the data-hint .bin filename contains 'daily', checkpoint is daily."""
    data_path = tmp_path / "crypto5_daily_train.bin"
    _write_mktd_header(data_path, ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"])

    checkpoint = tmp_path / "checkpoints" / "tp0.15_s314" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")

    monkeypatch.setitem(orchestrator._CHECKPOINT_DATA_HINTS, "tp0.15_s314", data_path)
    assert orchestrator._is_daily_checkpoint(checkpoint) is True


def test_is_daily_checkpoint_detects_daily_in_path():
    """When checkpoint path contains 'mass_daily', checkpoint is daily."""
    checkpoint = Path("pufferlib_market/checkpoints/mass_daily/tp0.15_s314/best.pt")
    assert orchestrator._is_daily_checkpoint(checkpoint) is True


def test_is_daily_checkpoint_returns_false_for_hourly():
    """Hourly checkpoints (no 'daily' in path or data hint) return False."""
    checkpoint = Path("pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt")
    assert orchestrator._is_daily_checkpoint(checkpoint) is False


def test_is_daily_checkpoint_returns_false_for_stock_hourly():
    """Stock hourly checkpoints return False."""
    checkpoint = Path("pufferlib_market/checkpoints/stocks13_featlag1_fee5bps_longonly_run4/best.pt")
    assert orchestrator._is_daily_checkpoint(checkpoint) is False


# ---------------------------------------------------------------------------
# _build_crypto_rl_signal_map with daily checkpoint
# ---------------------------------------------------------------------------

def test_build_crypto_rl_signal_map_daily_checkpoint(monkeypatch):
    """Daily checkpoint uses compute_daily_features and requires 60 bars."""
    trained_symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
    monkeypatch.setattr(
        orchestrator,
        "_read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: trained_symbols,
    )

    # Create daily frames with 90 bars (> 60 minimum)
    history_frames = {
        symbol: _make_history_frame(symbol, periods=90, base_price=100.0 + idx * 10.0, freq="D")
        for idx, symbol in enumerate(trained_symbols)
    }
    bridge = _FakeBridge(
        checkpoint_path="pufferlib_market/checkpoints/mass_daily/tp0.15_s314/best.pt",
        obs_size=90,
    )

    signal_map = orchestrator._build_crypto_rl_signal_map(history_frames, bridge)

    assert set(signal_map) == set(trained_symbols)
    assert bridge.calls[0]["num_symbols"] == 5


def test_build_crypto_rl_signal_map_daily_skips_insufficient_bars(monkeypatch):
    """Daily checkpoint skips symbols with < 60 bars."""
    trained_symbols = ["BTCUSD", "ETHUSD"]
    monkeypatch.setattr(
        orchestrator,
        "_read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: trained_symbols,
    )

    history_frames = {
        "BTCUSD": _make_history_frame("BTCUSD", periods=90, base_price=80000.0, freq="D"),
        "ETHUSD": _make_history_frame("ETHUSD", periods=30, base_price=2000.0, freq="D"),  # too few
    }
    bridge = _FakeBridge(
        checkpoint_path="pufferlib_market/checkpoints/mass_daily/tp0.10_s42/best.pt",
        obs_size=39,
    )

    signal_map = orchestrator._build_crypto_rl_signal_map(history_frames, bridge)

    # BTCUSD should be in signal map, ETHUSD skipped
    assert "BTCUSD" in signal_map
    # ETHUSD was skipped (< 60 bars) but bridge still gets called with all trained_symbols
    assert bridge.calls[0]["num_symbols"] == 2


# ---------------------------------------------------------------------------
# compute_daily_features smoke test
# ---------------------------------------------------------------------------

def test_compute_daily_features_returns_16_floats():
    """compute_daily_features returns a 16-element float32 array."""
    from pufferlib_market.inference_daily import compute_daily_features

    df = pd.DataFrame({
        "open": np.random.uniform(90, 110, 100),
        "high": np.random.uniform(100, 120, 100),
        "low": np.random.uniform(80, 100, 100),
        "close": np.random.uniform(90, 110, 100),
        "volume": np.random.uniform(1e5, 1e7, 100),
    })
    features = compute_daily_features(df)
    assert features.shape == (16,)
    assert features.dtype == np.float32
    # At least some features should be non-zero
    assert (features != 0).sum() > 0
