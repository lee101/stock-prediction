"""Tests for src/marketsim_video.py and src/artifacts_server.py."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src import artifacts_server
from src.artifacts_server import create_app
from src.marketsim_video import (
    MarketsimTrace,
    OrderTick,
    TradeSegment,
    _group_trade_segments,
    make_recording_policy,
    render_html_plotly,
    render_mp4,
    trace_from_portfolio_result,
)


def _make_trace(T: int = 12, S: int = 4):
    rng = np.random.default_rng(0)
    base = 100.0 + np.cumsum(rng.standard_normal((T, S)) * 0.5, axis=0).astype(np.float32)
    trace = MarketsimTrace(symbols=[f"SYM{i}" for i in range(S)], prices=base)
    # Hold sym0 long for steps 2..5, then short sym1 for 6..8, then flat.
    for t in range(T):
        if 2 <= t <= 5:
            trace.record(t, action_id=1, position_sym=0, position_is_short=False, equity=10000 + t * 10)
        elif 6 <= t <= 8:
            trace.record(t, action_id=2, position_sym=1, position_is_short=True, equity=10050 + (t - 5) * 5)
        else:
            trace.record(
                t,
                action_id=0,
                position_sym=-1,
                position_is_short=False,
                equity=10000 + t * 2,
                forecast=base[t] + 0.3,
            )
    return trace


def _require_mp4_deps() -> None:
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("matplotlib")


def test_trace_records_and_selects_symbols():
    trace = _make_trace()
    assert trace.num_steps() == 12
    assert trace.num_symbols() == 4
    sel = trace.select_symbol_indices(2)
    # Should pick sym0 and sym1 (the only ones held)
    assert sel == [0, 1]
    # No padding: only the symbols actually held are returned, even when k>active.
    sel4 = trace.select_symbol_indices(4)
    assert sel4 == [0, 1]


def test_render_mp4(tmp_path: Path):
    _require_mp4_deps()
    trace = _make_trace()
    out = tmp_path / "vid.mp4"
    render_mp4(trace, out, num_pairs=4, fps=2, title="test")
    assert out.exists()
    assert out.stat().st_size > 1000  # nontrivial mp4


def test_render_mp4_creates_missing_parent_dir(tmp_path: Path):
    _require_mp4_deps()
    trace = _make_trace()
    out = tmp_path / "nested" / "artifacts" / "vid.mp4"
    render_mp4(trace, out, num_pairs=4, fps=2, title="test")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_render_mp4_recreates_parent_when_writer_open_races_removed_dir(tmp_path: Path, monkeypatch) -> None:
    _require_mp4_deps()
    trace = _make_trace()
    out = tmp_path / "nested" / "artifacts" / "vid.mp4"
    append_checks: list[bool] = []

    class _FakeWriter:
        def __init__(self, path: Path) -> None:
            self.path = path

        def append_data(self, _frame) -> None:
            append_checks.append(self.path.parent.exists())

        def close(self) -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"x" * 2048)

    def _fake_open_video_writer(path: Path, *, fps: int, nvenc: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(path.parent)
        return _FakeWriter(path)

    monkeypatch.setattr("src.marketsim_video._open_video_writer", _fake_open_video_writer)

    render_mp4(trace, out, num_pairs=4, fps=2, title="test")

    assert append_checks
    assert append_checks[0] is True
    assert out.exists()
    assert out.stat().st_size == 2048


def test_render_mp4_retries_when_primary_encoder_writes_tiny_output(tmp_path: Path, monkeypatch) -> None:
    _require_mp4_deps()
    trace = _make_trace()
    out = tmp_path / "retry.mp4"
    open_calls: list[str] = []

    class _FakeWriter:
        def __init__(self, path: Path, payload: bytes) -> None:
            self.path = path
            self.payload = payload

        def append_data(self, _frame) -> None:
            return None

        def close(self) -> None:
            self.path.write_bytes(self.payload)

    def _fake_open_video_writer(path: Path, *, fps: int, nvenc: bool):
        open_calls.append("primary")
        return _FakeWriter(path, b"tiny")

    def _fake_open_software_video_writer(path: Path, *, fps: int):
        open_calls.append("software")
        return _FakeWriter(path, b"x" * 2048)

    monkeypatch.setattr("src.marketsim_video._open_video_writer", _fake_open_video_writer)
    monkeypatch.setattr("src.marketsim_video._open_software_video_writer", _fake_open_software_video_writer)

    render_mp4(trace, out, num_pairs=4, fps=2, title="retry-test")

    assert open_calls == ["primary", "software"]
    assert out.exists()
    assert out.stat().st_size == 2048


def test_fees_accumulate_on_transitions(tmp_path: Path):
    """fees should be charged on every open AND every close (2 fills per round-trip)."""
    _require_mp4_deps()
    trace = _make_trace()
    out = tmp_path / "fees.mp4"
    # default fee_rate=0.001 (10bps); the trace has at least 4 transitions
    # (flat→long, long→short, short→flat, ...). render shouldn't blow up and
    # the file should be nontrivial.
    render_mp4(trace, out, num_pairs=4, fps=2, fee_rate=0.001)
    assert out.stat().st_size > 1000


def test_render_mp4_no_forecast_no_held(tmp_path: Path):
    _require_mp4_deps()
    T, S = 6, 3
    prices = np.linspace(50, 60, T * S, dtype=np.float32).reshape(T, S)
    trace = MarketsimTrace(symbols=["A", "B", "C"], prices=prices)
    for t in range(T):
        trace.record(t, action_id=0, position_sym=-1, position_is_short=False, equity=1000.0)
    out = tmp_path / "flat.mp4"
    render_mp4(trace, out, num_pairs=3, fps=1)
    assert out.stat().st_size > 500


def test_recording_policy_wrapper():
    trace = MarketsimTrace(symbols=["X", "Y"], prices=np.zeros((4, 2), dtype=np.float32))
    base = lambda obs: int(obs.sum()) % 5  # noqa: E731
    wrapped = make_recording_policy(base, trace)
    for i in range(4):
        wrapped(np.array([float(i), 1.0], dtype=np.float32))
    assert len(trace.frames) == 4
    assert [f.action_id for f in trace.frames] == [(i + 1) % 5 for i in range(4)]


def test_trace_json_roundtrip(tmp_path: Path):
    trace = _make_trace()
    # decorate a frame with all optional fields
    S = trace.num_symbols()
    lt = np.full(S, np.nan, dtype=np.float32)
    st = np.full(S, np.nan, dtype=np.float32)
    lt[0] = 100.5
    st[1] = 99.5
    trace.frames[3].long_target = lt
    trace.frames[3].short_target = st
    trace.frames[3].forecast = np.linspace(100, 110, S, dtype=np.float32)
    trace.frames[3].orders = [OrderTick(sym=0, price=100.5, is_short=False)]

    p = tmp_path / "trace.json"
    trace.to_json(p)
    loaded = MarketsimTrace.from_json(p)
    assert loaded.num_steps() == trace.num_steps()
    assert loaded.symbols == trace.symbols
    assert np.allclose(loaded.prices, trace.prices)
    assert loaded.frames[3].long_target[0] == 100.5
    assert loaded.frames[3].short_target[1] == 99.5
    assert loaded.frames[3].orders[0].sym == 0
    assert loaded.frames[3].forecast is not None
    assert loaded.frames[3].forecast.shape == (S,)
    assert loaded.trades == []


def test_render_mp4_with_targets_and_forecast(tmp_path: Path):
    _require_mp4_deps()
    rng = np.random.default_rng(1)
    T, S = 10, 4
    close = 100 + np.cumsum(rng.standard_normal((T, S)) * 0.5, axis=0).astype(np.float32)
    ohlc = np.stack([close, close + 0.4, close - 0.4, close], axis=-1).astype(np.float32)
    trace = MarketsimTrace(symbols=[f"S{i}" for i in range(S)], prices=close, prices_ohlc=ohlc)
    for t in range(T):
        lt = np.full(S, np.nan, dtype=np.float32)
        st = np.full(S, np.nan, dtype=np.float32)
        lt[t % S] = float(close[t, t % S])
        fc = close[t] + 0.3
        trace.record(
            t, action_id=(t % S) + 1,
            position_sym=(t % S) if t > 2 else -1,
            position_is_short=False,
            equity=10000 + t * 5,
            orders=[OrderTick(sym=t % S, price=float(close[t, t % S]), is_short=False)],
            long_target=lt, short_target=st,
            forecast=fc,
        )
    out = tmp_path / "rich.mp4"
    render_mp4(trace, out, num_pairs=4, fps=4, frames_per_bar=2,
               title="rich test", width_px=640, height_px=360, dpi=80)
    assert out.stat().st_size > 1000


def test_render_html_plotly(tmp_path: Path):
    pytest.importorskip("plotly.graph_objects")

    trace = _make_trace()
    out = tmp_path / "view.html"
    render_html_plotly(trace, out, num_pairs=2, title="t")
    assert out.exists()
    assert out.stat().st_size > 1000
    assert "plotly" in out.read_text().lower()


def test_group_trade_segments_derives_legacy_positions_and_closes_open_trade() -> None:
    close = np.array(
        [
            [100.0, 200.0],
            [101.0, 201.0],
            [102.0, 202.0],
            [103.0, 203.0],
            [104.0, 204.0],
        ],
        dtype=np.float32,
    )
    trace = MarketsimTrace(symbols=["AAPL", "MSFT"], prices=close)
    trace.record(0, action_id=0, position_sym=-1, position_is_short=False, equity=10_000.0)
    trace.record(1, action_id=1, position_sym=0, position_is_short=False, equity=10_010.0)
    trace.record(2, action_id=1, position_sym=0, position_is_short=False, equity=10_020.0)
    trace.record(3, action_id=2, position_sym=1, position_is_short=True, equity=10_030.0)
    trace.record(4, action_id=2, position_sym=1, position_is_short=True, equity=10_040.0)

    grouped = _group_trade_segments(trace, close)

    assert grouped[0] == [
        TradeSegment(sym=0, entry_step=1, entry_price=101.0, exit_step=3, exit_price=103.0, is_short=False),
    ]
    assert grouped[1] == [
        TradeSegment(sym=1, entry_step=3, entry_price=203.0, exit_step=4, exit_price=204.0, is_short=True),
    ]


def test_trace_from_portfolio_result_preserves_trade_segments():
    bars = np.array(
        [
            ["2026-01-01T14:00:00Z", "AAPL", 100.0, 101.0, 99.0, 100.5],
            ["2026-01-01T15:00:00Z", "AAPL", 100.5, 102.0, 100.0, 101.8],
            ["2026-01-01T14:00:00Z", "MSFT", 200.0, 201.0, 199.5, 200.2],
            ["2026-01-01T15:00:00Z", "MSFT", 200.2, 202.0, 199.8, 201.5],
        ],
        dtype=object,
    )
    bars_df = pd.DataFrame(bars, columns=["timestamp", "symbol", "open", "high", "low", "close"])
    for column in ["open", "high", "low", "close"]:
        bars_df[column] = bars_df[column].astype(float)
    actions_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T14:00:00Z",
                "symbol": "AAPL",
                "side": "long",
                "buy_price": 100.1,
                "sell_price": 101.7,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "predicted_close_p50_h1": 101.9,
            }
        ]
    )

    class _Result:
        def __init__(self) -> None:
            self.equity_curve = pd.Series(
                [10_000.0, 10_120.0],
                index=pd.to_datetime(["2026-01-01T14:00:00Z", "2026-01-01T15:00:00Z"], utc=True),
            )
            self.metrics = {"final_equity": 10_120.0}
            self.trades = [
                type(
                    "Trade",
                    (),
                    {
                        "timestamp": pd.Timestamp("2026-01-01T14:00:00Z"),
                        "symbol": "AAPL",
                        "side": "buy",
                        "price": 100.1,
                    },
                )(),
                type(
                    "Trade",
                    (),
                    {
                        "timestamp": pd.Timestamp("2026-01-01T15:00:00Z"),
                        "symbol": "AAPL",
                        "side": "sell",
                        "price": 101.7,
                    },
                )(),
            ]

    trace = trace_from_portfolio_result(bars=bars_df, actions=actions_df, result=_Result(), symbols=["AAPL", "MSFT"])

    assert trace.symbols == ["AAPL", "MSFT"]
    assert trace.num_steps() == 2
    assert trace.trades == [
        TradeSegment(sym=0, entry_step=0, entry_price=100.1, exit_step=1, exit_price=101.7, is_short=False)
    ]
    assert trace.frames[0].orders[0].sym == 0
    assert trace.frames[0].forecast is not None
    assert trace.frames[0].forecast[0] == pytest.approx(101.9)


def test_trace_from_portfolio_result_handles_short_trades_and_open_short() -> None:
    bars_df = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T14:00:00Z", "symbol": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            {"timestamp": "2026-01-01T15:00:00Z", "symbol": "AAPL", "open": 100.5, "high": 101.0, "low": 98.5, "close": 99.2},
            {"timestamp": "2026-01-01T16:00:00Z", "symbol": "AAPL", "open": 99.2, "high": 100.0, "low": 98.0, "close": 99.0},
            {"timestamp": "2026-01-01T14:00:00Z", "symbol": "MSFT", "open": 200.0, "high": 201.0, "low": 199.0, "close": 200.5},
            {"timestamp": "2026-01-01T15:00:00Z", "symbol": "MSFT", "open": 200.5, "high": 202.0, "low": 200.0, "close": 201.8},
            {"timestamp": "2026-01-01T16:00:00Z", "symbol": "MSFT", "open": 201.8, "high": 203.0, "low": 201.0, "close": 202.4},
        ]
    )
    actions_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T14:00:00Z",
                "symbol": "AAPL",
                "side": "short",
                "buy_price": 99.0,
                "sell_price": 100.4,
                "buy_amount": 0.0,
                "sell_amount": 100.0,
                "predicted_close_p50_h1": 99.1,
            }
        ]
    )

    class _Result:
        def __init__(self) -> None:
            self.equity_curve = pd.Series(
                [10_000.0, 10_050.0, 10_025.0],
                index=pd.to_datetime(
                    ["2026-01-01T14:00:00Z", "2026-01-01T15:00:00Z", "2026-01-01T16:00:00Z"],
                    utc=True,
                ),
            )
            self.metrics = {"final_equity": 10_025.0}
            self.trades = [
                type("Trade", (), {"timestamp": pd.Timestamp("2026-01-01T14:00:00Z"), "symbol": "AAPL", "side": "short_sell", "price": 100.4})(),
                type("Trade", (), {"timestamp": pd.Timestamp("2026-01-01T15:00:00Z"), "symbol": "AAPL", "side": "buy_cover", "price": 99.2})(),
                type("Trade", (), {"timestamp": pd.Timestamp("2026-01-01T15:00:00Z"), "symbol": "MSFT", "side": "short_sell", "price": 201.3})(),
            ]

    trace = trace_from_portfolio_result(bars=bars_df, actions=actions_df, result=_Result(), symbols=["AAPL", "MSFT"])

    assert trace.trades[0] == TradeSegment(
        sym=0,
        entry_step=0,
        entry_price=100.4,
        exit_step=1,
        exit_price=99.2,
        is_short=True,
    )
    assert trace.trades[1].sym == 1
    assert trace.trades[1].entry_step == 1
    assert trace.trades[1].entry_price == pytest.approx(201.3)
    assert trace.trades[1].exit_step == 2
    assert trace.trades[1].exit_price == pytest.approx(202.4)
    assert trace.trades[1].is_short is True
    assert trace.frames[0].orders[0] == OrderTick(sym=0, price=100.4, is_short=True)
    assert trace.frames[0].short_target is not None
    assert trace.frames[0].short_target[0] == pytest.approx(100.4)


def test_artifacts_server_lists_and_serves(tmp_path: Path):
    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "video.mp4").write_bytes(b"fake-mp4-bytes")
    (tmp_path / "top.txt").write_text("hello")

    app = create_app(tmp_path)
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = client.get("/")
    assert r.status_code == 200
    names = {e["name"] for e in r.json()["entries"]}
    assert {"run1", "top.txt"} <= names

    r = client.get("/list/run1")
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert any(e["name"] == "video.mp4" for e in entries)

    r = client.get("/files/run1/video.mp4")
    assert r.status_code == 200
    assert r.content == b"fake-mp4-bytes"

    r = client.get("/list/does-not-exist")
    assert r.status_code == 404


def test_artifacts_server_path_traversal_blocked(tmp_path: Path):
    app = create_app(tmp_path)
    TestClient(app)
    # httpx normalizes `..` in URLs, so the request resolves to root listing.
    # The real defence is in _safe_join — exercise it directly.
    artifacts_server.ARTIFACTS_ROOT = tmp_path.resolve()
    with pytest.raises(HTTPException, match="path outside artifacts root"):
        artifacts_server._safe_join("../../etc/passwd")


def test_artifacts_server_safe_entry_path_blocks_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(HTTPException, match="path outside artifacts root"):
        artifacts_server._safe_entry_path(outside, root=tmp_path)


def test_artifacts_server_skips_external_symlink_and_blocks_fetch(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (artifacts_root / "safe.txt").write_text("ok")
    (artifacts_root / "escape.txt").symlink_to(outside)

    app = create_app(artifacts_root)
    client = TestClient(app)

    root_entries = {entry["name"] for entry in client.get("/").json()["entries"]}
    assert "safe.txt" in root_entries
    assert "escape.txt" not in root_entries

    response = client.get("/files/escape.txt")
    assert response.status_code == 400
    assert response.json()["detail"] == "path outside artifacts root"


def test_artifacts_server_paginates_root_listing(tmp_path: Path) -> None:
    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text(name, encoding="utf-8")

    client = TestClient(create_app(tmp_path))
    response = client.get("/?limit=2&offset=1")

    assert response.status_code == 200
    payload = response.json()
    assert [entry["name"] for entry in payload["entries"]] == ["b.txt", "c.txt"]
    assert payload["offset"] == 1
    assert payload["limit"] == 2
    assert payload["returned_entries"] == 2
    assert payload["total_entries"] == 3
    assert payload["has_more"] is False
    assert payload["next_offset"] is None


def test_artifacts_server_paginates_subdirectory_listing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    for idx in range(5):
        (run_dir / f"file{idx}.txt").write_text(str(idx), encoding="utf-8")

    client = TestClient(create_app(tmp_path))
    response = client.get("/list/run1?limit=2&offset=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == "run1"
    assert [entry["name"] for entry in payload["entries"]] == ["file2.txt", "file3.txt"]
    assert payload["returned_entries"] == 2
    assert payload["total_entries"] == 5
    assert payload["has_more"] is True
    assert payload["next_offset"] == 4


def test_artifacts_server_apps_keep_separate_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "artifacts-a"
    root_b = tmp_path / "artifacts-b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "only-a.txt").write_text("a", encoding="utf-8")
    (root_b / "only-b.txt").write_text("b", encoding="utf-8")

    client_a = TestClient(create_app(root_a))
    client_b = TestClient(create_app(root_b))

    payload_a = client_a.get("/").json()
    payload_b = client_b.get("/").json()

    assert payload_a["root"] == str(root_a.resolve())
    assert [entry["name"] for entry in payload_a["entries"]] == ["only-a.txt"]
    assert payload_b["root"] == str(root_b.resolve())
    assert [entry["name"] for entry in payload_b["entries"]] == ["only-b.txt"]
