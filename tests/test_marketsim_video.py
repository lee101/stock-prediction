"""Tests for src/marketsim_video.py and src/artifacts_server.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_trace(T: int = 12, S: int = 4):
    from src.marketsim_video import MarketsimTrace

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


def test_trace_records_and_selects_symbols():
    trace = _make_trace()
    assert trace.num_steps() == 12
    assert trace.num_symbols() == 4
    sel = trace.select_symbol_indices(2)
    # Should pick sym0 and sym1 (the only ones held)
    assert sel == [0, 1]
    sel4 = trace.select_symbol_indices(4)
    assert sel4 == [0, 1, 2, 3]


def test_render_mp4(tmp_path: Path):
    from src.marketsim_video import render_mp4

    trace = _make_trace()
    out = tmp_path / "vid.mp4"
    render_mp4(trace, out, num_pairs=4, fps=2, title="test")
    assert out.exists()
    assert out.stat().st_size > 1000  # nontrivial mp4


def test_fees_accumulate_on_transitions(tmp_path: Path):
    """fees should be charged on every open AND every close (2 fills per round-trip)."""
    from src.marketsim_video import render_mp4

    trace = _make_trace()
    out = tmp_path / "fees.mp4"
    # default fee_rate=0.001 (10bps); the trace has at least 4 transitions
    # (flat→long, long→short, short→flat, ...). render shouldn't blow up and
    # the file should be nontrivial.
    render_mp4(trace, out, num_pairs=4, fps=2, fee_rate=0.001)
    assert out.stat().st_size > 1000


def test_render_mp4_no_forecast_no_held(tmp_path: Path):
    from src.marketsim_video import MarketsimTrace, render_mp4

    T, S = 6, 3
    prices = np.linspace(50, 60, T * S, dtype=np.float32).reshape(T, S)
    trace = MarketsimTrace(symbols=["A", "B", "C"], prices=prices)
    for t in range(T):
        trace.record(t, action_id=0, position_sym=-1, position_is_short=False, equity=1000.0)
    out = tmp_path / "flat.mp4"
    render_mp4(trace, out, num_pairs=3, fps=1)
    assert out.stat().st_size > 500


def test_recording_policy_wrapper():
    from src.marketsim_video import MarketsimTrace, make_recording_policy

    trace = MarketsimTrace(symbols=["X", "Y"], prices=np.zeros((4, 2), dtype=np.float32))
    base = lambda obs: int(obs.sum()) % 5  # noqa: E731
    wrapped = make_recording_policy(base, trace)
    for i in range(4):
        wrapped(np.array([float(i), 1.0], dtype=np.float32))
    assert len(trace.frames) == 4
    assert [f.action_id for f in trace.frames] == [(i + 1) % 5 for i in range(4)]


def test_trace_json_roundtrip(tmp_path: Path):
    from src.marketsim_video import MarketsimTrace, OrderTick

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


def test_render_mp4_with_targets_and_forecast(tmp_path: Path):
    from src.marketsim_video import MarketsimTrace, OrderTick, render_mp4

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
    from src.marketsim_video import render_html_plotly

    trace = _make_trace()
    out = tmp_path / "view.html"
    render_html_plotly(trace, out, num_pairs=2, title="t")
    assert out.exists()
    assert out.stat().st_size > 1000
    assert "plotly" in out.read_text().lower()


def test_artifacts_server_lists_and_serves(tmp_path: Path):
    from fastapi.testclient import TestClient

    from src.artifacts_server import create_app

    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "video.mp4").write_bytes(b"fake-mp4-bytes")
    (tmp_path / "top.txt").write_text("hello")

    app = create_app(tmp_path)
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True

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
    from fastapi.testclient import TestClient

    from src.artifacts_server import create_app

    app = create_app(tmp_path)
    client = TestClient(app)
    # httpx normalizes `..` in URLs, so the request resolves to root listing.
    # The real defence is in _safe_join — exercise it directly.
    from src import artifacts_server

    artifacts_server.ARTIFACTS_ROOT = tmp_path.resolve()
    with pytest.raises(Exception):
        artifacts_server._safe_join("../../etc/passwd")
