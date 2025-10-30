from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.provider_latency_history_png import main as png_main


def write_history(tmp_path: Path) -> Path:
    path = tmp_path / "history.jsonl"
    snaps = [
        {
            "timestamp": "2025-10-24T20:00:00+00:00",
            "aggregates": {"yahoo": {"avg_ms": 300.0}},
        },
        {
            "timestamp": "2025-10-24T20:05:00+00:00",
            "aggregates": {"yahoo": {"avg_ms": 320.0}},
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in snaps:
            handle.write(json.dumps(row) + "\n")
    return path


def test_png_main_placeholder(tmp_path, monkeypatch):
    history = write_history(tmp_path)
    output = tmp_path / "plot.png"

    class DummyFigure:
        def write_image(self, *args, **kwargs):  # noqa: ANN001
            raise RuntimeError("kaleido not available")

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "plotly.graph_objects":
            class Module:
                class Figure(DummyFigure):
                    def __init__(self):
                        super().__init__()

                def __getattr__(self, item):
                    raise AttributeError

            return Module()
        return original_import(name, *args, **kwargs)

    original_import = __import__

    def mocked_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "plotly.graph_objects" or name == "matplotlib.pyplot":
            raise ImportError
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(sys, "argv", [
        "provider_latency_history_png.py",
        "--history",
        str(history),
        "--output",
        str(output),
        "--window",
        "5",
    ])

    # simulate ImportError leading to placeholder
    monkeypatch.setattr("builtins.__import__", mocked_import)

    png_main()
    assert output.exists()


def test_png_main_uses_matplotlib(tmp_path, monkeypatch):
    history = write_history(tmp_path)
    output = tmp_path / "plot.png"

    def fake_render_plotly(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("plotly failure")

    def fake_render_matplotlib(path, history, threshold):  # noqa: ANN001
        path.write_bytes(b"fakepng")

    monkeypatch.setattr(
        "scripts.provider_latency_history_png.render_with_plotly",
        fake_render_plotly,
    )
    monkeypatch.setattr(
        "scripts.provider_latency_history_png.render_with_matplotlib",
        fake_render_matplotlib,
    )

    monkeypatch.setattr(sys, "argv", [
        "provider_latency_history_png.py",
        "--history",
        str(history),
        "--output",
        str(output),
        "--window",
        "5",
    ])

    png_main()
    assert output.exists()
    assert output.read_bytes() == b"fakepng"
