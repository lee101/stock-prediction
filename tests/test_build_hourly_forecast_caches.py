from __future__ import annotations

from pathlib import Path

from scripts.build_hourly_forecast_caches import _load_symbols_from_file, _resolve_model_runtime


def test_resolve_model_runtime_uses_symbol_defaults_without_override() -> None:
    params = {
        "model_id": "amazon/chronos-2",
        "device_map": "cuda:1",
    }

    model_id, device_map, wrapper_factory = _resolve_model_runtime(
        params=params,
        model_id_override=None,
        device_map_override=None,
        context_hours=512,
        batch_size=64,
        quantiles=(0.1, 0.5, 0.9),
    )

    assert model_id == "amazon/chronos-2"
    assert device_map == "cuda:1"
    assert wrapper_factory is None


def test_resolve_model_runtime_builds_single_cached_wrapper_for_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[dict[str, object]] = []
    sentinel = object()

    class _FakeWrapper:
        @classmethod
        def from_pretrained(cls, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(dict(kwargs))
            return sentinel

    monkeypatch.setattr("scripts.build_hourly_forecast_caches.Chronos2OHLCWrapper", _FakeWrapper)

    model_id, device_map, wrapper_factory = _resolve_model_runtime(
        params={"model_id": "ignored", "device_map": "cuda"},
        model_id_override="chronos2_finetuned/DOGEUSDT_test/finetuned-ckpt",
        device_map_override="cuda:0",
        context_hours=384,
        batch_size=32,
        quantiles=(0.2, 0.5, 0.8),
    )

    assert model_id == "chronos2_finetuned/DOGEUSDT_test/finetuned-ckpt"
    assert device_map == "cuda:0"
    assert wrapper_factory is not None
    assert wrapper_factory() is sentinel
    assert wrapper_factory() is sentinel
    assert calls == [
        {
            "model_id": "chronos2_finetuned/DOGEUSDT_test/finetuned-ckpt",
            "device_map": "cuda:0",
            "default_context_length": 384,
            "default_batch_size": 32,
            "quantile_levels": (0.2, 0.5, 0.8),
        }
    ]


def test_load_symbols_from_file_supports_comments_commas_and_dedup(tmp_path: Path) -> None:
    path = tmp_path / "symbols.txt"
    path.write_text("aapl, msft\n# ignore\nAAPL\nNVDA\n", encoding="utf-8")

    assert _load_symbols_from_file(path) == ["AAPL", "MSFT", "NVDA"]
