from __future__ import annotations

from pathlib import Path

from src.models.chronos2_wrapper import _resolve_model_source


def test_resolve_model_source_falls_back_to_available_symbol_family(tmp_path: Path, monkeypatch) -> None:
    fallback = tmp_path / "chronos2_finetuned" / "BTCUSD_lora_20260203_051412" / "finetuned-ckpt"
    fallback.mkdir(parents=True)
    (fallback / "config.json").write_text("{}")
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_model_source(
        "chronos2_finetuned/BTC_lora_percent_change_ctx128_lr5e-05_r16_20260212_114916/finetuned-ckpt"
    )

    assert Path(resolved).resolve() == fallback.resolve()
