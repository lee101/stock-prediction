from __future__ import annotations

import os
import time
from pathlib import Path

import unified_hourly_experiment.rebuild_all_caches as rebuild


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok")


def test_resolve_model_path_prefers_configured_model(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "chronos2_finetuned"
    preferred = model_root / "NVDA_model_a" / "finetuned-ckpt"
    _touch(preferred)

    monkeypatch.setattr(rebuild, "MODEL_ROOT", model_root)
    resolved = rebuild._resolve_model_path("NVDA", "NVDA_model_a")
    assert resolved == preferred


def test_resolve_model_path_falls_back_to_latest_symbol_model(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "chronos2_finetuned"
    older = model_root / "NVDA_model_old" / "finetuned-ckpt"
    newer = model_root / "NVDA_model_new" / "finetuned-ckpt"
    _touch(older)
    _touch(newer)

    # Force deterministic order by mtime.
    now = time.time()
    os.utime(older, (now - 100.0, now - 100.0))
    os.utime(newer, (now, now))

    monkeypatch.setattr(rebuild, "MODEL_ROOT", model_root)
    resolved = rebuild._resolve_model_path("NVDA", "NVDA_missing_model")
    assert resolved == newer

