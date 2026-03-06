from __future__ import annotations

from pathlib import Path

import unified_hourly_experiment.rebuild_all_caches as rebuild_mod


def test_resolve_model_path_prefers_configured_checkpoint(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "chronos2_finetuned"
    preferred = model_root / "NVDA_model_a" / "finetuned-ckpt"
    preferred.mkdir(parents=True)

    monkeypatch.setattr(rebuild_mod, "MODEL_ROOT", model_root)

    resolved = rebuild_mod._resolve_model_path("NVDA", "NVDA_model_a")

    assert resolved == preferred


def test_resolve_model_path_falls_back_to_latest_symbol_checkpoint(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "chronos2_finetuned"
    older = model_root / "NVDA_old" / "finetuned-ckpt"
    newer = model_root / "NVDA_new" / "finetuned-ckpt"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    older_ts = 1_700_000_000
    newer_ts = older_ts + 100
    older.touch()
    newer.touch()
    older.parent.touch()
    newer.parent.touch()
    older.touch()
    newer.touch()
    older_stat_target = older
    newer_stat_target = newer
    import os
    os.utime(older_stat_target, (older_ts, older_ts))
    os.utime(newer_stat_target, (newer_ts, newer_ts))

    monkeypatch.setattr(rebuild_mod, "MODEL_ROOT", model_root)

    resolved = rebuild_mod._resolve_model_path("NVDA", "NVDA_missing")

    assert resolved == newer
