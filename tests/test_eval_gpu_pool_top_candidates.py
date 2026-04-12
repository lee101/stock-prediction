from __future__ import annotations

import csv
from pathlib import Path

from scripts.eval_gpu_pool_top_candidates import (
    build_eval_command,
    pick_top_rows,
    resolve_checkpoint_path,
)


def test_pick_top_rows_sorts_descending() -> None:
    rows = [
        {"description": "a", "val_return": "0.10"},
        {"description": "b", "val_return": "0.30"},
        {"description": "c", "val_return": "-0.20"},
    ]

    top = pick_top_rows(rows, sort_by="val_return", top_k=2)

    assert [row["description"] for row in top] == ["b", "a"]


def test_resolve_checkpoint_path_prefers_best(tmp_path: Path) -> None:
    trial_dir = tmp_path / "gpu0" / "trial_a"
    trial_dir.mkdir(parents=True)
    (trial_dir / "final.pt").write_bytes(b"f")
    (trial_dir / "best.pt").write_bytes(b"b")

    resolved = resolve_checkpoint_path(tmp_path, "trial_a", "0")

    assert resolved == trial_dir / "best.pt"


def test_build_eval_command_includes_hourly_intrabar() -> None:
    cmd = build_eval_command(
        checkpoint_path=Path("ckpts/best.pt"),
        val_data=Path("val.bin"),
        out_dir=Path("out"),
        hourly_data_root=Path("trainingdatahourly/stocks"),
        daily_start_date="2025-08-01T00:00:00+00:00",
    )

    assert cmd[0].endswith("python")
    assert "--execution-granularity" in cmd
    assert cmd[cmd.index("--execution-granularity") + 1] == "hourly_intrabar"
