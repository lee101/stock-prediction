"""Smoke test: compare_trainers --smoke writes a parseable markdown report."""

from __future__ import annotations

import re
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_trainers  # type: ignore  # noqa: E402


def test_compare_smoke(tmp_path, monkeypatch):
    rc = compare_trainers.main(["--smoke", "--trainers", "pufferlib_bf16"])
    assert rc == 0
    md_files = sorted(compare_trainers.RESULTS_DIR.glob("fp4_vs_baselines_*.md"))
    assert md_files, "expected at least one markdown report"
    text = md_files[-1].read_text()
    assert "# fp4 vs baseline trainer comparison" in text
    assert "## Recommendation" in text
    assert re.search(r"\*\*Winner:\s*`[^`]+`\*\*", text)
