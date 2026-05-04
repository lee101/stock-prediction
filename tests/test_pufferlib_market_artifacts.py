from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pufferlib_market import artifacts


def test_write_json_atomic_overwrites_and_cleans_temp(tmp_path: Path) -> None:
    out = tmp_path / "summary.json"

    artifacts.write_json_atomic(out, {"old": True})
    artifacts.write_json_atomic(out, {"new": True}, sort_keys=True)

    assert json.loads(out.read_text(encoding="utf-8")) == {"new": True}
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_write_json_atomic_cleans_temp_on_replace_failure(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "summary.json"
    out.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(artifacts, "_replace_path", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        artifacts.write_json_atomic(out, {"ok": True})

    assert out.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_save_torch_atomic_overwrites_and_cleans_temp(tmp_path: Path) -> None:
    out = tmp_path / "checkpoint.pt"

    artifacts.save_torch_atomic({"old": torch.tensor([1])}, out)
    artifacts.save_torch_atomic({"new": torch.tensor([2])}, out)

    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert torch.equal(payload["new"], torch.tensor([2]))
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_save_torch_atomic_cleans_temp_on_replace_failure(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "checkpoint.pt"
    torch.save({"old": torch.tensor([1])}, out)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(artifacts, "_replace_path", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        artifacts.save_torch_atomic({"new": torch.tensor([2])}, out)

    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert torch.equal(payload["old"], torch.tensor([1]))
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []
