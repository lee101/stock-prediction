from __future__ import annotations

import importlib
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from src.torch_load_utils import torch_load_compat


def _rewrite_pickle_to_require_pathlib_local(src_path: Path) -> Path:
    """Patch the checkpoint pickle to reference `pathlib._local.PosixPath`.

    This simulates checkpoints produced by newer Pythons where pathlib paths are
    pickled from `pathlib._local`, which breaks loading on older Pythons unless
    a shim is installed.
    """

    with zipfile.ZipFile(src_path, "r") as zin:
        entries = {name: zin.read(name) for name in zin.namelist()}

    data_key = None
    for name in entries:
        if name == "data.pkl" or name.endswith("/data.pkl"):
            data_key = name
            break
    if data_key is None:
        raise AssertionError("Expected torch zip checkpoint to include a data.pkl entry")

    data = entries[data_key]
    needle = b"cpathlib\nPosixPath\n"
    replacement = b"cpathlib._local\nPosixPath\n"
    if needle in data:
        entries[data_key] = data.replace(needle, replacement)
    elif replacement not in data:
        raise AssertionError(
            "Expected data.pkl to reference pathlib.PosixPath or pathlib._local.PosixPath via GLOBAL opcode"
        )
    # else: already uses pathlib._local (Python 3.13+), no patching needed

    patched = src_path.with_suffix(".patched.pt")
    with zipfile.ZipFile(patched, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, content in entries.items():
            zout.writestr(name, content)
    return patched


def test_torch_load_compat_installs_pathlib_local_shim(tmp_path: Path) -> None:
    original = tmp_path / "checkpoint.pt"
    torch.save({"path": Path("abc")}, original, pickle_protocol=2)
    patched = _rewrite_pickle_to_require_pathlib_local(original)

    # Check whether pathlib._local is natively available (Python 3.13+).
    has_real_module = False
    try:
        importlib.import_module("pathlib._local")
        has_real_module = True
    except Exception:
        has_real_module = False

    if not has_real_module:
        # On older Pythons, loading a checkpoint pickled with pathlib._local
        # should fail without the shim.
        saved = sys.modules.pop("pathlib._local", None)
        try:
            with pytest.raises(ModuleNotFoundError):
                torch.load(patched, map_location="cpu", weights_only=False)
        finally:
            if saved is not None:
                sys.modules["pathlib._local"] = saved

    loaded = torch_load_compat(patched, map_location="cpu", weights_only=False)
    assert isinstance(loaded, dict)
    assert str(loaded["path"]) == "abc"
    # On Python 3.13+ pathlib._local.PosixPath IS pathlib.PosixPath.
    # On older Pythons with the shim, the loaded path is also a real Path.
    # Use a duck-type check that works across all versions.
    assert hasattr(loaded["path"], "parts")
