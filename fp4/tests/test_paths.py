"""Tests for fp4.paths: REPO_ROOT, LOCAL_TMP, ensure_tmp()."""
from __future__ import annotations

from pathlib import Path


def test_repo_root_is_ancestor_of_fp4():
    from fp4.paths import REPO_ROOT
    fp4_init = Path(__file__).resolve().parents[1] / "fp4" / "__init__.py"
    assert fp4_init.exists(), f"sanity: {fp4_init} should exist"
    assert str(fp4_init).startswith(str(REPO_ROOT))


def test_local_tmp_is_under_repo_root():
    from fp4.paths import REPO_ROOT, LOCAL_TMP
    assert str(LOCAL_TMP).startswith(str(REPO_ROOT))
    assert LOCAL_TMP.name == "tmp"


def test_ensure_tmp_creates_dir():
    from fp4.paths import ensure_tmp, LOCAL_TMP
    result = ensure_tmp()
    assert result == LOCAL_TMP
    assert LOCAL_TMP.is_dir()


def test_local_tmp_is_gitignored():
    """tmp/ must be covered by .gitignore so build artefacts stay out of git."""
    from fp4.paths import REPO_ROOT
    gitignore = REPO_ROOT / ".gitignore"
    assert gitignore.exists()
    lines = gitignore.read_text().splitlines()
    # Accept either "tmp/" or "tmp*" as valid ignore patterns
    assert any(
        line.strip() in ("tmp/", "tmp*") for line in lines
    ), "tmp/ or tmp* not found in .gitignore"


def test_no_bare_tmp_in_fp4_source():
    """No remaining /tmp references in fp4/ Python source (excluding tests/comments)."""
    from fp4.paths import REPO_ROOT
    fp4_dir = REPO_ROOT / "fp4" / "fp4"
    violations = []
    for py_file in fp4_dir.rglob("*.py"):
        # Skip __pycache__
        if "__pycache__" in str(py_file):
            continue
        text = py_file.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            # Look for literal /tmp usage that sets TMPDIR or creates paths
            if '"/tmp' in line or "'/tmp" in line:
                violations.append(f"{py_file.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not violations, "Found /tmp references in fp4/ source:\n" + "\n".join(violations)


def test_no_bare_tmp_in_gpu_trading_env():
    """No remaining /tmp references in gpu_trading_env/ Python source."""
    from fp4.paths import REPO_ROOT
    pkg_dir = REPO_ROOT / "gpu_trading_env" / "python" / "gpu_trading_env"
    if not pkg_dir.exists():
        return  # skip if package not present
    violations = []
    for py_file in pkg_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        text = py_file.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if '"/tmp' in line or "'/tmp" in line:
                violations.append(f"{py_file.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not violations, "Found /tmp references in gpu_trading_env/ source:\n" + "\n".join(violations)
