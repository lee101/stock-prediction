from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "autoresearch_stock"
SRC_PACKAGE_ROOT = REPO_ROOT / "src" / "autoresearch_stock"


def test_autoresearch_stock_mirrored_modules_stay_in_sync() -> None:
    root_files = sorted(path.relative_to(PACKAGE_ROOT) for path in PACKAGE_ROOT.rglob("*.py"))
    src_files = sorted(path.relative_to(SRC_PACKAGE_ROOT) for path in SRC_PACKAGE_ROOT.rglob("*.py"))

    assert root_files == src_files, "autoresearch_stock/ and src/autoresearch_stock/ expose different Python modules"

    for relative_path in root_files:
        root_path = PACKAGE_ROOT / relative_path
        src_path = SRC_PACKAGE_ROOT / relative_path
        assert root_path.read_text(encoding="utf-8") == src_path.read_text(encoding="utf-8"), (
            f"{relative_path} drifted between autoresearch_stock/ and src/autoresearch_stock/"
        )
