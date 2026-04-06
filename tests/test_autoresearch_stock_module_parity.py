from __future__ import annotations

import autoresearch_stock.agent_scheduler as root_agent_scheduler_module
import autoresearch_stock.prepare as root_prepare_module
import autoresearch_stock.train as root_train_module
from pathlib import Path
import src.autoresearch_stock.agent_scheduler as src_agent_scheduler_module
import src.autoresearch_stock.prepare as src_prepare_module
import src.autoresearch_stock.train as src_train_module


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "autoresearch_stock"
SRC_PACKAGE_ROOT = REPO_ROOT / "src" / "autoresearch_stock"
SHIMMED_MODULES = {
    Path("agent_scheduler.py"),
    Path("prepare.py"),
    Path("train.py"),
}


def test_autoresearch_stock_mirrored_modules_stay_in_sync() -> None:
    root_files = sorted(path.relative_to(PACKAGE_ROOT) for path in PACKAGE_ROOT.rglob("*.py"))
    src_files = sorted(path.relative_to(SRC_PACKAGE_ROOT) for path in SRC_PACKAGE_ROOT.rglob("*.py"))

    assert root_files == src_files, "autoresearch_stock/ and src/autoresearch_stock/ expose different Python modules"

    for relative_path in root_files:
        root_path = PACKAGE_ROOT / relative_path
        src_path = SRC_PACKAGE_ROOT / relative_path
        if relative_path in SHIMMED_MODULES:
            continue
        assert root_path.read_text(encoding="utf-8") == src_path.read_text(encoding="utf-8"), (
            f"{relative_path} drifted between autoresearch_stock/ and src/autoresearch_stock/"
        )


def test_autoresearch_stock_root_prepare_train_and_scheduler_delegate_to_src_modules() -> None:
    assert root_agent_scheduler_module.main is src_agent_scheduler_module.main
    assert root_agent_scheduler_module.parse_train_log is src_agent_scheduler_module.parse_train_log
    assert root_prepare_module.main is src_prepare_module.main
    assert root_prepare_module.prepare_task is src_prepare_module.prepare_task
    assert root_train_module.main is src_train_module.main
    assert root_train_module.resolve_learning_rate is src_train_module.resolve_learning_rate
