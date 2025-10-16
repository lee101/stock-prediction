from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    tomllib = None  # type: ignore[assignment]


DEFAULT_SPREAD_SYMBOLS: Sequence[str] = (
    "AAPL",
    "AMD",
    "GOOG",
    "MSFT",
    "NVDA",
    "TSLA",
    "BTCUSD",
    "ETHUSD",
)

DEFAULT_COLLECTION_INTERVAL_SECONDS = 300


@dataclass(slots=True)
class DashboardConfig:
    """Runtime configuration for the dashboards package."""

    db_path: Path
    shelf_files: List[Path] = field(default_factory=list)
    spread_symbols: List[str] = field(default_factory=list)
    log_files: Dict[str, Path] = field(default_factory=dict)
    collection_interval_seconds: int = DEFAULT_COLLECTION_INTERVAL_SECONDS
    snapshot_chunk_size: int = 512 * 1024  # avoid massive sqlite rows accidentally

    @property
    def repo_root(self) -> Path:
        return self.db_path.resolve().parent.parent

    def ensure_paths(self) -> None:
        """Make sure all runtime paths are ready before use."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def _load_config_from_toml(path: Path) -> dict:
    if not tomllib:
        raise RuntimeError(
            f"Attempted to load {path} but tomllib is unavailable. "
            "Use config.json or upgrade to Python 3.11+."
        )
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _load_config_from_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _collect_candidate_files(dashboards_dir: Path) -> Iterable[Path]:
    yield dashboards_dir / "config.toml"
    yield dashboards_dir / "config.json"


def _coerce_shelf_paths(raw_paths: Iterable[str], repo_root: Path) -> List[Path]:
    shelves: List[Path] = []
    for raw in raw_paths:
        raw = raw.strip()
        if not raw:
            continue
        path = (repo_root / raw).resolve() if not raw.startswith("/") else Path(raw)
        shelves.append(path)
    return shelves


def _coerce_log_paths(raw_logs: dict, repo_root: Path, dashboards_dir: Path) -> Dict[str, Path]:
    log_files: Dict[str, Path] = {}
    if not isinstance(raw_logs, dict):
        return log_files
    for name, raw_path in raw_logs.items():
        if not isinstance(raw_path, str):
            continue
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            repo_candidate = (repo_root / candidate).resolve()
            dashboards_candidate = (dashboards_dir / candidate).resolve()
            if repo_candidate.exists():
                candidate = repo_candidate
            elif dashboards_candidate.exists():
                candidate = dashboards_candidate
            else:
                candidate = repo_candidate
        log_files[name.lower()] = candidate
    return log_files


def load_config(base_dir: Path | None = None) -> DashboardConfig:
    """
    Load the dashboards configuration.

    Preference order:
        1. dashboards/config.toml
        2. dashboards/config.json
    """
    dashboards_dir = base_dir or Path(__file__).resolve().parent
    repo_root = dashboards_dir.parent

    raw_config: dict = {}
    for candidate in _collect_candidate_files(dashboards_dir):
        if candidate.exists():
            loader = _load_config_from_toml if candidate.suffix == ".toml" else _load_config_from_json
            raw_config = loader(candidate)
            break

    db_path = raw_config.get("db_path")
    if db_path:
        db_path = Path(db_path)
        if not db_path.is_absolute():
            db_path = (dashboards_dir / db_path).resolve()
    else:
        db_path = dashboards_dir / "metrics.db"

    shelf_files = raw_config.get("shelf_files")
    if not shelf_files:
        default_shelf = repo_root / "positions_shelf.json"
        shelf_files = [str(default_shelf)] if default_shelf.exists() else []

    spread_symbols = raw_config.get("spread_symbols") or list(DEFAULT_SPREAD_SYMBOLS)
    collection_interval_seconds = int(
        raw_config.get("collection_interval_seconds", DEFAULT_COLLECTION_INTERVAL_SECONDS)
    )
    log_files = _coerce_log_paths(raw_config.get("logs", {}), repo_root=repo_root, dashboards_dir=dashboards_dir)

    if not log_files:
        default_trade = repo_root / "trade_stock_e2e.log"
        default_alpaca = repo_root / "alpaca_cli.log"
        if default_trade.exists():
            log_files["trade"] = default_trade.resolve()
        if default_alpaca.exists():
            log_files["alpaca"] = default_alpaca.resolve()

    config = DashboardConfig(
        db_path=Path(db_path).resolve(),
        shelf_files=_coerce_shelf_paths(shelf_files, repo_root=repo_root),
        spread_symbols=[symbol.upper() for symbol in spread_symbols],
        log_files=log_files,
        collection_interval_seconds=collection_interval_seconds,
        snapshot_chunk_size=int(raw_config.get("snapshot_chunk_size", 512 * 1024)),
    )
    config.ensure_paths()
    return config


__all__ = ["DashboardConfig", "load_config"]
