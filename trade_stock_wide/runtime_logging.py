from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _build_file_logger(name: str, path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.FileHandler(path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@dataclass
class WideRunLogger:
    run_dir: Path
    central: logging.Logger
    _symbol_loggers: dict[str, logging.Logger] = field(default_factory=dict)

    @classmethod
    def create(cls, root: Path | str = "trade_stock_wide/logs") -> "WideRunLogger":
        ts = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path(root) / ts
        symbol_dir = run_dir / "symbols"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        central = _build_file_logger(f"trade_stock_wide.{ts}.central", run_dir / "central.log")
        return cls(run_dir=run_dir, central=central)

    def symbol_logger(self, symbol: str) -> logging.Logger:
        normalized = symbol.upper()
        logger = self._symbol_loggers.get(normalized)
        if logger is not None:
            return logger
        path = self.run_dir / "symbols" / f"{normalized}.log"
        logger = _build_file_logger(
            f"trade_stock_wide.{self.run_dir.name}.{normalized}",
            path,
        )
        self._symbol_loggers[normalized] = logger
        return logger

    def event(self, message: str, *, symbol: str | None = None, level: int = logging.INFO) -> None:
        self.central.log(level, message)
        if symbol:
            self.symbol_logger(symbol).log(level, message)
