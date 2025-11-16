"""Symbol source helpers for strategytraining utilities."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List


class _TradeStockSymbolExtractor(ast.NodeVisitor):
    """Extract the literal ``symbols = [...]`` assignment anywhere in the module."""

    def __init__(self) -> None:
        self._targets = ("symbols", "fallback_symbols")
        self._candidates: Dict[str, List[str]] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self._targets and target.id not in self._candidates:
                values = self._literal_strings(node.value)
                if values:
                    self._candidates[target.id] = values
        self.generic_visit(node)

    @staticmethod
    def _literal_strings(node: ast.AST) -> List[str] | None:
        try:
            value = ast.literal_eval(node)
        except Exception:
            return None
        if isinstance(value, (list, tuple)):
            strings: List[str] = []
            for item in value:
                if isinstance(item, str):
                    strings.append(item.strip().upper())
            return strings
        return None

    def symbols(self) -> List[str] | None:
        for key in self._targets:
            candidate = self._candidates.get(key)
            if candidate:
                return candidate
        return None


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def load_trade_stock_symbols(script_path: Path | str = Path("trade_stock_e2e.py")) -> List[str]:
    """
    Extract the hard-coded symbol universe from ``trade_stock_e2e.py``.

    Args:
        script_path: Optional override path to the trading script.

    Returns:
        Ordered list of upper-case symbols exactly as declared in the trading loop.

    Raises:
        FileNotFoundError: if the script path does not exist.
        ValueError: if the symbols list cannot be located or parsed.
    """

    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Trade script not found: {path}")

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    extractor = _TradeStockSymbolExtractor()
    extractor.visit(tree)
    symbols = extractor.symbols()
    if not symbols:
        raise ValueError(f"Unable to locate symbol list inside {path}")
    return _dedupe_preserve(symbols)


__all__ = ["load_trade_stock_symbols"]
