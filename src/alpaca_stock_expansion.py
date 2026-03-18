from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import json

from src.hourly_data_utils import resolve_hourly_symbol_path


@dataclass(frozen=True)
class StockExpansionCandidate:
    symbol: str
    side: str = "long"
    sector: str = ""
    thesis: str = ""
    priority: int = 0

    def normalized(self) -> "StockExpansionCandidate":
        side = str(self.side or "long").strip().lower().replace("-", "_")
        side_aliases = {
            "long_only": "long",
            "short_only": "short",
            "long_short": "both",
            "both_sides": "both",
        }
        side = side_aliases.get(side, side)
        if side not in {"long", "short", "both"}:
            raise ValueError(f"Unsupported candidate side {self.side!r} for {self.symbol!r}")
        return StockExpansionCandidate(
            symbol=str(self.symbol or "").strip().upper(),
            side=side,
            sector=str(self.sector or "").strip(),
            thesis=str(self.thesis or "").strip(),
            priority=int(self.priority or 0),
        )


DEFAULT_STOCK_EXPANSION_CANDIDATES: tuple[StockExpansionCandidate, ...] = (
    StockExpansionCandidate("PLTR", sector="technology_services", thesis="AI software leader already showing some short-side edge.", priority=10),
    StockExpansionCandidate("NFLX", sector="technology_services", thesis="Large-cap software/media name with useful liquidity and prior seeded-short edge.", priority=9),
    StockExpansionCandidate("AVGO", sector="electronic_technology", thesis="Semiconductor/platform exposure beyond NVDA/AMD.", priority=8),
    StockExpansionCandidate("MU", sector="electronic_technology", thesis="Memory-cycle semiconductor exposure with high realized volatility.", priority=8),
    StockExpansionCandidate("ORCL", sector="technology_services", thesis="Enterprise software/cloud exposure outside mega-cap internet.", priority=8),
    StockExpansionCandidate("CSCO", sector="electronic_technology", thesis="Networking infrastructure anchor with different trend regime.", priority=7),
    StockExpansionCandidate("LRCX", sector="producer_manufacturing", thesis="Semicap equipment exposure complementary to NVDA/AMD.", priority=7),
    StockExpansionCandidate("AMAT", sector="producer_manufacturing", thesis="Semicap equipment with large-cap liquidity.", priority=7),
    StockExpansionCandidate("IBM", sector="technology_services", thesis="Older enterprise tech factor with AI narrative but different price structure.", priority=7),
    StockExpansionCandidate("CRM", sector="technology_services", thesis="Large-cap SaaS exposure.", priority=7),
    StockExpansionCandidate("ANET", sector="electronic_technology", thesis="AI-networking adjacency and strong trendiness.", priority=7),
    StockExpansionCandidate("SHOP", sector="commercial_services", thesis="High-beta software/ecommerce platform.", priority=7),
    StockExpansionCandidate("PANW", sector="technology_services", thesis="Cybersecurity leader with software-like behavior.", priority=7),
    StockExpansionCandidate("INTU", sector="technology_services", thesis="Stable software compounder that may smooth PnL.", priority=6),
    StockExpansionCandidate("NOW", sector="technology_services", thesis="Large-cap workflow SaaS exposure.", priority=6),
    StockExpansionCandidate("UBER", sector="transportation", thesis="High-liquidity platform business with different intraday regime.", priority=6),
    StockExpansionCandidate("BKNG", sector="consumer_services", thesis="Travel platform exposure; complements existing short basket dynamics.", priority=5),
    StockExpansionCandidate("WMT", sector="retail_trade", thesis="Defensive mega-cap retail anchor for diversity.", priority=4),
    StockExpansionCandidate("JPM", sector="finance", thesis="Highly liquid finance bellwether for regime diversification.", priority=4),
    StockExpansionCandidate("V", sector="finance", thesis="Payments network with resilient trend behavior.", priority=4),
)


def default_stock_expansion_candidates() -> list[StockExpansionCandidate]:
    return [candidate.normalized() for candidate in DEFAULT_STOCK_EXPANSION_CANDIDATES]


def candidates_to_jsonable(candidates: Sequence[StockExpansionCandidate]) -> list[dict[str, Any]]:
    return [asdict(candidate.normalized()) for candidate in candidates]


def write_stock_expansion_manifest(
    path: Path,
    *,
    base_stock_universe: str,
    default_checkpoint: str,
    candidates: Sequence[StockExpansionCandidate],
) -> Path:
    payload = {
        "base_stock_universe": str(base_stock_universe),
        "default_checkpoint": str(default_checkpoint),
        "candidates": candidates_to_jsonable(candidates),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_stock_expansion_manifest(path: Path) -> tuple[str, str, list[StockExpansionCandidate]]:
    payload = json.loads(Path(path).read_text())
    base_stock_universe = str(payload.get("base_stock_universe") or "").strip()
    default_checkpoint = str(payload.get("default_checkpoint") or "").strip()
    raw_candidates = payload.get("candidates") or []
    candidates = [candidate_from_mapping(row) for row in raw_candidates]
    return base_stock_universe, default_checkpoint, candidates


def candidate_from_mapping(payload: Mapping[str, Any]) -> StockExpansionCandidate:
    return StockExpansionCandidate(
        symbol=str(payload.get("symbol") or ""),
        side=str(payload.get("side") or "long"),
        sector=str(payload.get("sector") or ""),
        thesis=str(payload.get("thesis") or ""),
        priority=int(payload.get("priority") or 0),
    ).normalized()


def filter_candidates_with_hourly_data(
    candidates: Sequence[StockExpansionCandidate],
    *,
    data_root: Path,
) -> tuple[list[StockExpansionCandidate], list[StockExpansionCandidate]]:
    ready, _, missing = split_candidates_by_history(candidates, data_root=data_root)
    return ready, missing


def count_candidate_history_rows(symbol: str, *, data_root: Path) -> int:
    resolved = resolve_hourly_symbol_path(str(symbol or "").strip().upper(), Path(data_root))
    if resolved is None or not resolved.exists():
        return 0
    rows = 0
    with resolved.open("r", encoding="utf-8") as handle:
        for rows, _line in enumerate(handle, start=0):
            pass
    return max(0, rows)


def split_candidates_by_history(
    candidates: Sequence[StockExpansionCandidate],
    *,
    data_root: Path,
    min_history_rows: int = 0,
) -> tuple[list[StockExpansionCandidate], list[StockExpansionCandidate], list[StockExpansionCandidate]]:
    ready: list[StockExpansionCandidate] = []
    insufficient: list[StockExpansionCandidate] = []
    missing: list[StockExpansionCandidate] = []
    for candidate in candidates:
        normalized = candidate.normalized()
        history_rows = count_candidate_history_rows(normalized.symbol, data_root=Path(data_root))
        if history_rows <= 0:
            missing.append(normalized)
        elif min_history_rows > 0 and history_rows < int(min_history_rows):
            insufficient.append(normalized)
        else:
            ready.append(normalized)
    return ready, insufficient, missing


def extract_reforecast_metrics(summary: Mapping[str, Any], scenario: str) -> Optional[dict[str, float | bool | str]]:
    modes = summary.get("modes")
    if not isinstance(modes, list):
        return None
    for mode_row in modes:
        scenarios = mode_row.get("scenarios")
        if not isinstance(scenarios, list):
            continue
        for row in scenarios:
            if str(row.get("scenario") or "") != scenario:
                continue
            metrics = row.get("metrics") or {}
            return {
                "total_return": float(metrics.get("total_return", 0.0) or 0.0),
                "sortino": float(metrics.get("sortino", 0.0) or 0.0),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0) or 0.0),
                "pnl_abs": float(metrics.get("pnl_abs", 0.0) or 0.0),
                "periods": int(metrics.get("periods", 0) or 0),
                "terminated_early": bool(metrics.get("terminated_early", False)),
                "termination_reason": str(metrics.get("termination_reason") or ""),
            }
    return None


def summarize_reforecast_result(summary: Mapping[str, Any]) -> dict[str, Any]:
    best = summary.get("best_mode_metrics") or {}
    return {
        "best_mode": str(summary.get("best_mode") or ""),
        "best_scenario": str((summary.get("modes") or [{}])[0].get("best_scenario") or ""),
        "best_total_return": float(best.get("total_return", 0.0) or 0.0),
        "best_sortino": float(best.get("sortino", 0.0) or 0.0),
        "best_max_drawdown": float(best.get("max_drawdown", 0.0) or 0.0),
        "flat": extract_reforecast_metrics(summary, "flat"),
    }


def stock_expansion_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    flat = row.get("flat") or {}
    return (
        float(flat.get("sortino", float("-inf"))),
        float(flat.get("total_return", float("-inf"))),
        float(row.get("best_sortino", float("-inf"))),
        float(row.get("best_total_return", float("-inf"))),
    )


def candidate_lora_command(
    symbol: str,
    *,
    data_root: str = "trainingdatahourly/stocks",
    output_root: str = "chronos2_finetuned",
    num_steps: int = 1500,
    learning_rate: float = 5e-5,
    context_length: int = 1024,
) -> str:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        raise ValueError("symbol is required")
    return (
        "python scripts/retrain_chronos2_hourly_loras.py "
        f"--symbol {normalized} "
        f"--data-root {data_root} "
        f"--output-root {output_root} "
        f"--context-length {int(context_length)} "
        f"--learning-rate {float(learning_rate)} "
        f"--num-steps {int(num_steps)} "
        f"--save-name-suffix stockexp"
    )


__all__ = [
    "DEFAULT_STOCK_EXPANSION_CANDIDATES",
    "StockExpansionCandidate",
    "candidate_from_mapping",
    "candidate_lora_command",
    "count_candidate_history_rows",
    "candidates_to_jsonable",
    "default_stock_expansion_candidates",
    "extract_reforecast_metrics",
    "filter_candidates_with_hourly_data",
    "load_stock_expansion_manifest",
    "split_candidates_by_history",
    "stock_expansion_sort_key",
    "summarize_reforecast_result",
    "write_stock_expansion_manifest",
]
