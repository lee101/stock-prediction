from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import json
import pandas as pd

from src.chronos2_objective import build_correlation_cohort_map, build_correlation_matrix
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
    StockExpansionCandidate("OKLO", sector="utilities", thesis="Nuclear-power growth exposure with policy/news regime distinct from software and semis.", priority=8),
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


def build_candidate_sector_buckets(
    candidates: Sequence[StockExpansionCandidate],
    *,
    include_symbols: Sequence[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Group symbols into deterministic sector buckets for wide-universe planning."""
    include_filter = {
        str(symbol or "").strip().upper()
        for symbol in (include_symbols or ())
        if str(symbol or "").strip()
    }
    buckets: dict[str, list[tuple[int, str]]] = {}
    for candidate in candidates:
        normalized = candidate.normalized()
        if include_filter and normalized.symbol not in include_filter:
            continue
        sector = str(normalized.sector or "").strip().lower() or "unspecified"
        buckets.setdefault(sector, []).append((int(normalized.priority), normalized.symbol))

    result: dict[str, tuple[str, ...]] = {}
    for sector, rows in sorted(buckets.items()):
        ranked = sorted(rows, key=lambda row: (-row[0], row[1]))
        result[sector] = tuple(symbol for _priority, symbol in ranked)
    return result


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
    side_defaults = manifest_side_defaults(payload)
    candidates = [
        candidate_from_mapping(
            row,
            default_side=side_defaults.get(str((row or {}).get("symbol") or "").strip().upper(), "long"),
        )
        for row in raw_candidates
    ]
    return base_stock_universe, default_checkpoint, candidates


def manifest_side_defaults(payload: Mapping[str, Any]) -> dict[str, str]:
    raw_policy = payload.get("side_policy")
    if not isinstance(raw_policy, Mapping):
        return {}

    defaults: dict[str, str] = {}

    def _apply(symbols: Any, side: str) -> None:
        if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)):
            return
        for symbol in symbols:
            normalized = str(symbol or "").strip().upper()
            if normalized:
                defaults[normalized] = side

    _apply(raw_policy.get("long_only_symbols"), "long")
    _apply(raw_policy.get("already_live_long_only_symbols"), "long")
    _apply(raw_policy.get("short_only_symbols"), "short")
    _apply(raw_policy.get("evaluate_both_sides_symbols"), "both")
    _apply(raw_policy.get("both_sides_symbols"), "both")
    return defaults


def candidate_from_mapping(payload: Mapping[str, Any], *, default_side: str = "long") -> StockExpansionCandidate:
    return StockExpansionCandidate(
        symbol=str(payload.get("symbol") or ""),
        side=str(payload.get("side") or default_side or "long"),
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


def resolve_hourly_hyperparams_path(
    symbol: str,
    *,
    hyperparam_root: Path = Path("hyperparams"),
) -> Path:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        raise ValueError("symbol is required")
    return Path(hyperparam_root) / "chronos2" / "hourly" / f"{normalized}.json"


def has_hourly_hyperparams(
    symbol: str,
    *,
    hyperparam_root: Path = Path("hyperparams"),
) -> bool:
    return resolve_hourly_hyperparams_path(symbol, hyperparam_root=hyperparam_root).exists()


def _resolve_hourly_context_length(
    symbol: str,
    *,
    hyperparam_root: Path = Path("hyperparams"),
    default: int = 1024,
) -> int:
    path = resolve_hourly_hyperparams_path(symbol, hyperparam_root=hyperparam_root)
    if not path.exists():
        return int(default)
    try:
        payload = json.loads(path.read_text())
        config = payload.get("config") if isinstance(payload, Mapping) else {}
        context_length = int((config or {}).get("context_length", default))
    except Exception:
        return int(default)
    return max(1, int(context_length))


def build_hourly_return_correlation_cohorts(
    symbols: Sequence[str],
    *,
    data_root: Path,
    lookback_hours: int = 24 * 120,
    min_periods: int = 24 * 5,
    max_size: int = 4,
    min_abs_corr: float = 0.25,
    include_negative: bool = False,
) -> dict[str, tuple[str, ...]]:
    normalized_symbols: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = str(symbol or "").strip().upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_symbols.append(normalized)

    close_history: dict[str, pd.Series] = {}
    data_root = Path(data_root)
    for symbol in normalized_symbols:
        csv_path = resolve_hourly_symbol_path(symbol, data_root)
        if csv_path is None or not csv_path.exists():
            continue
        try:
            frame = pd.read_csv(csv_path, usecols=["timestamp", "close"])
        except Exception:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.floor("h")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
        if lookback_hours > 0 and len(frame) > int(lookback_hours) + 24:
            frame = frame.tail(int(lookback_hours) + 24).copy()
        if frame.empty:
            continue
        close_history[symbol] = pd.Series(
            frame["close"].to_numpy(dtype="float64"),
            index=frame["timestamp"],
        )

    corr = build_correlation_matrix(
        close_history,
        lookback=int(lookback_hours) if lookback_hours > 0 else None,
        min_periods=int(min_periods),
    )
    return build_correlation_cohort_map(
        corr,
        max_size=int(max_size),
        min_abs_corr=float(min_abs_corr),
        include_negative=bool(include_negative),
    )


def candidate_hourly_tuning_command(
    symbol: str,
    *,
    holdout_hours: int = 168,
    prediction_length: int = 24,
    cohort_size: int = 4,
    cohort_min_abs_corr: float = 0.25,
) -> str:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        raise ValueError("symbol is required")
    return (
        "python hyperparam_chronos_hourly.py "
        f"--symbols {normalized} "
        "--quick "
        f"--holdout-hours {int(holdout_hours)} "
        f"--prediction-length {int(prediction_length)} "
        "--objective composite "
        f"--cohort-size {int(cohort_size)} "
        f"--cohort-min-abs-corr {float(cohort_min_abs_corr)} "
        "--save-hyperparams"
    )


def candidate_lora_command(
    symbol: str,
    *,
    data_root: str = "trainingdatahourly/stocks",
    output_root: str = "chronos2_finetuned",
    num_steps: int = 1500,
    learning_rate: float = 5e-5,
    context_length: int = 1024,
    covariate_symbols: Sequence[str] | None = None,
    covariate_cols: Sequence[str] | None = ("close",),
) -> str:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        raise ValueError("symbol is required")
    command = (
        "python scripts/retrain_chronos2_hourly_loras.py "
        f"--symbol {normalized} "
        f"--data-root {data_root} "
        f"--output-root {output_root} "
        f"--context-length {int(context_length)} "
        f"--learning-rate {float(learning_rate)} "
        f"--num-steps {int(num_steps)} "
        f"--save-name-suffix stockexp"
    )
    normalized_covariates = [
        str(peer or "").strip().upper()
        for peer in (covariate_symbols or ())
        if str(peer or "").strip()
    ]
    if normalized_covariates:
        command += f" --covariate-symbols {','.join(normalized_covariates)}"
    normalized_covariate_cols = [str(col or "").strip() for col in (covariate_cols or ()) if str(col or "").strip()]
    if normalized_covariates and normalized_covariate_cols:
        command += f" --covariate-cols {','.join(normalized_covariate_cols)}"
    return command


def candidate_training_plan(
    symbol: str,
    *,
    comparison_symbols: Sequence[str],
    data_root: Path = Path("trainingdatahourly/stocks"),
    hyperparam_root: Path = Path("hyperparams"),
    correlation_cohorts: Mapping[str, Sequence[str]] | None = None,
    lookback_hours: int = 24 * 120,
    min_periods: int = 24 * 5,
    max_cohort_size: int = 4,
    min_abs_corr: float = 0.25,
    include_negative: bool = False,
    tuning_holdout_hours: int = 168,
    tuning_prediction_length: int = 24,
    lora_num_steps: int = 1500,
    lora_learning_rate: float = 5e-5,
    default_context_length: int = 1024,
) -> dict[str, Any]:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        raise ValueError("symbol is required")

    if correlation_cohorts is None:
        symbols = [normalized, *comparison_symbols]
        correlation_cohorts = build_hourly_return_correlation_cohorts(
            symbols,
            data_root=Path(data_root),
            lookback_hours=int(lookback_hours),
            min_periods=int(min_periods),
            max_size=int(max_cohort_size),
            min_abs_corr=float(min_abs_corr),
            include_negative=bool(include_negative),
        )

    peers = tuple(
        str(peer or "").strip().upper()
        for peer in correlation_cohorts.get(normalized, ())
        if str(peer or "").strip() and str(peer or "").strip().upper() != normalized
    )
    hourly_config_exists = has_hourly_hyperparams(normalized, hyperparam_root=Path(hyperparam_root))
    context_length = _resolve_hourly_context_length(
        normalized,
        hyperparam_root=Path(hyperparam_root),
        default=int(default_context_length),
    )
    lora_context_length = context_length if hourly_config_exists else int(default_context_length)
    if not hourly_config_exists:
        recommended_next_step = "hourly_tune"
        recommended_reason = "No stored hourly Chronos2 config; tune base hourly params before LoRA."
    elif peers:
        recommended_next_step = "multivariate_lora"
        recommended_reason = f"Use correlated peers to test multivariate LoRA: {', '.join(peers)}"
    else:
        recommended_next_step = "single_symbol_lora"
        recommended_reason = "Hourly config exists; proceed with single-symbol LoRA baseline."

    return {
        "hourly_config_exists": hourly_config_exists,
        "recommended_next_step": recommended_next_step,
        "recommended_reason": recommended_reason,
        "recommended_peer_symbols": list(peers),
        "hourly_tuning_command": candidate_hourly_tuning_command(
            normalized,
            holdout_hours=int(tuning_holdout_hours),
            prediction_length=int(tuning_prediction_length),
            cohort_size=int(max_cohort_size),
            cohort_min_abs_corr=float(min_abs_corr),
        ),
        "lora_command": candidate_lora_command(
            normalized,
            context_length=int(lora_context_length),
            learning_rate=float(lora_learning_rate),
            num_steps=int(lora_num_steps),
            covariate_symbols=peers,
        ),
    }


_SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_DEFAULT_SP500_CACHE = "trainingdatadaily/stocks/sp500_symbols.txt"


def get_sp500_symbols(
    use_cache: bool = True,
    cache_file: str = _DEFAULT_SP500_CACHE,
) -> list[str]:
    """Return S&P500 constituent symbols.

    If use_cache is True and cache_file exists, reads from the cached file.
    Otherwise fetches the current list from Wikipedia and (if use_cache is True)
    saves it to cache_file for future calls.
    """
    import pandas as pd  # noqa: PLC0415

    cache_path = Path(cache_file)
    if use_cache and cache_path.exists():
        lines = cache_path.read_text(encoding="utf-8").splitlines()
        return [line.strip().upper() for line in lines if line.strip() and not line.startswith("#")]

    tables = pd.read_html(_SP500_WIKIPEDIA_URL)
    df = tables[0]
    col = next((c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker")), df.columns[0])
    symbols = [str(s).strip().replace(".", "-") for s in df[col].tolist() if str(s).strip()]

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(symbols) + "\n", encoding="utf-8")

    return symbols


__all__ = [
    "DEFAULT_STOCK_EXPANSION_CANDIDATES",
    "StockExpansionCandidate",
    "build_candidate_sector_buckets",
    "build_hourly_return_correlation_cohorts",
    "candidate_from_mapping",
    "candidate_hourly_tuning_command",
    "candidate_lora_command",
    "candidate_training_plan",
    "count_candidate_history_rows",
    "candidates_to_jsonable",
    "default_stock_expansion_candidates",
    "extract_reforecast_metrics",
    "filter_candidates_with_hourly_data",
    "get_sp500_symbols",
    "has_hourly_hyperparams",
    "load_stock_expansion_manifest",
    "manifest_side_defaults",
    "resolve_hourly_hyperparams_path",
    "split_candidates_by_history",
    "stock_expansion_sort_key",
    "summarize_reforecast_result",
    "write_stock_expansion_manifest",
]
