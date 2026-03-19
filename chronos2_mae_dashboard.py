#!/usr/bin/env python3
"""Chronos2 MAE Dashboard for Trading Symbols.

Shows forecast MAE for all trading symbols and identifies which need LoRA improvement.
Reads JSON configs from:
  - preaugstrategies/best/hourly/{SYMBOL}.json  (preaug MAE%)
  - best/{SYMBOL}.json                          (overall best model MAE)
  - hyperparams/chronos2/hourly/{SYMBOL}.json   (chronos2 baseline MAE)
  - hyperparams/crypto_lora_sweep/               (LoRA sweep results)
  - chronos2_finetuned/                          (existence of LoRA models)
"""

import csv
import json
import sys
from pathlib import Path


def find_project_roots():
    """Find the project root(s).

    Returns (worktree_root, main_root) -- worktree_root is the directory
    containing this script; main_root is the actual repo root (may be the
    same, or may differ when running inside a git worktree).  Data dirs
    like chronos2_finetuned/ and best/ live only in the main root.
    """
    here = Path(__file__).resolve().parent
    worktree_root = here
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            worktree_root = p
            break

    # Detect git worktree: .git is a file with "gitdir: ..." content
    git_path = worktree_root / ".git"
    main_root = worktree_root
    if git_path.is_file():
        text = git_path.read_text().strip()
        if text.startswith("gitdir:"):
            # e.g. gitdir: /repo/.git/worktrees/name -> main root is /repo
            gitdir = Path(text.split(":", 1)[1].strip())
            # Walk up from .git/worktrees/name to .git, then to repo root
            candidate = gitdir.resolve()
            while candidate.name != ".git" and candidate != candidate.parent:
                candidate = candidate.parent
            if candidate.name == ".git":
                main_root = candidate.parent

    return worktree_root, main_root


WORKTREE_ROOT, MAIN_ROOT = find_project_roots()

# Trading universe definitions
CRYPTO5 = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
CRYPTO_EXTRA = ["LINKUSD", "UNIUSD", "DOGEUSD", "AAVEUSD"]

STOCKS_CORE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA",
    "AMD", "PLTR", "NET", "COIN", "ADBE", "INTC",
]

STOCKS_EXTENDED = [
    "ADSK", "COUR", "CRM", "DBX", "EBAY", "EXPE", "GE", "GS",
    "MTCH", "NFLX", "NYT", "QUBT", "SAP", "SHOP", "SONY",
    "TRIP", "U", "YELP", "BKNG", "LLY", "MA", "V",
    "BAC", "JPM", "WMT", "XOM",
]

# Binance USDT symbols (separate category)
BINANCE_USDT = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "UNIUSDT", "AAVEUSDT", "ATOMUSDT", "APTUSDT",
    "BCHUSDT", "NEARUSDT", "SUIUSDT", "SHIBUSDT", "TIAUSDT",
    "SEIUSDT", "FETUSDT", "FILUSDT", "ICPUSDT", "INJUSDT",
    "OPUSDT", "PEPEUSDT", "BONKUSDT", "WLDUSDT", "TRXUSDT",
    "POLUSDT",
]

ALL_SYMBOLS = sorted(set(
    CRYPTO5 + CRYPTO_EXTRA + STOCKS_CORE + STOCKS_EXTENDED + BINANCE_USDT
))


def load_json(path):
    """Load a JSON file, returning None if it doesn't exist or is invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _resolve(relpath):
    """Find a file/dir in either the worktree root or the main repo root."""
    for root in [WORKTREE_ROOT, MAIN_ROOT]:
        candidate = root / relpath
        if candidate.exists():
            return candidate
    return None


def get_preaug_data(symbol):
    """Get preaug strategy data for a symbol."""
    path = _resolve(Path("preaugstrategies") / "best" / "hourly" / f"{symbol}.json")
    return load_json(path) if path else None


def get_best_data(symbol):
    """Get best model config data for a symbol."""
    path = _resolve(Path("best") / f"{symbol}.json")
    return load_json(path) if path else None


def get_chronos2_hourly_data(symbol):
    """Get chronos2 hourly hyperparams for a symbol."""
    path = _resolve(Path("hyperparams") / "chronos2" / "hourly" / f"{symbol}.json")
    return load_json(path) if path else None


def _build_lora_sweep_index():
    """Scan LoRA sweep directories once and build {symbol: best_result} index."""
    index = {}  # symbol -> (mae_percent_mean, data_dict)
    for root in [WORKTREE_ROOT, MAIN_ROOT]:
        sweep_dir = root / "hyperparams" / "crypto_lora_sweep"
        if not sweep_dir.exists():
            continue
        for f in sweep_dir.iterdir():
            if not f.name.endswith(".json") or "_lora_" not in f.name:
                continue
            symbol = f.name.split("_lora_", 1)[0]
            data = load_json(f)
            if data is None:
                continue
            mae_pct = data.get("val", {}).get("mae_percent_mean")
            if mae_pct is None:
                continue
            prev_mae = index.get(symbol, (float("inf"), None))[0]
            if mae_pct < prev_mae:
                index[symbol] = (mae_pct, data)
    return {sym: data for sym, (_mae, data) in index.items()}


def _build_lora_model_counts():
    """Scan chronos2_finetuned/ once and build {symbol: count} index."""
    counts = {}
    seen = set()
    for root in [WORKTREE_ROOT, MAIN_ROOT]:
        finetuned_dir = root / "chronos2_finetuned"
        if not finetuned_dir.exists():
            continue
        for d in finetuned_dir.iterdir():
            if not d.is_dir() or "_lora_" not in d.name or d.name in seen:
                continue
            seen.add(d.name)
            symbol = d.name.split("_lora_", 1)[0]
            counts[symbol] = counts.get(symbol, 0) + 1
    return counts


_CRYPTO5_SET = set(CRYPTO5)
_CRYPTO_EXTRA_SET = set(CRYPTO_EXTRA)
_BINANCE_USDT_SET = set(BINANCE_USDT)
_STOCKS_CORE_SET = set(STOCKS_CORE)


def classify_symbol(symbol):
    """Classify a symbol into its group."""
    if symbol in _CRYPTO5_SET:
        return "crypto5"
    if symbol in _CRYPTO_EXTRA_SET:
        return "crypto_extra"
    if symbol in _BINANCE_USDT_SET:
        return "binance_usdt"
    if symbol in _STOCKS_CORE_SET:
        return "stock_core"
    return "stock_extended"


def compute_priority(mae_pct, has_lora, lora_count):
    """Compute a priority score (higher = more urgent for retraining).

    Factors:
      - Higher MAE% = higher priority
      - No LoRA = higher priority
      - Fewer LoRA experiments = higher priority
    """
    if mae_pct is None:
        return 0.0  # No data, can't prioritize
    score = mae_pct * 100  # Scale up for readability
    if not has_lora:
        score *= 2.0  # Double priority if no LoRA exists
    elif lora_count < 3:
        score *= 1.5  # Moderate boost if few experiments
    return round(score, 2)


def build_dashboard():
    """Build the full dashboard data for all trading symbols."""
    lora_sweep_index = _build_lora_sweep_index()
    lora_model_counts = _build_lora_model_counts()
    rows = []

    for symbol in ALL_SYMBOLS:
        preaug = get_preaug_data(symbol)
        best = get_best_data(symbol)
        chronos2_hourly = get_chronos2_hourly_data(symbol)
        lora_sweep = lora_sweep_index.get(symbol)
        lora_count = lora_model_counts.get(symbol, 0)

        # Extract preaug MAE%
        preaug_mae_pct = None
        preaug_strategy = None
        if preaug:
            preaug_mae_pct = preaug.get("mae_percent")
            preaug_strategy = preaug.get("best_strategy")

        # Extract chronos2 hourly baseline MAE
        chronos2_val_mae = None
        if chronos2_hourly:
            val = chronos2_hourly.get("validation", {})
            chronos2_val_mae = val.get("pct_return_mae")

        # Extract best model MAE (may be toto, chronos2, or kronos)
        best_model = None
        best_val_mae = None
        chronos2_candidate_mae = None
        if best:
            best_model = best.get("model")
            val = best.get("validation", {})
            best_val_mae = val.get("pct_return_mae")
            # Also get chronos2-specific MAE from candidates
            meta = best.get("metadata", {})
            chronos2_candidate_mae = meta.get("chronos2_pct_return_mae")

        # Extract LoRA sweep best MAE
        lora_mae_pct = None
        if lora_sweep:
            lora_mae_pct = lora_sweep.get("val", {}).get("mae_percent_mean")

        # Compute improvement from LoRA over preaug baseline
        lora_improvement_pct = None
        if lora_mae_pct is not None and preaug_mae_pct is not None and preaug_mae_pct > 0:
            lora_improvement_pct = (preaug_mae_pct - lora_mae_pct) / preaug_mae_pct * 100

        # Use preaug MAE% as the primary metric; fall back to chronos2 val MAE
        primary_mae = preaug_mae_pct
        if primary_mae is None and chronos2_val_mae is not None:
            # Convert pct_return_mae to percentage for comparability
            primary_mae = chronos2_val_mae * 100
        if primary_mae is None and chronos2_candidate_mae is not None:
            primary_mae = chronos2_candidate_mae * 100

        priority = compute_priority(primary_mae, lora_count > 0, lora_count)

        rows.append({
            "symbol": symbol,
            "group": classify_symbol(symbol),
            "mae_pct": primary_mae,
            "best_preaug": preaug_strategy or "",
            "chronos2_val_mae": chronos2_val_mae,
            "best_model": best_model or "",
            "best_val_mae": best_val_mae,
            "has_lora": "Y" if lora_count > 0 else "N",
            "lora_count": lora_count,
            "lora_mae_pct": lora_mae_pct,
            "lora_improvement_pct": lora_improvement_pct,
            "priority": priority,
        })

    # Sort by priority descending (worst MAE / no LoRA first)
    rows.sort(key=lambda r: r["priority"], reverse=True)
    return rows


def fmt(val, decimals=4):
    """Format a numeric value for display, or return '-' if None."""
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"


def print_table(rows):
    """Print the dashboard as a formatted table."""
    # Header
    header = (
        f"{'Symbol':<12} {'Group':<14} {'MAE%':<10} {'Preaug':<14} "
        f"{'C2 ValMAE':<12} {'BestModel':<10} {'LoRA?':<6} {'#LoRA':<6} "
        f"{'LoRA MAE%':<10} {'Improv%':<10} {'Priority':<10}"
    )
    print("=" * len(header))
    print("CHRONOS2 MAE DASHBOARD — Forecast Quality by Symbol")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in rows:
        line = (
            f"{r['symbol']:<12} {r['group']:<14} {fmt(r['mae_pct']):<10} "
            f"{r['best_preaug']:<14} {fmt(r['chronos2_val_mae'], 6):<12} "
            f"{r['best_model']:<10} {r['has_lora']:<6} {r['lora_count']:<6} "
            f"{fmt(r['lora_mae_pct']):<10} {fmt(r['lora_improvement_pct'], 1):<10} "
            f"{fmt(r['priority'], 1):<10}"
        )
        print(line)

    print("-" * len(header))


def print_summary(rows):
    """Print summary statistics."""
    total = len(rows)
    with_lora = sum(1 for r in rows if r["has_lora"] == "Y")
    without_lora = total - with_lora
    mae_values = [r["mae_pct"] for r in rows if r["mae_pct"] is not None]
    avg_mae = sum(mae_values) / len(mae_values) if mae_values else 0

    with_preaug = sum(1 for r in rows if r["best_preaug"])
    with_chronos2 = sum(1 for r in rows if r["chronos2_val_mae"] is not None)
    with_best = sum(1 for r in rows if r["best_model"])

    print()
    print("SUMMARY")
    print(f"  Total symbols:            {total}")
    print(f"  With preaug data:         {with_preaug}")
    print(f"  With chronos2 hourly:     {with_chronos2}")
    print(f"  With best-model config:   {with_best}")
    print(f"  With LoRA models:         {with_lora}")
    print(f"  Need LoRA retraining:     {without_lora}")
    print(f"  Average MAE%:             {avg_mae:.4f}")
    print()

    # Top 10 priorities
    top10 = [r for r in rows if r["priority"] > 0][:10]
    if top10:
        print("TOP 10 RETRAIN PRIORITIES:")
        for i, r in enumerate(top10, 1):
            lora_status = f"has {r['lora_count']} LoRA(s)" if r["lora_count"] > 0 else "NO LoRA"
            print(f"  {i:2d}. {r['symbol']:<12} MAE%={fmt(r['mae_pct'])} priority={fmt(r['priority'], 1)} ({lora_status})")


def save_csv(rows, path):
    """Save the dashboard to a CSV file."""
    fieldnames = [
        "symbol", "group", "mae_pct", "best_preaug", "chronos2_val_mae",
        "best_model", "best_val_mae", "has_lora", "lora_count",
        "lora_mae_pct", "lora_improvement_pct", "priority",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {path}")


def main():
    rows = build_dashboard()
    print_table(rows)
    print_summary(rows)

    csv_path = WORKTREE_ROOT / "chronos2_mae_dashboard.csv"
    save_csv(rows, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
