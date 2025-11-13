from __future__ import annotations

import os
import signal
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from src.hourly_data_utils import (
    HourlyDataIssue,
    HourlyDataStatus,
    HourlyDataValidator,
    discover_hourly_symbols,
    summarize_statuses,
)
from src.portfolio_filters import filter_positive_forecasts
from src.hourly_scheduler import HourlyRunCoordinator, resolve_hourly_symbols
from src.hourly_data_refresh import HourlyDataRefresher
from src.hourly_pnl_gate import should_block_trade_by_pnl
from src.logging_utils import setup_logging, get_log_filename
from src.symbol_filtering import filter_symbols_by_tradable_pairs, get_filter_info
from src.symbol_utils import is_crypto_symbol
from src.trade_stock_forecast_snapshot import reset_forecast_cache

# Use a dedicated trade-state namespace so hourly runs never touch daily files.
os.environ.setdefault("TRADE_STATE_SUFFIX", "hourly")
os.environ.setdefault("CHRONOS2_FREQUENCY", "hourly")

REPO_ROOT = Path(__file__).resolve().parent
HOURLY_RESULTS_DIR = REPO_ROOT / "results_hourly"
HOURLY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Lazily load heavy trade stack once env overrides are applied.
import trade_stock_e2e as base  # noqa: E402
from src.date_utils import is_nyse_trading_day_now  # noqa: E402

logger = setup_logging(get_log_filename("trade_stock_e2e.log", is_hourly=True))
base.logger = logger


def _hourly_results_dir() -> Path:
    return HOURLY_RESULTS_DIR


base._results_dir = _hourly_results_dir  # type: ignore[attr-defined]
reset_forecast_cache()

EST = ZoneInfo("America/New_York")
DEFAULT_HOURLY_SYMBOLS: List[str] = [
    # Top performing equities (high Sharpe, good win rates)
    "EQIX",
    "GS",
    "COST",
    "CRM",
    "AXP",
    "BA",
    "GE",
    "LLY",
    "AVGO",
    "SPY",
    "SHOP",
    "GLD",
    "PLTR",
    "MCD",
    "V",
    "VTI",
    "QQQ",
    "MA",
    "SAP",
    # Keep existing profitable ones
    "ADBE",
    "COUR",
    # Top crypto performers
    "BTCUSD",
    "ETHUSD",
    "LINKUSD",
    "UNIUSD",
    "PAXGUSD",
]
SYMBOL_FILES = [
    REPO_ROOT / "symbolsofinterest.txt",
    REPO_ROOT / "hourly_symbols.txt",
]
HOURLY_ANALYSIS_WINDOW_MINUTES = int(os.getenv("HOURLY_ANALYSIS_WINDOW_MINUTES", "12"))
HOURLY_LOOP_SLEEP_SECONDS = max(15, int(os.getenv("HOURLY_LOOP_SLEEP_SECONDS", "60")))
HOURLY_MARKET_CLOSE_WINDOW = int(os.getenv("HOURLY_MARKET_CLOSE_WINDOW", "25"))
HOURLY_DATA_ROOT = REPO_ROOT / "trainingdatahourly"
HOURLY_DATA_MAX_STALENESS_HOURS = max(1, int(os.getenv("HOURLY_DATA_MAX_STALENESS_HOURS", "6")))
_TRUTHY = {"1", "true", "yes", "on"}
HOURLY_FAIL_ON_DATA_GAPS = os.getenv("HOURLY_FAIL_ON_DATA_GAPS", "0").strip().lower() in _TRUTHY
HOURLY_DATA_VALIDATOR = HourlyDataValidator(
    HOURLY_DATA_ROOT,
    max_staleness_hours=HOURLY_DATA_MAX_STALENESS_HOURS,
)
HOURLY_REQUIRE_POSITIVE_FORECAST = os.getenv("HOURLY_REQUIRE_POSITIVE_FORECAST", "1").strip().lower() in _TRUTHY
HOURLY_REQUIRE_POSITIVE_AVG_RETURN = os.getenv("HOURLY_REQUIRE_POSITIVE_AVG_RETURN", "1").strip().lower() in _TRUTHY
HOURLY_ENABLE_PNL_GATE = os.getenv("HOURLY_ENABLE_PNL_GATE", "1").strip().lower() in _TRUTHY
HOURLY_PNL_GATE_MAX_TRADES = max(1, int(os.getenv("HOURLY_PNL_GATE_MAX_TRADES", "2")))
HOURLY_CRYPTO_MAX_STALENESS_HOURS = max(0.1, float(os.getenv("HOURLY_CRYPTO_MAX_STALENESS_HOURS", "1.5")))
HOURLY_REFRESH_BACKFILL_HOURS = max(6, int(os.getenv("HOURLY_REFRESH_BACKFILL_HOURS", "48")))
HOURLY_REFRESH_OVERLAP_HOURS = max(0, int(os.getenv("HOURLY_REFRESH_OVERLAP_HOURS", "2")))
HOURLY_DATA_REFRESHER = HourlyDataRefresher(
    HOURLY_DATA_ROOT,
    HOURLY_DATA_VALIDATOR,
    backfill_hours=HOURLY_REFRESH_BACKFILL_HOURS,
    overlap_hours=HOURLY_REFRESH_OVERLAP_HOURS,
    crypto_max_staleness_hours=HOURLY_CRYPTO_MAX_STALENESS_HOURS,
    logger_override=logger,
)


def _log_data_root_status() -> None:
    if HOURLY_DATA_ROOT.exists():
        logger.info("Hourly data root detected at %s", HOURLY_DATA_ROOT)
    else:
        logger.warning(
            "Hourly data root %s is missing; hourly fills will fall back to stale simulated bars",
            HOURLY_DATA_ROOT,
        )


def _hourly_statuses_with_refresh(symbols: Sequence[str]) -> Tuple[List[HourlyDataStatus], List[HourlyDataIssue]]:
    statuses, issues = HOURLY_DATA_REFRESHER.refresh(symbols)
    statuses, crypto_issues = _apply_crypto_strictness(statuses)
    if crypto_issues:
        issues = list(issues) + crypto_issues
    return statuses, issues


def _apply_crypto_strictness(statuses: Sequence[HourlyDataStatus]) -> Tuple[List[HourlyDataStatus], List[HourlyDataIssue]]:
    if HOURLY_CRYPTO_MAX_STALENESS_HOURS <= 0:
        return list(statuses), []
    crypto_issues: List[HourlyDataIssue] = []
    stale_symbols = {
        status.symbol
        for status in statuses
        if is_crypto_symbol(status.symbol) and status.staleness_hours > HOURLY_CRYPTO_MAX_STALENESS_HOURS
    }
    if not stale_symbols:
        return list(statuses), crypto_issues
    for status in statuses:
        if status.symbol in stale_symbols:
            crypto_issues.append(
                HourlyDataIssue(
                    symbol=status.symbol,
                    reason="stale",
                    detail=(
                        f"{status.symbol} hourly data is {status.staleness_hours:.2f}h old "
                        f"(crypto threshold {HOURLY_CRYPTO_MAX_STALENESS_HOURS:.2f}h)"
                    ),
                )
            )
    filtered = [status for status in statuses if status.symbol not in stale_symbols]
    return filtered, crypto_issues


def _validated_symbol_statuses(symbols: Sequence[str]) -> List[HourlyDataStatus]:
    statuses, issues = _hourly_statuses_with_refresh(symbols)
    for issue in issues:
        log_fn = logger.error if HOURLY_FAIL_ON_DATA_GAPS else logger.warning
        log_fn(
            "Excluding %s from hourly loop (%s): %s",
            issue.symbol,
            issue.reason,
            issue.detail,
        )
    if not statuses:
        raise RuntimeError("No hourly symbols passed data validation.")
    logger.info("Hourly data ready: %s", summarize_statuses(statuses))
    logger.debug(
        "Hourly data details: %s",
        "; ".join(
            f"{status.symbol}@{status.latest_timestamp.isoformat()} "
            f"(lag={status.staleness_hours:.2f}h, close={status.latest_close:.4f})"
            for status in statuses
        ),
    )
    return statuses


def _resolve_symbols() -> Tuple[List[str], List[HourlyDataStatus]]:
    env_value = os.getenv("HOURLY_TRADE_SYMBOLS")
    symbols = resolve_hourly_symbols(env_value, SYMBOL_FILES, DEFAULT_HOURLY_SYMBOLS)
    if not symbols:
        discovered = discover_hourly_symbols(HOURLY_DATA_ROOT)
        if discovered:
            logger.info(
                "Hourly symbol list empty; discovered %d symbols from %s",
                len(discovered),
                HOURLY_DATA_ROOT,
            )
            symbols = discovered
    if not symbols:
        raise RuntimeError("No symbols resolved for hourly trading loop")

    # Filter symbols by TRADABLE_PAIRS env var if set
    original_symbols = symbols
    symbols = filter_symbols_by_tradable_pairs(symbols)
    filter_info = get_filter_info(original_symbols, symbols)
    if filter_info["was_filtered"]:
        logger.info(
            "TRADABLE_PAIRS filter: %d/%d symbols selected: %s",
            filter_info["filtered_count"],
            filter_info["original_count"],
            ", ".join(symbols)
        )

    logger.info("Hourly trading universe resolved to %d symbols.", len(symbols))
    logger.debug("Hourly symbols (pre-validation): %s", ", ".join(symbols))
    statuses = _validated_symbol_statuses(symbols)
    validated = [status.symbol for status in statuses]
    if set(validated) != set(symbols):
        logger.info("Trading subset after hourly validation: %s", ", ".join(validated))
    return validated, statuses


class HourlyTradingEngine:
    def __init__(self) -> None:
        self.symbols, self._latest_data_statuses = _resolve_symbols()
        self.coordinator = HourlyRunCoordinator(
            analysis_window_minutes=HOURLY_ANALYSIS_WINDOW_MINUTES,
            allow_immediate_start=True,
            allow_catch_up=True,
        )
        self.previous_picks: Dict[str, Dict] = {}
        self.last_market_close_date: Optional[date] = None
        self._log_hourly_data_statuses(self._latest_data_statuses)

    def _filter_by_recent_pnl(
        self,
        picks: Dict[str, Dict],
        analysis: Dict[str, Dict],
    ) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """Filter picks based on recent PnL performance.

        Blocks trading on symbol+side pairs where the last 1-2 trades
        (depending on availability) had negative sum PnL.

        Args:
            picks: Portfolio picks to filter
            analysis: Analysis data for symbols (includes strategy info)

        Returns:
            Tuple of (filtered_picks, blocked_symbols_with_reasons)
        """
        if not HOURLY_ENABLE_PNL_GATE:
            return picks, {}

        filtered_picks = {}
        blocked = {}

        for symbol, pick_data in picks.items():
            side = pick_data.get("side", "buy")
            strategy = analysis.get(symbol, {}).get("strategy")

            should_block, reason = should_block_trade_by_pnl(
                base._get_trade_history_store,
                symbol,
                side,
                strategy=strategy,
                max_trades=HOURLY_PNL_GATE_MAX_TRADES,
                logger=logger,
            )

            if should_block:
                blocked[symbol] = reason
            else:
                filtered_picks[symbol] = pick_data

        return filtered_picks, blocked

    def run_cycle(self, now: datetime) -> None:
        logger.info("Starting hourly analysis @ %s", now.isoformat())
        try:
            statuses = self._ensure_hourly_data_ready()
        except RuntimeError as exc:
            logger.error("Skipping hourly cycle due to data validation failure: %s", exc)
            return
        self._log_hourly_data_statuses(statuses)
        analysis = base.analyze_symbols(self.symbols)
        self._log_strategy_mix(analysis)
        picks = base.build_portfolio(
            analysis,
            min_positions=base.DEFAULT_MIN_CORE_POSITIONS,
            max_positions=base.DEFAULT_MAX_PORTFOLIO,
            max_expanded=base.EXPANDED_PORTFOLIO,
        )
        picks, dropped = filter_positive_forecasts(
            picks,
            require_positive_forecast=HOURLY_REQUIRE_POSITIVE_FORECAST,
            require_positive_avg_return=HOURLY_REQUIRE_POSITIVE_AVG_RETURN,
        )
        if dropped:
            for symbol, record in dropped.items():
                logger.info(
                    "Hourly filter removed %s (forecast=%.4f avg=%.4f)",
                    symbol,
                    record.forecast,
                    record.avg_return,
                )

        # Apply PnL-based blocking for symbols with recent losses
        picks, pnl_blocked = self._filter_by_recent_pnl(picks, analysis)
        if pnl_blocked:
            for symbol, reason in pnl_blocked.items():
                logger.warning("Hourly PnL gate blocked %s: %s", symbol, reason)

        if not picks:
            logger.info("Hourly portfolio empty after filtering; skipping execution cycle.")
            self.coordinator.mark_executed(now)
            return
        base.log_trading_plan(picks, f"HOURLY {now.astimezone(EST):%Y-%m-%d %H:%M}")
        base.manage_positions(picks, self.previous_picks, analysis)
        self.previous_picks = picks
        self.coordinator.mark_executed(now)
        self._maybe_run_market_close(now, analysis)

    def loop(self) -> None:
        _log_data_root_status()
        while True:
            now = datetime.now(timezone.utc)
            try:
                if self.coordinator.should_run(now):
                    self.run_cycle(now)
            except Exception as exc:
                logger.exception("Hourly trading loop failure: %s", exc)
            finally:
                try:
                    base.release_model_resources()
                except Exception as cleanup_exc:
                    logger.debug("Model release failed: %s", cleanup_exc)
                time.sleep(HOURLY_LOOP_SLEEP_SECONDS)

    def _log_strategy_mix(self, analysis: Dict[str, Dict]) -> None:
        if not analysis:
            logger.warning("Hourly analysis produced no candidates")
            return
        total = len(analysis)
        maxdiff_candidates = [
            symbol for symbol, data in analysis.items() if data.get("strategy") in base.MAXDIFF_STRATEGIES
        ]
        logger.info(
            "Hourly analysis summary: total=%d maxdiff_strategies=%d",
            total,
            len(maxdiff_candidates),
        )

    def _ensure_hourly_data_ready(self) -> List[HourlyDataStatus]:
        statuses, issues = _hourly_statuses_with_refresh(self.symbols)
        if issues:
            message = "; ".join(f"{issue.symbol}:{issue.reason}" for issue in issues)
            if HOURLY_FAIL_ON_DATA_GAPS:
                raise RuntimeError(message)
            logger.warning("Filtering symbols without fresh hourly data: %s", message)
            valid_symbols = {status.symbol for status in statuses}
            if not valid_symbols:
                raise RuntimeError("All hourly symbols missing usable data")
            if valid_symbols != set(self.symbols):
                logger.info("Reduced hourly universe to %s", ", ".join(sorted(valid_symbols)))
                self.symbols = [symbol for symbol in self.symbols if symbol in valid_symbols]
        if not statuses:
            raise RuntimeError("No hourly data found for configured symbols")
        self._latest_data_statuses = statuses
        return statuses

    def _log_hourly_data_statuses(self, statuses: Sequence[HourlyDataStatus]) -> None:
        if not statuses:
            return
        logger.info(
            "Hourly data freshness check passed: %s (root=%s)",
            summarize_statuses(statuses),
            HOURLY_DATA_ROOT,
        )
        for status in statuses:
            logger.debug(
                "  %s -> %s (lag=%.2fh close=%.4f)",
                status.symbol,
                status.latest_timestamp.isoformat(),
                status.staleness_hours,
                status.latest_close,
            )

    def _maybe_run_market_close(self, now: datetime, analysis: Dict[str, Dict]) -> None:
        if not is_nyse_trading_day_now(now):
            return
        _, market_close = base.get_market_hours()
        est_now = now.astimezone(EST)
        window_start = market_close - timedelta(minutes=HOURLY_MARKET_CLOSE_WINDOW)
        if not (window_start <= est_now <= market_close):
            return
        if self.last_market_close_date == est_now.date():
            return
        logger.info("Hourly loop entering market-close backout window")
        self.previous_picks = base.manage_market_close(self.symbols, self.previous_picks, analysis)
        self.last_market_close_date = est_now.date()


def _signal_handler(signum, _frame) -> None:
    signal_name = signal.Signals(signum).name
    logger.info("Received %s; shutting down hourly trading loop.", signal_name)
    try:
        base.cleanup_spawned_processes()
    except Exception as exc:
        logger.warning("Cleanup failed: %s", exc)
    try:
        base.release_model_resources()
    except Exception as exc:
        logger.warning("Model release failed: %s", exc)
    sys.exit(0)


def main() -> None:
    engine = HourlyTradingEngine()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    logger.info(
        "trade_stock_e2e_hourly initialised with %d symbols, window=%d min, sleep=%d s",
        len(engine.symbols),
        HOURLY_ANALYSIS_WINDOW_MINUTES,
        HOURLY_LOOP_SLEEP_SECONDS,
    )
    engine.loop()


if __name__ == "__main__":
    main()
