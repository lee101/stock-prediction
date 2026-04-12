from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pandas as pd

from backtest_test3_inline import backtest_forecasts, release_model_resources
import marketsimulator.backtest_test3_inline as fallback_backtest_module

from src.trading_server.client import TradingServerClient

from .gemini_overlay import refine_candidate_days_with_gemini, refine_daily_candidates_with_gemini
from .live import WideLiveExecutionConfig, submit_live_entry_orders
from .planner import WidePlannerConfig, build_daily_candidates, build_wide_plan, render_plan_text
from .selection import (
    WideSelectionConfig,
    build_symbol_rl_prior,
    rank_candidates,
    rerank_candidate_days,
    resolve_torch_device,
)
from .intraday import load_hourly_histories, simulate_intraday_day
from .replay import simulate_wide_strategy
from .runtime_logging import WideRunLogger


def _configure_third_party_logging() -> None:
    level = os.getenv("TRADE_STOCK_WIDE_LOGURU_LEVEL", "ERROR").strip().upper()
    try:
        from loguru import logger as loguru_logger
    except Exception:
        return
    try:
        loguru_logger.remove()
        loguru_logger.add(lambda _msg: None, level=level)
    except Exception:
        return


def discover_symbols(data_root: Path, *, limit: int | None = None) -> list[str]:
    symbols = sorted(path.stem.upper() for path in data_root.glob("*.csv"))
    if limit is not None:
        return symbols[: max(int(limit), 0)]
    return symbols


def _load_price_history(symbol: str, data_root: Path) -> pd.DataFrame | None:
    path = data_root / f"{symbol.upper()}.csv"
    if not path.exists():
        return None
    history = pd.read_csv(path)
    rename_map = {column: str(column).strip().lower() for column in history.columns}
    history = history.rename(columns=rename_map)
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(history.columns):
        return None
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    history = history.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return history[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)


def _attach_realized_ohlc(frame: pd.DataFrame, symbol: str, data_root: Path) -> pd.DataFrame:
    history = _load_price_history(symbol, data_root)
    if history is None or history.empty:
        return frame.reset_index(drop=True)

    enriched = frame.copy().reset_index(drop=True)
    if "timestamp" in enriched.columns:
        enriched["_ts"] = pd.to_datetime(enriched["timestamp"], utc=True, errors="coerce")
        history_exact = history.rename(columns={"timestamp": "_ts"})
        merged = enriched.merge(history_exact, on="_ts", how="left", suffixes=("", "_realized"))
        realized_match_count = int(merged[[f"{column}_realized" for column in ("open", "high", "low", "close") if f"{column}_realized" in merged.columns]].notna().any(axis=1).sum()) if any(f"{column}_realized" in merged.columns for column in ("open", "high", "low", "close")) else 0
        if realized_match_count == 0:
            history_daily = history.copy()
            history_daily["_session_day"] = history_daily["timestamp"].dt.floor("D")
            enriched_daily = enriched.drop(columns=["_ts"], errors="ignore").copy()
            enriched_daily["_session_day"] = pd.to_datetime(enriched["timestamp"], utc=True, errors="coerce").dt.floor("D")
            merged = enriched_daily.merge(
                history_daily.rename(
                    columns={
                        "timestamp": "timestamp_realized",
                        "open": "open_realized",
                        "high": "high_realized",
                        "low": "low_realized",
                        "close": "close_realized",
                        "symbol": "symbol_realized",
                        "volume": "volume_realized",
                        "trade_count": "trade_count_realized",
                        "vwap": "vwap_realized",
                    }
                ),
                on="_session_day",
                how="left",
            )
        for column in ("open", "high", "low", "close"):
            realized_col = f"{column}_realized"
            if realized_col not in merged.columns:
                continue
            if column not in merged.columns:
                merged[column] = merged[realized_col]
            else:
                merged[column] = merged[column].fillna(merged[realized_col])
        needs_tail_alignment = all(
            column not in merged.columns or merged[column].notna().sum() == 0
            for column in ("open", "high", "low")
        )
        if needs_tail_alignment:
            aligned = history.tail(len(merged)).iloc[::-1].reset_index(drop=True)
            for column in ("open", "high", "low", "close"):
                if column not in merged.columns:
                    merged[column] = aligned[column]
                else:
                    merged[column] = merged[column].fillna(aligned[column])
        return merged.drop(
            columns=[
                "_ts",
                "_session_day",
                "timestamp_realized",
                "symbol_realized",
                "volume_realized",
                "trade_count_realized",
                "vwap_realized",
                "open_realized",
                "high_realized",
                "low_realized",
                "close_realized",
            ],
            errors="ignore",
        )

    aligned = history.tail(len(enriched)).iloc[::-1].reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        if column not in enriched.columns:
            enriched[column] = aligned[column]
        else:
            enriched[column] = enriched[column].fillna(aligned[column])
    return enriched


def _compute_backtest_frame(
    symbol: str,
    *,
    num_simulations: int | None = None,
    model_override: str | None = None,
) -> pd.DataFrame:
    use_primary = os.getenv("TRADE_STOCK_WIDE_USE_PRIMARY_BACKTEST", "").strip().lower() in {"1", "true", "yes", "on"}
    if not use_primary:
        return fallback_backtest_module._fallback_backtest(symbol, num_simulations=num_simulations)
    try:
        return backtest_forecasts(symbol, num_simulations=num_simulations, model_override=model_override)
    except Exception:
        return fallback_backtest_module._fallback_backtest(symbol, num_simulations=num_simulations)


def load_backtests(
    symbols: Sequence[str],
    *,
    data_root: Path,
    model_override: str | None = None,
    num_simulations: int | None = None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = _compute_backtest_frame(
            symbol,
            num_simulations=num_simulations,
            model_override=model_override,
        )
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames[symbol] = _attach_realized_ohlc(frame, symbol, data_root)
    return frames


def main(argv: Sequence[str] | None = None) -> int:
    _configure_third_party_logging()
    ap = argparse.ArgumentParser(description="Wide stock planner and 30-day replay harness")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbol list. Defaults to trainingdata/*.csv")
    ap.add_argument("--data-root", default="trainingdata", help="Daily training-data root for symbol discovery")
    ap.add_argument("--limit-symbols", type=int, default=None, help="Optional discovery cap for batch runs")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--account-equity", type=float, default=100000.0)
    ap.add_argument("--pair-notional-fraction", type=float, default=0.50)
    ap.add_argument("--max-pair-notional-fraction", type=float, default=0.50)
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--backtest-days", type=int, default=30)
    ap.add_argument("--num-simulations", type=int, default=20)
    ap.add_argument("--model-override", default="chronos2")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--fill-buffer-bps", type=float, default=5.0)
    ap.add_argument("--watch-activation-pct", type=float, default=0.005, help="Wake an order watcher once price gets within this fraction of entry")
    ap.add_argument("--steal-protection-pct", type=float, default=0.004, help="Do not cancel an active watcher once it is this close to entry")
    ap.add_argument("--selection-objective", choices=("pnl", "sortino", "hybrid", "tiny_net", "torch_mlp"), default="hybrid")
    ap.add_argument("--selection-lookback-days", type=int, default=20)
    ap.add_argument("--tiny-net-hidden-dim", type=int, default=8)
    ap.add_argument("--tiny-net-epochs", type=int, default=120)
    ap.add_argument("--tiny-net-learning-rate", type=float, default=0.03)
    ap.add_argument("--tiny-net-l2", type=float, default=1e-4)
    ap.add_argument("--tiny-net-augment-copies", type=int, default=3)
    ap.add_argument("--tiny-net-noise-scale", type=float, default=0.04)
    ap.add_argument("--tiny-net-min-train-samples", type=int, default=12)
    ap.add_argument("--selection-seed", type=int, default=1337)
    ap.add_argument("--selection-torch-device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--selection-torch-batch-size", type=int, default=256)
    ap.add_argument("--rl-prior-leaderboard", default=None, help="Optional CSV from scripts/run_autoresearch_stock_group_sweep.py")
    ap.add_argument("--rl-prior-weight", type=float, default=0.0)
    ap.add_argument("--rl-prior-scale", type=float, default=2.0)
    ap.add_argument("--hourly-root", default="trainingdatahourly", help="Hourly bar root for realistic intraday replay")
    ap.add_argument("--daily-only-replay", action="store_true", help="Keep using the simplified daily replay instead of hourly sessions")
    ap.add_argument("--submit-plan", action="store_true", help="Submit the current top-k buy plan through the trading server")
    ap.add_argument("--trading-server-base-url", default=None)
    ap.add_argument("--trading-account", default="test-paper")
    ap.add_argument("--trading-bot-id", default="trade_stock_wide")
    ap.add_argument("--trading-execution-mode", choices=("paper", "live"), default="paper")
    ap.add_argument("--writer-ttl-seconds", type=int, default=None)
    ap.add_argument("--skip-writer-heartbeat", action="store_true")
    ap.add_argument("--min-order-notional", type=float, default=1.0)
    ap.add_argument("--gemini-overlay-mode", choices=("off", "compare", "force"), default="off")
    ap.add_argument("--gemini-model", default="gemini-3.1-flash-lite-preview")
    ap.add_argument("--gemini-min-confidence", type=float, default=0.35)
    ap.add_argument("--gemini-cache-dir", default="trade_stock_wide/gemini_cache")
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_root = Path(args.data_root)
    if args.symbols:
        symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    else:
        symbols = discover_symbols(data_root, limit=args.limit_symbols)
    if not symbols:
        raise SystemExit("No symbols resolved for trade_stock_wide")

    planner = WidePlannerConfig(
        top_k=args.top_k,
        pair_notional_fraction=args.pair_notional_fraction,
        max_pair_notional_fraction=args.max_pair_notional_fraction,
        max_leverage=args.max_leverage,
        fee_bps=args.fee_bps,
        fill_buffer_bps=args.fill_buffer_bps,
        watch_activation_pct=args.watch_activation_pct,
        steal_protection_pct=args.steal_protection_pct,
    )

    frames = load_backtests(
        symbols,
        data_root=data_root,
        model_override=args.model_override,
        num_simulations=args.num_simulations,
    )
    if not frames:
        raise SystemExit("No backtest frames were produced")

    run_logger = WideRunLogger.create()
    rl_prior_by_symbol: dict[str, float] = {}
    if args.rl_prior_leaderboard:
        rl_prior_rows = pd.read_csv(args.rl_prior_leaderboard)
        rl_prior_by_symbol = build_symbol_rl_prior(
            rl_prior_rows,
            score_column="robust_score",
            symbols_column="symbols",
            aggregation="max",
            min_score=0.0,
            scale=args.rl_prior_scale,
        )
    selection = WideSelectionConfig(
        objective=args.selection_objective,
        lookback_days=args.selection_lookback_days,
        tiny_net_hidden_dim=args.tiny_net_hidden_dim,
        tiny_net_epochs=args.tiny_net_epochs,
        tiny_net_learning_rate=args.tiny_net_learning_rate,
        tiny_net_l2=args.tiny_net_l2,
        tiny_net_augment_copies=args.tiny_net_augment_copies,
        tiny_net_noise_scale=args.tiny_net_noise_scale,
        tiny_net_min_train_samples=args.tiny_net_min_train_samples,
        seed=args.selection_seed,
        torch_device=args.selection_torch_device,
        torch_batch_size=args.selection_torch_batch_size,
        rl_prior_weight=args.rl_prior_weight,
        rl_prior_scale=args.rl_prior_scale,
        rl_prior_by_symbol=rl_prior_by_symbol,
    )
    selection_device = resolve_torch_device(selection.torch_device)

    raw_today_candidates = build_daily_candidates(frames, day_index=0, allow_short=planner.allow_short)
    raw_candidate_days: list[list] = []
    max_days = min(args.backtest_days, min(len(frame) for frame in frames.values()))
    for offset in reversed(range(max_days)):
        raw_candidate_days.append(
            build_daily_candidates(
                frames,
                day_index=offset,
                allow_short=planner.allow_short,
                require_realized_ohlc=True,
            )
        )
    candidate_days = rerank_candidate_days(
        raw_candidate_days,
        config=selection,
        fee_bps=args.fee_bps,
        fill_buffer_bps=args.fill_buffer_bps,
    )
    history_days_for_today = raw_candidate_days[:-1] if raw_candidate_days else ()
    today_candidates = rank_candidates(
        raw_today_candidates,
        history_days=history_days_for_today,
        config=selection,
        fee_bps=args.fee_bps,
        fill_buffer_bps=args.fill_buffer_bps,
    )
    today_orders = build_wide_plan(today_candidates, account_equity=args.account_equity, config=planner)
    selection_note = (
        f"selection objective={selection.objective} lookback={selection.lookback_days}d "
        f"tiny_net_hidden={selection.tiny_net_hidden_dim} tiny_net_epochs={selection.tiny_net_epochs} "
        f"torch_device={selection_device} torch_batch_size={selection.torch_batch_size} "
        f"rl_prior_symbols={len(rl_prior_by_symbol)} rl_prior_weight={selection.rl_prior_weight:.3f}"
    )
    print(selection_note)
    run_logger.event(selection_note, level=logging.INFO)

    selected_today_candidates = today_candidates
    selected_candidate_days = candidate_days
    selected_overlay_label = "base"
    gemini_cache_dir = Path(args.gemini_cache_dir)
    if args.gemini_overlay_mode != "off":
        gemini_today_candidates, _ = refine_daily_candidates_with_gemini(
            today_candidates,
            account_equity=args.account_equity,
            planner=planner,
            backtests_by_symbol=frames,
            cache_root=gemini_cache_dir,
            model=args.gemini_model,
            min_confidence=args.gemini_min_confidence,
        )
        gemini_candidate_days, overlay_stats = refine_candidate_days_with_gemini(
            candidate_days,
            starting_equity=args.account_equity,
            planner=planner,
            backtests_by_symbol=frames,
            cache_root=gemini_cache_dir,
            model=args.gemini_model,
            min_confidence=args.gemini_min_confidence,
        )
        overlay_note = (
            f"Gemini overlay model={args.gemini_model} prompts={overlay_stats.prompt_count} "
            f"cache_hits={overlay_stats.cache_hits} adjusted={overlay_stats.adjusted_count} "
            f"skipped={overlay_stats.skipped_count} invalid={overlay_stats.invalid_count}"
        )
        print(overlay_note)
        run_logger.event(overlay_note, level=logging.INFO)

        if args.gemini_overlay_mode == "force":
            selected_today_candidates = gemini_today_candidates
            selected_candidate_days = gemini_candidate_days
            selected_overlay_label = "gemini_force"
        else:
            selected_candidate_days = gemini_candidate_days
            selected_overlay_label = "gemini_compare_pending"
        today_orders = build_wide_plan(selected_today_candidates, account_equity=args.account_equity, config=planner)

    def _simulate_days(days):
        if args.daily_only_replay:
            return simulate_wide_strategy(days, starting_equity=args.account_equity, config=planner)

        hourly_by_symbol = load_hourly_histories(frames.keys(), Path(args.hourly_root))
        equity = args.account_equity
        peak = equity
        max_drawdown = 0.0
        day_results = []
        trade_count = 0
        filled_count = 0
        for day_index, day_candidates in enumerate(days):
            result = simulate_intraday_day(
                day_candidates,
                account_equity=equity,
                hourly_by_symbol=hourly_by_symbol,
                config=planner,
                day_index=day_index,
                logger=run_logger,
            )
            day_results.append(result)
            equity = result.end_equity
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = min(max_drawdown, (equity / peak) - 1.0)
            trade_count += len(result.fills)
            filled_count += sum(1 for fill in result.fills if fill.filled)
        summary = SimpleNamespace(
            total_pnl=equity - args.account_equity,
            total_return=(equity / args.account_equity) - 1.0 if args.account_equity > 0 else 0.0,
            monthly_return=0.0,
            max_drawdown=max_drawdown,
            filled_count=filled_count,
            trade_count=trade_count,
            day_results=tuple(day_results),
        )
        if day_results:
            from .replay import _monthly_from_total  # local reuse
            summary.monthly_return = _monthly_from_total(summary.total_return, len(day_results))
        return summary

    base_summary = _simulate_days(candidate_days)
    summary = base_summary
    if args.gemini_overlay_mode == "compare":
        gemini_summary = _simulate_days(selected_candidate_days)
        compare_note = (
            f"overlay compare base_monthly={base_summary.monthly_return * 100:+.2f}% "
            f"gemini_monthly={gemini_summary.monthly_return * 100:+.2f}%"
        )
        print(compare_note)
        run_logger.event(compare_note, level=logging.INFO)
        if gemini_summary.monthly_return > base_summary.monthly_return:
            summary = gemini_summary
            selected_today_candidates = refine_daily_candidates_with_gemini(
                today_candidates,
                account_equity=args.account_equity,
                planner=planner,
                backtests_by_symbol=frames,
                cache_root=gemini_cache_dir,
                model=args.gemini_model,
                min_confidence=args.gemini_min_confidence,
            )[0]
            today_orders = build_wide_plan(selected_today_candidates, account_equity=args.account_equity, config=planner)
            selected_overlay_label = "gemini_compare_win"
        else:
            selected_overlay_label = "base_compare_win"
    elif args.gemini_overlay_mode == "force":
        summary = _simulate_days(selected_candidate_days)
    else:
        selected_overlay_label = "base"

    today_orders = build_wide_plan(selected_today_candidates, account_equity=args.account_equity, config=planner)
    plan_text = render_plan_text(today_orders, account_equity=args.account_equity, config=planner)
    print(plan_text)
    run_logger.event(plan_text)
    if args.submit_plan:
        live_config = WideLiveExecutionConfig(
            execution_mode=args.trading_execution_mode,
            writer_ttl_seconds=args.writer_ttl_seconds,
            require_writer_heartbeat=not args.skip_writer_heartbeat,
            min_order_notional=args.min_order_notional,
        )
        with TradingServerClient(
            base_url=args.trading_server_base_url,
            account=args.trading_account,
            bot_id=args.trading_bot_id,
            execution_mode=args.trading_execution_mode,
        ) as trading_client:
            submitted = submit_live_entry_orders(
                today_orders,
                client=trading_client,
                config=live_config,
            )
        submit_summary = (
            f"submitted {len(submitted)} entry orders via trading_server "
            f"account={args.trading_account} mode={args.trading_execution_mode}"
        )
        print(submit_summary)
        run_logger.event(submit_summary, level=logging.INFO)
    print("")
    print(
        f"30-day replay [{selected_overlay_label}] "
        f"pnl=${summary.total_pnl:,.2f} return={summary.total_return * 100:+.2f}% "
        f"monthly={summary.monthly_return * 100:+.2f}% max_dd={summary.max_drawdown * 100:.2f}% "
        f"filled={summary.filled_count}/{summary.trade_count}"
    )
    run_logger.event(
        (
            f"30-day replay [{selected_overlay_label}] "
            f"pnl=${summary.total_pnl:,.2f} return={summary.total_return * 100:+.2f}% "
            f"monthly={summary.monthly_return * 100:+.2f}% max_dd={summary.max_drawdown * 100:.2f}% "
            f"filled={summary.filled_count}/{summary.trade_count}"
        ),
        level=logging.INFO,
    )
    print(f"log_dir={run_logger.run_dir}")

    release_model_resources(force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
