"""
Parallel version of backtest_forecasts using ThreadPoolExecutor.
Threads share GPU models and avoid pickling issues.
"""

from concurrent.futures import ThreadPoolExecutor
import functools
from backtest_test3_inline import *

# Override backtest_forecasts with parallel version
_original_backtest_forecasts = backtest_forecasts


def backtest_forecasts_parallel(symbol, num_simulations=50, max_workers=4, *, model_override=None):
    """Parallel version using threads (shares GPU models)"""
    current_time_formatted = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    if __name__ == "__main__":
        current_time_formatted = "2024-09-07--03-36-27"

    stock_data = download_daily_stock_data(current_time_formatted, symbols=[symbol])

    if stock_data.empty:
        logger.error(f"No data available for {symbol}")
        return pd.DataFrame()

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / current_time_formatted

    global SPREAD
    spread = fetch_spread(symbol)
    logger.info(f"spread: {spread}")
    previous_spread = SPREAD
    SPREAD = spread

    try:
        if len(stock_data) < num_simulations:
            logger.warning(
                f"Not enough historical data for {num_simulations} simulations. Using {len(stock_data)} instead."
            )
            num_simulations = len(stock_data)

        from loss_utils import CRYPTO_TRADING_FEE, TRADING_FEE
        is_crypto = symbol in crypto_symbols
        trading_fee = CRYPTO_TRADING_FEE if is_crypto else TRADING_FEE

        # Prepare arguments for each simulation
        sim_args = []
        for sim_number in range(num_simulations):
            simulation_data = stock_data.iloc[: -(sim_number + 1)].copy(deep=True)
            if simulation_data.empty:
                logger.warning(f"No data left for simulation {sim_number + 1}")
                continue
            sim_args.append(
                (
                    simulation_data,
                    symbol,
                    trading_fee,
                    is_crypto,
                    sim_number,
                    spread,
                    model_override,
                )
            )

        logger.info(f"Running {len(sim_args)} simulations in parallel with {max_workers} workers...")

        # Run simulations in parallel using threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def _run(args):
                simulation_data, sym, fee, crypto_flag, idx, sim_spread, override = args
                return run_single_simulation(
                    simulation_data,
                    sym,
                    fee,
                    crypto_flag,
                    idx,
                    sim_spread,
                    model_override=override,
                )

            results = list(executor.map(_run, sim_args))

        results_df = pd.DataFrame(results)
        walk_forward_stats = compute_walk_forward_stats(results_df)
        for key, value in walk_forward_stats.items():
            results_df[key] = value

        # Forecast next day's PnL for each strategy using Toto
        strategy_pnl_columns = [
            ("simple_strategy_avg_daily_return", "simple_forecasted_pnl"),
            ("all_signals_strategy_avg_daily_return", "all_signals_forecasted_pnl"),
            ("buy_hold_avg_daily_return", "buy_hold_forecasted_pnl"),
            ("unprofit_shutdown_avg_daily_return", "unprofit_shutdown_forecasted_pnl"),
            ("entry_takeprofit_avg_daily_return", "entry_takeprofit_forecasted_pnl"),
            ("highlow_avg_daily_return", "highlow_forecasted_pnl"),
            ("maxdiff_avg_daily_return", "maxdiff_forecasted_pnl"),
            ("maxdiffalwayson_avg_daily_return", "maxdiffalwayson_forecasted_pnl"),
        ]

        for pnl_col, forecast_col in strategy_pnl_columns:
            if pnl_col in results_df.columns:
                pnl_series = results_df[pnl_col].dropna()
                if len(pnl_series) > 0:
                    forecasted = _forecast_pnl_with_toto(pnl_series, f"{symbol}_{pnl_col}")
                    results_df[forecast_col] = forecasted
                    logger.info(f"{symbol} {forecast_col}: {forecasted:.6f}")
                else:
                    results_df[forecast_col] = 0.0
            else:
                results_df[forecast_col] = 0.0

        # Log metrics (same as original)
        tb_writer.add_scalar(
            f"{symbol}/final_metrics/simple_avg_return",
            results_df["simple_strategy_avg_daily_return"].mean(),
            0,
        )
        # ... (rest of logging code)

        _log_validation_losses(results_df)
        _log_strategy_summary(results_df, symbol, num_simulations)

        # Determine best strategy
        avg_simple = results_df["simple_strategy_return"].mean()
        avg_allsignals = results_df["all_signals_strategy_return"].mean()
        avg_takeprofit = results_df["entry_takeprofit_return"].mean()
        avg_highlow = results_df["highlow_return"].mean()
        if "maxdiff_return" in results_df:
            avg_maxdiff = float(results_df["maxdiff_return"].mean())
            if not np.isfinite(avg_maxdiff):
                avg_maxdiff = float("-inf")
        else:
            avg_maxdiff = float("-inf")

        best_return = max(avg_simple, avg_allsignals, avg_takeprofit, avg_highlow, avg_maxdiff)
        if best_return == avg_highlow:
            best_strategy = "highlow"
        elif best_return == avg_takeprofit:
            best_strategy = "takeprofit"
        elif best_return == avg_maxdiff:
            best_strategy = "maxdiff"
        elif best_return == avg_allsignals:
            best_strategy = "all_signals"
        else:
            best_strategy = "simple"

        set_strategy_for_symbol(symbol, best_strategy)
        evaluate_and_save_close_policy(symbol, num_comparisons=10)

        return results_df
    finally:
        SPREAD = previous_spread


# Override the function
backtest_forecasts = backtest_forecasts_parallel


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSD"
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    print(f"Testing parallel backtest: {symbol}, {num_sims} sims, {workers} workers")
    import time
    start = time.time()
    results = backtest_forecasts_parallel(symbol, num_sims, max_workers=workers)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s ({elapsed/num_sims:.3f}s per sim)")
    print(f"Results shape: {results.shape}")
