#ifndef WORKSTEAL_SIM_H
#define WORKSTEAL_SIM_H

typedef struct {
    double dip_pct;
    double proximity_pct;
    double profit_target_pct;
    double stop_loss_pct;
    double trailing_stop_pct;
    double margin_annual_rate;
    double max_position_pct;
    int max_positions;
    int max_hold_days;
    int lookback_days;
    int sma_filter_period;
    double initial_cash;
    double max_leverage;
    double maker_fee;
    double max_drawdown_exit;
    int enable_shorts;
    double short_pump_pct;
    int reentry_cooldown_days;
    int momentum_period;
    double momentum_min;
    /* market breadth filter: skip entries if fraction of symbols w/ negative day return > threshold */
    double market_breadth_filter;
    /* risk-off trigger: cross-symbol SMA + momentum checks */
    int risk_off_sma_period;       /* 0=disabled, e.g. 30 */
    int risk_off_momentum_period;  /* 0=disabled, e.g. 7 */
    double risk_off_momentum_threshold; /* e.g. -0.05 */
    double risk_off_breadth_threshold;  /* stricter breadth when risk-off, e.g. 0.70 */
    /* ATR-based adaptive dip threshold */
    int atr_period;         /* 0=disabled, e.g. 14 */
    double atr_dip_mult;    /* ATR multiplier for adaptive dip, e.g. 2.0 */
} WorkStealConfig;

typedef struct {
    double total_return;
    double sortino;
    double sharpe;
    double max_drawdown;
    double win_rate;
    double final_equity;
    double mean_daily_return;
    int total_trades;
    int n_days;
} SimResult;

/*
 * Run work-stealing backtest on multi-symbol daily bar data.
 *
 * Data layout (all arrays length n_bars * n_symbols, symbol-major order):
 *   For symbol s, bar i: index = s * n_bars + i
 *
 * timestamps: array of n_bars doubles (unix epoch days, shared across symbols)
 * valid_mask: n_symbols * n_bars, 1 if bar exists for that symbol/day, 0 otherwise
 * opens, highs, lows, closes: n_symbols * n_bars price arrays
 * fee_rates: per-symbol fee rates, length n_symbols
 * equity_curve: caller-allocated output buffer, length n_bars
 */
void worksteal_simulate(
    const double *timestamps,
    const int *valid_mask,
    const double *opens,
    const double *highs,
    const double *lows,
    const double *closes,
    const double *fee_rates,
    int n_bars,
    int n_symbols,
    const WorkStealConfig *cfg,
    SimResult *result,
    double *equity_curve
);

void worksteal_simulate_batch(
    const double *timestamps,
    const int *valid_mask,
    const double *opens,
    const double *highs,
    const double *lows,
    const double *closes,
    const double *fee_rates,
    int n_bars,
    int n_symbols,
    const WorkStealConfig *configs,
    SimResult *results,
    int n_configs
);

#endif
