#ifndef MARKET_SIM_H
#define MARKET_SIM_H

#include <math.h>

#define MAX_SYMBOLS 64
#define ANNUALIZE_FACTOR 8760.0

typedef struct {
    double max_leverage;
    int can_short;
    double maker_fee;
    double margin_hourly_rate;
    double initial_cash;
    double fill_buffer_pct;
    double min_edge;
    int max_hold_bars;
    double intensity_scale;
} SimConfig;

typedef struct {
    double total_return;
    double sortino;
    double max_drawdown;
    double final_equity;
    int num_trades;
    double margin_cost_total;
} SimResult;

typedef struct {
    double fee_rate;
    int can_short;
    int can_long;
} SymbolConfig;

typedef struct {
    double qty;
    double entry_price;
    int bars_held;
} Position;

typedef struct {
    int n_symbols;
    double max_leverage;
    double margin_hourly_rate;
    double initial_cash;
    double fill_buffer_pct;
    double min_edge;
    int max_hold_bars;
    double intensity_scale;
    int max_positions;
    double force_close_slippage;
    SymbolConfig sym_cfgs[MAX_SYMBOLS];
} MultiSimConfig;

typedef struct {
    double total_return;
    double sortino;
    double max_drawdown;
    double final_equity;
    int num_trades;
    double margin_cost_total;
    int trades_per_symbol[MAX_SYMBOLS];
} MultiSimResult;

/* single-symbol sim (ported from csim/market_sim.c) */
void simulate(
    const double *open, const double *high, const double *low, const double *close,
    const double *buy_price, const double *sell_price,
    const double *buy_amount, const double *sell_amount,
    int n_bars,
    const SimConfig *cfg,
    SimResult *result,
    double *equity_curve
);

void simulate_batch(
    const double *open, const double *high, const double *low, const double *close,
    const double *buy_price, const double *sell_price,
    const double *buy_amount, const double *sell_amount,
    int n_bars,
    const SimConfig *configs,
    SimResult *results,
    int n_configs
);

/* multi-symbol portfolio sim */
void simulate_multi(
    int n_bars,
    int n_symbols,
    const double *close,       /* n_bars * n_symbols, row-major */
    const double *high,        /* n_bars * n_symbols */
    const double *low,         /* n_bars * n_symbols */
    const double *buy_price,   /* n_bars * n_symbols */
    const double *sell_price,  /* n_bars * n_symbols */
    const double *buy_amount,  /* n_bars * n_symbols */
    const double *sell_amount, /* n_bars * n_symbols */
    const MultiSimConfig *cfg,
    MultiSimResult *result,
    double *equity_curve       /* caller-allocated, n_bars+1 */
);

/* metric helpers */
double compute_sortino(const double *equity_curve, int n_eq);
double compute_max_drawdown(const double *equity_curve, int n_eq);

#endif
