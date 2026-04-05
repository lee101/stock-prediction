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

typedef struct {
    double initial_cash;
    double max_gross_leverage;
    double fee_rate;
    double borrow_rate_per_period;
    double periods_per_year;
    int can_short;
} WeightSimConfig;

typedef struct {
    double total_return;
    double annualized_return;
    double sortino;
    double max_drawdown;
    double final_equity;
    double total_turnover;
    double total_fees;
    double total_borrow_cost;
} WeightSimResult;

typedef struct {
    int lookback;
    int episode_steps;
    double reward_scale;
} WeightEnvConfig;

typedef struct {
    const double *close;  /* n_bars * n_symbols, row-major */
    int n_bars;
    int n_symbols;
    WeightEnvConfig env_cfg;
    WeightSimConfig sim_cfg;
    int start_index;
    int t;
    int steps;
    double equity;
    double peak_equity;
    double recent_return;
    double total_turnover;
    double total_fees;
    double total_borrow_cost;
    double current_weights[MAX_SYMBOLS];
    double *equity_curve; /* episode_steps + 1 */
    double *returns;      /* episode_steps */
} WeightEnv;

typedef struct {
    double reward;
    double turnover;
    double fees;
    double borrow_cost;
    double equity;
    double period_return;
    int done;
    WeightSimResult summary;
} WeightEnvStepInfo;

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
double compute_annualized_return(double total_return, int n_periods, double periods_per_year);

/* continuous target-weight portfolio simulator */
void simulate_target_weights(
    int n_bars,
    int n_symbols,
    const double *close,          /* n_bars * n_symbols, row-major */
    const double *target_weights, /* n_bars * n_symbols, row-major; row t applied over t->t+1 */
    const WeightSimConfig *cfg,
    WeightSimResult *result,
    double *equity_curve          /* caller-allocated, n_bars */
);

int weight_env_obs_dim(const WeightEnv *env);
int weight_env_init(
    WeightEnv *env,
    const double *close,
    int n_bars,
    int n_symbols,
    const WeightEnvConfig *env_cfg,
    const WeightSimConfig *sim_cfg
);
void weight_env_free(WeightEnv *env);
int weight_env_reset(WeightEnv *env, int start_index);
int weight_env_get_obs(const WeightEnv *env, double *out_obs, int obs_len);
int weight_env_step(
    WeightEnv *env,
    const double *raw_scores,
    int raw_len,
    WeightEnvStepInfo *out_info
);

#endif
