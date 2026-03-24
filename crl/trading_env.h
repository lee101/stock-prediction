#ifndef TRADING_ENV_H
#define TRADING_ENV_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#ifndef LIKELY
#  define LIKELY(x)   __builtin_expect(!!(x), 1)
#  define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#define MAX_SYMBOLS       64
#define FEATURES_PER_SYM  16
#define PRICE_FEATS       5
#define SYM_NAME_LEN      16
#define HEADER_SIZE        64
#define INITIAL_CASH       10000.0f

typedef struct {
    char     magic[4];
    unsigned int version;
    unsigned int num_symbols;
    unsigned int num_timesteps;
    unsigned int features_per_sym;
    unsigned int price_features;
    char     padding[40];
} DataHeader;

typedef struct {
    float total_return;
    float sortino;
    float max_drawdown;
    float num_trades;
    float win_rate;
    float avg_hold_hours;
    float n;
} Log;

typedef struct {
    int       num_symbols;
    int       num_timesteps;
    int       features_per_sym;
    char      sym_names[MAX_SYMBOLS][SYM_NAME_LEN];
    float*    features;
    float*    prices;
    unsigned char* tradable;
    void*     file_buf;
    size_t    file_size;
} MarketData;

typedef struct {
    float  cash;
    int    position_sym;
    float  position_qty;
    float  entry_price;
    int    hold_hours;
    float  peak_equity;
    float  initial_equity;
    int    num_trades;
    int    winning_trades;
    float  sum_neg_sq;
    float  sum_ret;
    int    ret_count;
    float  prev_ret;
    float  max_drawdown;
    int    data_offset;
    int    step;
} AgentState;

typedef struct {
    Log            log;
    float*         observations;
    int*           actions;
    float*         rewards;
    unsigned char* terminals;

    int            max_steps;
    float          fee_rate;
    float          max_leverage;
    float          short_borrow_apr;
    float          periods_per_year;
    int            max_hold_hours;
    int            enable_drawdown_profit_early_exit;
    int            drawdown_profit_early_exit_verbose;
    int            drawdown_profit_early_exit_min_steps;
    float          drawdown_profit_early_exit_progress_fraction;

    int            action_allocation_bins;
    int            action_level_bins;
    float          action_max_offset_bps;

    float          reward_scale;
    float          reward_clip;
    float          cash_penalty;
    float          drawdown_penalty;
    float          downside_penalty;
    float          smooth_downside_penalty;
    float          smooth_downside_temperature;
    float          trade_penalty;
    float          smoothness_penalty;
    float          fill_slippage_bps;
    float          fill_probability;

    int            forced_offset;
    MarketData*    data;
    AgentState     agent;
    int            obs_size;
    int            num_actions;
} TradingEnv;

void c_reset(TradingEnv* env);
__attribute__((hot)) void c_step(TradingEnv* env);
void c_close(TradingEnv* env);
static inline void c_render(TradingEnv* env) { (void)env; }

MarketData* market_data_load(const char* path);
void        market_data_free(MarketData* md);

static inline float get_feature(const MarketData* md, int t, int s, int f) {
    return md->features[(t * md->num_symbols + s) * md->features_per_sym + f];
}
static inline float get_price(const MarketData* md, int t, int s, int p) {
    return md->prices[(t * md->num_symbols + s) * PRICE_FEATS + p];
}
static inline int is_tradable(const MarketData* md, int t, int s) {
    if (md->tradable == NULL) return 1;
    return md->tradable[t * md->num_symbols + s] != 0;
}

#define P_OPEN  0
#define P_HIGH  1
#define P_LOW   2
#define P_CLOSE 3
#define P_VOL   4

#endif
