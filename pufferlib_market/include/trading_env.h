#ifndef TRADING_ENV_H
#define TRADING_ENV_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

/* ---------- constants ---------- */
#define MAX_SYMBOLS       32
#define FEATURES_PER_SYM  16
#define PRICE_FEATS       5   /* O H L C V */
#define SYM_NAME_LEN      16
#define HEADER_SIZE        64
#define INITIAL_CASH       10000.0f

/* ---------- binary-file header (matches export_data.py) ---------- */
typedef struct {
    char     magic[4];          /* "MKTD" */
    unsigned int version;
    unsigned int num_symbols;
    unsigned int num_timesteps;
    unsigned int features_per_sym;
    unsigned int price_features;
    char     padding[40];
} DataHeader;

/* ---------- pufferlib Log (all floats, 'n' last) ---------- */
typedef struct {
    float total_return;
    float sortino;
    float max_drawdown;
    float num_trades;
    float win_rate;
    float avg_hold_hours;
    float n;                    /* required last by pufferlib */
} Log;

/* ---------- shared read-only market data (one per process) ---------- */
typedef struct {
    int       num_symbols;
    int       num_timesteps;
    char      sym_names[MAX_SYMBOLS][SYM_NAME_LEN];

    /* pointers into mmapped / malloced data */
    float*    features;         /* [T][S][FEATURES_PER_SYM] */
    float*    prices;           /* [T][S][PRICE_FEATS] */
    unsigned char* tradable;    /* optional [T][S] uint8 mask (1=tradable, 0=market closed) */

    /* owned memory (free on close) */
    void*     file_buf;         /* raw file buffer */
    size_t    file_size;
} MarketData;

/* ---------- per-agent state ---------- */
typedef struct {
    float  cash;
    int    position_sym;        /* -1 = flat, 0..N-1 = long, N..2N-1 = short */
    float  position_qty;        /* shares held (positive) */
    float  entry_price;
    int    hold_hours;
    float  peak_equity;
    float  initial_equity;
    int    num_trades;
    int    winning_trades;
    float  sum_neg_sq;          /* for sortino */
    float  sum_ret;
    int    ret_count;
    int    data_offset;         /* starting row in the data (randomised) */
    int    step;                /* current step within episode */
} AgentState;

/* ---------- main environment struct (pufferlib compatible) ---------- */
typedef struct {
    /* --- pufferlib required fields (must be first) --- */
    Log            log;
    float*         observations;    /* [obs_size] */
    int*           actions;         /* [1] */
    float*         rewards;         /* [1] */
    unsigned char* terminals;       /* [1] */

    /* --- env config --- */
    int            max_steps;       /* episode length in hours */
    float          fee_rate;        /* transaction cost (e.g. 0.001) */
    float          max_leverage;    /* 1.0 = no leverage */
    float          periods_per_year;/* annualisation factor for metrics (e.g. 8760 for hourly, 365 for daily) */

    /* --- action-space config ---
       Action layout:
         0 = go flat
         1..K = long actions
         K+1..2K = short actions
       where K = num_symbols * action_allocation_bins * action_level_bins.
       Within each side, action factors as (symbol, allocation_bin, level_bin). */
    int            action_allocation_bins;  /* >=1, target exposure bins (e.g. 5 => 20/40/60/80/100%) */
    int            action_level_bins;       /* >=1, execution level bins around close */
    float          action_max_offset_bps;   /* max abs level offset in bps (e.g. 50 => +/-0.50%) */

    /* --- reward shaping config --- */
    float          reward_scale;    /* multiply return by this (default 10) */
    float          reward_clip;     /* clip abs(reward) to this (default 5) */
    float          cash_penalty;    /* per-step penalty for being flat (default 0.01) */
    float          drawdown_penalty;/* penalty scale for drawdown from peak (default 0) */
    float          downside_penalty;/* penalty scale for negative returns (ret^2) (default 0) */
    float          smooth_downside_penalty;  /* smooth downside penalty scale (softplus-ret proxy)^2 */
    float          smooth_downside_temperature; /* temperature for smooth downside penalty */
    float          trade_penalty;   /* per-trade penalty (counting opens/closes) (default 0) */

    /* --- shared data (NOT owned, do not free) --- */
    MarketData*    data;

    /* --- per-agent state --- */
    AgentState     agent;

    /* --- derived sizes --- */
    int            obs_size;
    int            num_actions;     /* 1 + 2*(num_symbols*allocation_bins*level_bins) */
} TradingEnv;

/* ---------- observation layout ----------
 *
 * For each symbol s in [0, S):
 *   obs[s*FEATURES_PER_SYM + 0..15]  = feature vector at current timestep
 *
 * Then portfolio state:
 *   obs[S*FEATURES_PER_SYM + 0]      = cash / INITIAL_CASH
 *   obs[S*FEATURES_PER_SYM + 1]      = position_value / INITIAL_CASH  (signed: neg for short)
 *   obs[S*FEATURES_PER_SYM + 2]      = unrealised_pnl / INITIAL_CASH
 *   obs[S*FEATURES_PER_SYM + 3]      = hours_in_position / max_steps
 *   obs[S*FEATURES_PER_SYM + 4]      = episode_progress
 *   obs[S*FEATURES_PER_SYM + 5..4+S] = one-hot current position (0 if flat)
 *
 * Total obs_size = S*FEATURES_PER_SYM + 5 + S
 * ---------------------------------------- */

/* ---------- core pufferlib functions ---------- */
void c_reset(TradingEnv* env);
void c_step(TradingEnv* env);
void c_close(TradingEnv* env);

/* no-op render */
static inline void c_render(TradingEnv* env) { (void)env; }

/* ---------- data loading ---------- */
MarketData* market_data_load(const char* path);
void        market_data_free(MarketData* md);

/* ---------- inline helpers ---------- */

static inline float get_feature(const MarketData* md, int t, int s, int f) {
    return md->features[(t * md->num_symbols + s) * FEATURES_PER_SYM + f];
}

static inline float get_price(const MarketData* md, int t, int s, int p) {
    return md->prices[(t * md->num_symbols + s) * PRICE_FEATS + p];
}

static inline int is_tradable(const MarketData* md, int t, int s) {
    if (md->tradable == NULL) return 1;
    return md->tradable[t * md->num_symbols + s] != 0;
}

/* price indices */
#define P_OPEN  0
#define P_HIGH  1
#define P_LOW   2
#define P_CLOSE 3
#define P_VOL   4

#endif /* TRADING_ENV_H */
