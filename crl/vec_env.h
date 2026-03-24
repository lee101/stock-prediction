#ifndef CRL_VEC_ENV_H
#define CRL_VEC_ENV_H

#include "trading_env.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_envs;
    MarketData* data;
    TradingEnv* envs;

    float* obs_buf;       /* [num_envs * obs_size] contiguous */
    int*   act_buf;       /* [num_envs] */
    float* rew_buf;       /* [num_envs] */
    unsigned char* done_buf; /* [num_envs] */

    int obs_size;
    int num_actions;
    int pinned;           /* 1 if CUDA pinned memory */
} VecEnv;

typedef struct {
    int max_steps;
    float fee_rate;
    float max_leverage;
    float short_borrow_apr;
    float periods_per_year;
    int max_hold_hours;
    int action_allocation_bins;
    int action_level_bins;
    float action_max_offset_bps;
    float reward_scale;
    float reward_clip;
    float cash_penalty;
    float drawdown_penalty;
    float downside_penalty;
    float smooth_downside_penalty;
    float smooth_downside_temperature;
    float trade_penalty;
    float smoothness_penalty;
    float fill_slippage_bps;
    float fill_probability;
    int enable_drawdown_profit_early_exit;
    int drawdown_profit_early_exit_min_steps;
    float drawdown_profit_early_exit_progress_fraction;
} VecEnvConfig;

VecEnv* vec_env_create(const char* data_path, int num_envs, const VecEnvConfig* cfg);
void vec_env_reset(VecEnv* ve, unsigned int seed);
void vec_env_step(VecEnv* ve);
void vec_env_set_offset(VecEnv* ve, int env_idx, int offset);
void vec_env_free(VecEnv* ve);

/* Get episode log for env i, zero if no completed episodes */
void vec_env_get_log(const VecEnv* ve, int env_idx, Log* out);
void vec_env_clear_logs(VecEnv* ve);

#ifdef __cplusplus
}
#endif

#endif
