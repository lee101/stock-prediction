#include "vec_env.h"
#include "config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

VecEnv* vec_env_create(const char* data_path, int num_envs, const VecEnvConfig* cfg) {
    MarketData* md = market_data_load(data_path);
    if (!md) return NULL;

    VecEnv* ve = (VecEnv*)calloc(1, sizeof(VecEnv));
    ve->num_envs = num_envs;
    ve->data = md;

    int S = md->num_symbols;
    int F = md->features_per_sym;
    ve->obs_size = S * F + 5 + S;
    int alloc_bins = cfg->action_allocation_bins > 0 ? cfg->action_allocation_bins : CRL_DEFAULT_ALLOC_BINS;
    int level_bins = cfg->action_level_bins > 0 ? cfg->action_level_bins : CRL_DEFAULT_LEVEL_BINS;
    ve->num_actions = 1 + 2 * S * alloc_bins * level_bins;

    ve->obs_buf = (float*)calloc(num_envs * ve->obs_size, sizeof(float));
    ve->act_buf = (int*)calloc(num_envs, sizeof(int));
    ve->rew_buf = (float*)calloc(num_envs, sizeof(float));
    ve->done_buf = (unsigned char*)calloc(num_envs, sizeof(unsigned char));
    ve->pinned = 0;

    ve->envs = (TradingEnv*)calloc(num_envs, sizeof(TradingEnv));
    for (int i = 0; i < num_envs; i++) {
        TradingEnv* e = &ve->envs[i];
        e->data = md;
        e->observations = &ve->obs_buf[i * ve->obs_size];
        e->actions = &ve->act_buf[i];
        e->rewards = &ve->rew_buf[i];
        e->terminals = &ve->done_buf[i];

        e->max_steps = cfg->max_steps > 0 ? cfg->max_steps : CRL_DEFAULT_MAX_STEPS;
        e->fee_rate = cfg->fee_rate > 0 ? cfg->fee_rate : CRL_DEFAULT_FEE_RATE;
        e->max_leverage = cfg->max_leverage > 0 ? cfg->max_leverage : 1.0f;
        e->short_borrow_apr = cfg->short_borrow_apr;
        e->periods_per_year = cfg->periods_per_year > 0 ? cfg->periods_per_year : CRL_DEFAULT_PERIODS_PER_YEAR;
        e->max_hold_hours = cfg->max_hold_hours;
        e->action_allocation_bins = alloc_bins;
        e->action_level_bins = level_bins;
        e->action_max_offset_bps = cfg->action_max_offset_bps > 0 ? cfg->action_max_offset_bps : CRL_DEFAULT_MAX_OFFSET_BPS;
        e->reward_scale = cfg->reward_scale > 0 ? cfg->reward_scale : CRL_DEFAULT_REWARD_SCALE;
        e->reward_clip = cfg->reward_clip > 0 ? cfg->reward_clip : CRL_DEFAULT_REWARD_CLIP;
        e->cash_penalty = cfg->cash_penalty;
        e->drawdown_penalty = cfg->drawdown_penalty;
        e->downside_penalty = cfg->downside_penalty;
        e->smooth_downside_penalty = cfg->smooth_downside_penalty;
        e->smooth_downside_temperature = cfg->smooth_downside_temperature > 0 ? cfg->smooth_downside_temperature : 0.02f;
        e->trade_penalty = cfg->trade_penalty;
        e->smoothness_penalty = cfg->smoothness_penalty;
        e->fill_slippage_bps = cfg->fill_slippage_bps;
        e->fill_probability = cfg->fill_probability > 0 ? cfg->fill_probability : 1.0f;
        e->enable_drawdown_profit_early_exit = cfg->enable_drawdown_profit_early_exit;
        e->drawdown_profit_early_exit_min_steps = cfg->drawdown_profit_early_exit_min_steps;
        e->drawdown_profit_early_exit_progress_fraction = cfg->drawdown_profit_early_exit_progress_fraction > 0 ? cfg->drawdown_profit_early_exit_progress_fraction : 0.5f;
        e->forced_offset = -1;
        e->obs_size = ve->obs_size;
        e->num_actions = ve->num_actions;
        memset(&e->log, 0, sizeof(Log));
    }

    fprintf(stderr, "vec_env_create: %d envs, obs=%d, actions=%d, symbols=%d\n",
            num_envs, ve->obs_size, ve->num_actions, S);
    return ve;
}

void vec_env_reset(VecEnv* ve, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < ve->num_envs; i++) {
        memset(&ve->envs[i].log, 0, sizeof(Log));
        c_reset(&ve->envs[i]);
    }
}

void vec_env_step(VecEnv* ve) {
    for (int i = 0; i < ve->num_envs; i++) {
        c_step(&ve->envs[i]);
    }
}

void vec_env_set_offset(VecEnv* ve, int env_idx, int offset) {
    if (env_idx >= 0 && env_idx < ve->num_envs) {
        ve->envs[env_idx].forced_offset = offset;
    }
}

void vec_env_get_log(const VecEnv* ve, int env_idx, Log* out) {
    if (env_idx >= 0 && env_idx < ve->num_envs) {
        *out = ve->envs[env_idx].log;
    } else {
        memset(out, 0, sizeof(Log));
    }
}

void vec_env_clear_logs(VecEnv* ve) {
    for (int i = 0; i < ve->num_envs; i++) {
        memset(&ve->envs[i].log, 0, sizeof(Log));
    }
}

void vec_env_free(VecEnv* ve) {
    if (!ve) return;
    if (ve->envs) free(ve->envs);
    if (ve->obs_buf) free(ve->obs_buf);
    if (ve->act_buf) free(ve->act_buf);
    if (ve->rew_buf) free(ve->rew_buf);
    if (ve->done_buf) free(ve->done_buf);
    if (ve->data) market_data_free(ve->data);
    free(ve);
}
