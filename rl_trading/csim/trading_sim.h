#ifndef TRADING_SIM_H
#define TRADING_SIM_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#define MAX_SYMBOLS 8

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_pnl_pct;
    float max_drawdown;
    float n_trades;
    float win_rate;
    float episode_length;
    float n;
    float score;
};

typedef struct Client Client;
struct Client {};

typedef struct TradingSim TradingSim;
struct TradingSim {
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    Log log;
    Client* client;

    float* close;
    float* high;
    float* low;
    float* features;
    int n_symbols;
    int n_bars;
    int n_features;
    int obs_dim;

    int current_bar;
    int start_bar;
    int episode_length;
    float cash;
    float inventory;
    int holding_symbol;
    float entry_price;
    int bars_held;

    float initial_cash;
    float fee_rate;
    int max_hold_bars;
    int data_start;
    int data_end;

    float episode_pnl;
    float peak_equity;
    float max_dd;
    int trade_count;
    int win_count;
    float prev_equity;
};

static inline float ts_get_close(TradingSim* env, int sym, int bar) {
    return env->close[sym * env->n_bars + bar];
}

static inline float ts_get_high(TradingSim* env, int sym, int bar) {
    return env->high[sym * env->n_bars + bar];
}

static inline float ts_get_low(TradingSim* env, int sym, int bar) {
    return env->low[sym * env->n_bars + bar];
}

static inline float ts_get_feature(TradingSim* env, int sym, int bar, int feat) {
    return env->features[(sym * env->n_bars + bar) * env->n_features + feat];
}

static inline float ts_equity(TradingSim* env) {
    float eq = env->cash;
    if (env->holding_symbol >= 0 && env->inventory > 0.0f) {
        eq += env->inventory * ts_get_close(env, env->holding_symbol, env->current_bar);
    }
    return eq;
}

static void ts_close_position(TradingSim* env) {
    if (env->holding_symbol < 0) return;
    float price = ts_get_close(env, env->holding_symbol, env->current_bar);
    float proceeds = env->inventory * price;
    float fee = proceeds * env->fee_rate;
    float pnl = proceeds - fee - (env->inventory * env->entry_price);
    env->cash += proceeds - fee;
    env->trade_count++;
    if (pnl > 0.0f) env->win_count++;
    env->inventory = 0.0f;
    env->holding_symbol = -1;
    env->entry_price = 0.0f;
    env->bars_held = 0;
}

static void ts_open_position(TradingSim* env, int sym) {
    float price = ts_get_close(env, sym, env->current_bar);
    if (price <= 0.0f) return;
    float cost = env->cash;
    float fee = cost * env->fee_rate;
    float spend = cost - fee;
    env->inventory = spend / price;
    env->cash = 0.0f;
    env->holding_symbol = sym;
    env->entry_price = price;
    env->bars_held = 0;
}

void compute_observations(TradingSim* env) {
    int idx = 0;
    for (int s = 0; s < env->n_symbols; s++) {
        for (int f = 0; f < env->n_features; f++) {
            env->observations[idx++] = ts_get_feature(env, s, env->current_bar, f);
        }
    }
    for (int s = 0; s < env->n_symbols; s++) {
        env->observations[idx++] = (env->holding_symbol == s) ? 1.0f : 0.0f;
    }
    float eq = ts_equity(env);
    float unrealized = 0.0f;
    if (env->holding_symbol >= 0 && env->entry_price > 0.0f) {
        float cur = ts_get_close(env, env->holding_symbol, env->current_bar);
        unrealized = (cur - env->entry_price) / env->entry_price;
    }
    env->observations[idx++] = unrealized;
    env->observations[idx++] = (env->max_hold_bars > 0) ?
        (float)env->bars_held / (float)env->max_hold_bars : 0.0f;
    env->observations[idx++] = eq / env->initial_cash;
}

void add_log(TradingSim* env) {
    float eq = ts_equity(env);
    float pnl_pct = (eq - env->initial_cash) / env->initial_cash;
    env->log.episode_return += eq / env->initial_cash;
    env->log.episode_pnl_pct += pnl_pct;
    env->log.max_drawdown += env->max_dd;
    env->log.n_trades += (float)env->trade_count;
    env->log.win_rate += (env->trade_count > 0) ?
        (float)env->win_count / (float)env->trade_count : 0.0f;
    env->log.episode_length += (float)(env->current_bar - env->start_bar);
    env->log.score += pnl_pct;
    env->log.n += 1.0f;
}

void init(TradingSim* env) {
    memset(&env->log, 0, sizeof(Log));
}

void allocate(TradingSim* env) {
    init(env);
    env->observations = (float*)calloc(env->obs_dim, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(TradingSim* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

void c_close(TradingSim* env) {
}

void c_render(TradingSim* env) {
}

void c_reset(TradingSim* env) {
    int range = env->data_end - env->data_start - env->episode_length;
    if (range <= 0) range = 1;
    env->start_bar = env->data_start + (rand() % range);
    env->current_bar = env->start_bar;
    env->cash = env->initial_cash;
    env->inventory = 0.0f;
    env->holding_symbol = -1;
    env->entry_price = 0.0f;
    env->bars_held = 0;
    env->episode_pnl = 0.0f;
    env->peak_equity = env->initial_cash;
    env->max_dd = 0.0f;
    env->trade_count = 0;
    env->win_count = 0;
    env->prev_equity = env->initial_cash;
    compute_observations(env);
}

void c_step(TradingSim* env) {
    int a = (int)env->actions[0];
    int n = env->n_symbols;

    if (a < 0 || a > 2 * n) a = 0;

    if (a >= 1 && a <= n) {
        int sym = a - 1;
        if (env->holding_symbol != sym) {
            if (env->holding_symbol >= 0) ts_close_position(env);
            ts_open_position(env, sym);
        }
    } else if (a >= n + 1 && a <= 2 * n) {
        int sym = a - n - 1;
        if (env->holding_symbol == sym) {
            ts_close_position(env);
        }
    }

    if (env->holding_symbol >= 0) {
        env->bars_held++;
        if (env->max_hold_bars > 0 && env->bars_held >= env->max_hold_bars) {
            ts_close_position(env);
        }
    }

    env->current_bar++;

    float eq = ts_equity(env);
    float reward = 0.0f;
    if (env->prev_equity > 0.0f) {
        reward = (eq - env->prev_equity) / env->prev_equity;
    }
    env->rewards[0] = reward * 100.0f;

    if (eq > env->peak_equity) env->peak_equity = eq;
    float dd = (env->peak_equity - eq) / env->peak_equity;  // positive drawdown
    if (dd > env->max_dd) env->max_dd = dd;
    env->prev_equity = eq;

    bool truncated = (env->current_bar >= env->start_bar + env->episode_length) ||
                     (env->current_bar >= env->data_end - 1);
    bool terminated = (eq < env->initial_cash * 0.5f);
    bool done = truncated || terminated;

    env->terminals[0] = terminated ? 1 : 0;

    if (done) {
        if (env->holding_symbol >= 0) ts_close_position(env);
        add_log(env);
        c_reset(env);
    }

    compute_observations(env);
}

#endif
