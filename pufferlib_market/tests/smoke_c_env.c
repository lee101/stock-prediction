/*
 * smoke_c_env.c — minimal standalone harness that exercises trading_env.c
 * end-to-end without the Python binding. Loads a market data file, allocates
 * a TradingEnv, calls c_reset + c_step in a loop, and asserts invariants.
 *
 * Build:
 *   clang -O1 -g -Wall -Wextra -fsanitize=address,undefined \
 *         -fno-omit-frame-pointer -I pufferlib_market/include \
 *         pufferlib_market/src/trading_env.c \
 *         pufferlib_market/tests/smoke_c_env.c -lm -o /tmp/smoke_c_env
 *
 * Run:
 *   /tmp/smoke_c_env pufferlib_market/data/alpaca_daily_train.bin
 */
#include "../include/trading_env.h"
#include <assert.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s data.bin [num_steps]\n", argv[0]);
        return 2;
    }
    int num_steps = (argc >= 3) ? atoi(argv[2]) : 200;

    MarketData* md = market_data_load(argv[1]);
    if (!md) return 1;

    TradingEnv env;
    memset(&env, 0, sizeof(env));
    env.data = md;
    env.max_steps = 100;
    env.fee_rate = 0.001f;
    env.max_leverage = 1.0f;
    env.periods_per_year = 8760.0f;
    env.action_allocation_bins = 1;
    env.action_level_bins = 1;
    env.action_max_offset_bps = 0.0f;
    env.reward_scale = 1.0f;
    env.reward_clip = 5.0f;
    env.forced_offset = -1;

    int per_sym = env.action_allocation_bins * env.action_level_bins;
    env.num_actions = 1 + 2 * md->num_symbols * per_sym;
    env.obs_size = md->num_symbols * md->features_per_sym + 5 + md->num_symbols;

    env.observations = (float*)calloc(env.obs_size, sizeof(float));
    env.actions      = (int*)calloc(1, sizeof(int));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    int n_terminals = 0;
    int n_trades = 0;
    for (int i = 0; i < num_steps; i++) {
        env.actions[0] = i % env.num_actions;
        c_step(&env);
        if (env.terminals[0]) n_terminals++;
        assert(env.observations[0] == env.observations[0]); /* not NaN */
    }
    n_trades = env.agent.num_trades;

    fprintf(stderr, "smoke: steps=%d terminals=%d trades=%d final_cash=%.2f\n",
            num_steps, n_terminals, n_trades, env.agent.cash);

    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
    market_data_free(md);
    return 0;
}
