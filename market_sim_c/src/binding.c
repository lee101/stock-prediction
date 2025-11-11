#include "../include/market_env.h"

#define Env MarketEnv
#include "../../external/pufferlib-3.0.0/pufferlib/ocean/env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    int num_assets = unpack(kwargs, "num_assets");
    int num_agents = unpack(kwargs, "num_agents");
    int max_steps = unpack(kwargs, "max_steps");

    // Initialize the environment structure directly
    env->num_assets = (num_assets > 0 && num_assets <= MAX_ASSETS) ? num_assets : 10;
    env->num_agents = (num_agents > 0) ? num_agents : 1;
    env->max_steps = (max_steps > 0) ? max_steps : 1000;
    env->current_step = 0;

    // Allocate arrays
    env->assets = (AssetState*)calloc(env->num_assets, sizeof(AssetState));

    // Observation size: OHLCV features * history + portfolio state
    int obs_size = PRICE_FEATURES * MAX_HISTORY + env->num_assets + 3;

    env->cash = (float*)calloc(env->num_agents, sizeof(float));
    env->positions = (float*)calloc(env->num_agents * env->num_assets, sizeof(float));
    env->portfolio_values = (float*)calloc(env->num_agents, sizeof(float));
    env->initial_values = (float*)calloc(env->num_agents, sizeof(float));
    env->peak_values = (float*)calloc(env->num_agents, sizeof(float));
    env->total_trades = (int*)calloc(env->num_agents, sizeof(int));
    env->winning_trades = (int*)calloc(env->num_agents, sizeof(int));

    // Trading parameters
    env->transaction_cost = 0.001f;
    env->max_position_size = 0.25f;
    env->leverage = 1.0f;

    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "total_return", log->total_return);
    assign_to_dict(dict, "sharpe_ratio", log->sharpe_ratio);
    assign_to_dict(dict, "max_drawdown", log->max_drawdown);
    assign_to_dict(dict, "num_trades", log->num_trades);
    assign_to_dict(dict, "win_rate", log->win_rate);
    return 0;
}

static void my_free(Env* env) {
    if (env->assets) free(env->assets);
    if (env->cash) free(env->cash);
    if (env->positions) free(env->positions);
    if (env->portfolio_values) free(env->portfolio_values);
    if (env->initial_values) free(env->initial_values);
    if (env->peak_values) free(env->peak_values);
    if (env->total_trades) free(env->total_trades);
    if (env->winning_trades) free(env->winning_trades);
}
