#include "../include/market_env.h"
#include <stdio.h>

// Create and initialize market environment
MarketEnv* create_market_env(int num_assets, int num_agents, int max_steps) {
    MarketEnv* env = (MarketEnv*)malloc(sizeof(MarketEnv));

    env->num_assets = num_assets;
    env->num_agents = num_agents;
    env->max_steps = max_steps;
    env->current_step = 0;

    // Allocate arrays
    env->assets = (AssetState*)calloc(num_assets, sizeof(AssetState));

    // Observation size: OHLCV features * history + portfolio state
    int obs_size = PRICE_FEATURES * MAX_HISTORY + num_assets + 3; // +3 for cash, total_value, step_ratio
    env->observations = (float*)calloc(num_agents * obs_size, sizeof(float));

    env->actions = (int*)calloc(num_agents, sizeof(int));
    env->rewards = (float*)calloc(num_agents, sizeof(float));
    env->terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));

    env->cash = (float*)calloc(num_agents, sizeof(float));
    env->positions = (float*)calloc(num_agents * num_assets, sizeof(float));
    env->portfolio_values = (float*)calloc(num_agents, sizeof(float));
    env->initial_values = (float*)calloc(num_agents, sizeof(float));
    env->peak_values = (float*)calloc(num_agents, sizeof(float));
    env->total_trades = (int*)calloc(num_agents, sizeof(int));
    env->winning_trades = (int*)calloc(num_agents, sizeof(int));

    // Trading parameters
    env->transaction_cost = 0.001f;  // 0.1% transaction cost
    env->max_position_size = 0.25f;  // Max 25% of portfolio per position
    env->leverage = 1.0f;             // No leverage by default

    return env;
}

void free_market_env(MarketEnv* env) {
    if (env) {
        free(env->assets);
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        free(env->cash);
        free(env->positions);
        free(env->portfolio_values);
        free(env->initial_values);
        free(env->peak_values);
        free(env->total_trades);
        free(env->winning_trades);
        free(env);
    }
}

// Simple random walk price generator (replace with real data in production)
void update_prices(MarketEnv* env) {
    for (int i = 0; i < env->num_assets; i++) {
        AssetState* asset = &env->assets[i];

        // Generate synthetic price movement (geometric brownian motion)
        float drift = 0.0001f;
        float volatility = 0.02f;
        float random_shock = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

        float price_change = asset->current_price * (drift + volatility * random_shock);
        asset->current_price += price_change;

        // Ensure positive price
        if (asset->current_price < 0.01f) {
            asset->current_price = 0.01f;
        }

        // Update OHLCV (simplified)
        int idx = (env->current_step % MAX_HISTORY) * PRICE_FEATURES;
        asset->prices[idx] = asset->current_price;      // Open
        asset->prices[idx + 1] = asset->current_price;  // High
        asset->prices[idx + 2] = asset->current_price;  // Low
        asset->prices[idx + 3] = asset->current_price;  // Close
        asset->prices[idx + 4] = 1000.0f;               // Volume

        asset->history_len = (asset->history_len < MAX_HISTORY) ?
                             asset->history_len + 1 : MAX_HISTORY;
    }
}

void calculate_observations(MarketEnv* env, int agent_idx) {
    int obs_size = PRICE_FEATURES * MAX_HISTORY + env->num_assets + 3;
    float* obs = &env->observations[agent_idx * obs_size];
    int idx = 0;

    // Price history for first asset (expand to all assets as needed)
    for (int i = 0; i < MAX_HISTORY * PRICE_FEATURES; i++) {
        obs[idx++] = (i < env->assets[0].history_len * PRICE_FEATURES) ?
                     env->assets[0].prices[i] / 100.0f : 0.0f; // Normalize
    }

    // Current positions
    for (int i = 0; i < env->num_assets; i++) {
        obs[idx++] = env->positions[agent_idx * env->num_assets + i];
    }

    // Portfolio state
    obs[idx++] = env->cash[agent_idx] / 100000.0f;  // Normalized cash
    obs[idx++] = env->portfolio_values[agent_idx] / 100000.0f;  // Normalized value
    obs[idx++] = (float)env->current_step / env->max_steps;  // Progress
}

float calculate_portfolio_value(MarketEnv* env, int agent_idx) {
    float value = env->cash[agent_idx];

    for (int i = 0; i < env->num_assets; i++) {
        float position = env->positions[agent_idx * env->num_assets + i];
        value += position * env->assets[i].current_price;
    }

    return value;
}

void execute_trade(MarketEnv* env, int agent_idx, int action) {
    // Action space: [0: hold, 1-N: buy asset i, N+1-2N: sell asset i]
    if (action == 0) {
        return; // Hold
    }

    int asset_idx = (action - 1) % env->num_assets;
    bool is_buy = (action - 1) < env->num_assets;

    float current_price = env->assets[asset_idx].current_price;
    float portfolio_value = calculate_portfolio_value(env, agent_idx);
    float max_trade_value = portfolio_value * env->max_position_size;

    if (is_buy) {
        // Buy
        float max_shares = (env->cash[agent_idx] * (1.0f - env->transaction_cost)) / current_price;
        float shares_to_buy = fminf(max_shares, max_trade_value / current_price);

        if (shares_to_buy > 0.0f) {
            float cost = shares_to_buy * current_price * (1.0f + env->transaction_cost);
            if (cost <= env->cash[agent_idx]) {
                env->cash[agent_idx] -= cost;
                env->positions[agent_idx * env->num_assets + asset_idx] += shares_to_buy;
                env->total_trades[agent_idx]++;
            }
        }
    } else {
        // Sell
        float current_position = env->positions[agent_idx * env->num_assets + asset_idx];

        if (current_position > 0.0f) {
            float old_value = current_position * current_price;
            env->cash[agent_idx] += old_value * (1.0f - env->transaction_cost);

            // Track win/loss
            float position_cost = old_value; // Simplified, should track actual cost basis
            if (old_value > position_cost) {
                env->winning_trades[agent_idx]++;
            }

            env->positions[agent_idx * env->num_assets + asset_idx] = 0.0f;
            env->total_trades[agent_idx]++;
        }
    }
}

void c_reset(MarketEnv* env) {
    env->current_step = 0;

    // Reset all agents
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        env->cash[agent_idx] = 100000.0f;  // Starting cash
        env->initial_values[agent_idx] = 100000.0f;
        env->portfolio_values[agent_idx] = 100000.0f;
        env->peak_values[agent_idx] = 100000.0f;
        env->total_trades[agent_idx] = 0;
        env->winning_trades[agent_idx] = 0;
        env->terminals[agent_idx] = 0;
        env->rewards[agent_idx] = 0.0f;

        // Clear positions
        for (int i = 0; i < env->num_assets; i++) {
            env->positions[agent_idx * env->num_assets + i] = 0.0f;
        }
    }

    // Initialize asset prices
    for (int i = 0; i < env->num_assets; i++) {
        env->assets[i].current_price = 100.0f + (float)(rand() % 100);
        env->assets[i].history_len = 0;
    }

    // Generate initial price history
    for (int step = 0; step < MAX_HISTORY; step++) {
        update_prices(env);
    }

    // Calculate initial observations
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        calculate_observations(env, agent_idx);
    }
}

void c_step(MarketEnv* env) {
    // Update prices
    update_prices(env);

    // Execute actions for all agents
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        float old_value = calculate_portfolio_value(env, agent_idx);

        // Execute trade
        execute_trade(env, agent_idx, env->actions[agent_idx]);

        // Calculate new portfolio value
        float new_value = calculate_portfolio_value(env, agent_idx);
        env->portfolio_values[agent_idx] = new_value;

        // Reward is the percentage change in portfolio value
        float return_pct = (new_value - old_value) / old_value;
        env->rewards[agent_idx] = return_pct * 100.0f;  // Scale reward

        // Update peak for drawdown calculation
        if (new_value > env->peak_values[agent_idx]) {
            env->peak_values[agent_idx] = new_value;
        }

        // Update observations
        calculate_observations(env, agent_idx);

        // Check terminal condition
        env->terminals[agent_idx] = 0;
        if (env->current_step >= env->max_steps - 1) {
            env->terminals[agent_idx] = 1;

            // Update log statistics
            float total_return = (new_value - env->initial_values[agent_idx]) /
                               env->initial_values[agent_idx];
            float max_drawdown = (env->peak_values[agent_idx] - new_value) /
                               env->peak_values[agent_idx];
            float win_rate = (env->total_trades[agent_idx] > 0) ?
                           (float)env->winning_trades[agent_idx] / env->total_trades[agent_idx] : 0.0f;

            env->log.total_return = total_return;
            env->log.max_drawdown = max_drawdown;
            env->log.num_trades = (float)env->total_trades[agent_idx];
            env->log.win_rate = win_rate;
            env->log.n += 1;
        }
    }

    env->current_step++;
}

void c_close(MarketEnv* env) {
    // Cleanup if needed
}
