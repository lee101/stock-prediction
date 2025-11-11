#ifndef MARKET_ENV_H
#define MARKET_ENV_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// Maximum number of assets in the portfolio
#define MAX_ASSETS 512
#define MAX_HISTORY 256
#define PRICE_FEATURES 5  // Open, High, Low, Close, Volume

// Logging structure - only use floats!
typedef struct {
    float total_return;
    float sharpe_ratio;
    float max_drawdown;
    float num_trades;
    float win_rate;
    float n; // Required as the last field for pufferlib compatibility
} Log;

// Market state for a single asset
typedef struct {
    float prices[MAX_HISTORY * PRICE_FEATURES];  // OHLCV data
    float current_price;
    float volatility;
    int history_len;
} AssetState;

// Trading environment
typedef struct {
    Log log;                     // Required field for pufferlib
    float* observations;         // Required field: [num_agents, obs_size]
    int* actions;                // Required field: [num_agents] action indices
    float* rewards;              // Required field: [num_agents]
    unsigned char* terminals;    // Required field: [num_agents]

    // Market simulation state
    AssetState* assets;
    int num_assets;
    int num_agents;              // Number of parallel agents
    int current_step;
    int max_steps;

    // Portfolio state per agent
    float* cash;                 // [num_agents]
    float* positions;            // [num_agents, num_assets]
    float* portfolio_values;     // [num_agents]
    float* initial_values;       // [num_agents]

    // Trading constraints
    float transaction_cost;
    float max_position_size;
    float leverage;

    // Performance tracking per agent
    float* peak_values;          // [num_agents] for drawdown calculation
    int* total_trades;           // [num_agents]
    int* winning_trades;         // [num_agents]
} MarketEnv;

// Core functions
void c_reset(MarketEnv* env);
void c_step(MarketEnv* env);
void c_close(MarketEnv* env);

// Helper functions
void update_prices(MarketEnv* env);
void calculate_observations(MarketEnv* env, int agent_idx);
void execute_trade(MarketEnv* env, int agent_idx, int action);
float calculate_portfolio_value(MarketEnv* env, int agent_idx);
float calculate_sharpe_ratio(float* returns, int n_returns);

// Memory management
MarketEnv* create_market_env(int num_assets, int num_agents, int max_steps);
void free_market_env(MarketEnv* env);

#endif // MARKET_ENV_H
