#ifndef TRADE_LOOP_H
#define TRADE_LOOP_H

#include "binance_rest.h"
#include "policy_infer.h"
#include "market_sim.h"

#define MAX_TRADE_SYMBOLS 32

typedef struct {
    char symbols[MAX_TRADE_SYMBOLS][BINANCE_MAX_SYMBOL];
    int n_symbols;
    int lookback_bars;
    int cycle_interval_sec;
    double max_leverage;
    double fee_rate;
    int max_hold_hours;
    int max_positions;
    double force_close_slippage;
    int dry_run;
    char log_path[256];
} TradeConfig;

typedef struct {
    Kline bars[MAX_TRADE_SYMBOLS][BINANCE_MAX_KLINES];
    int bar_counts[MAX_TRADE_SYMBOLS];
    Position positions[MAX_TRADE_SYMBOLS];
    int active_positions;
    double cash;
    double equity;
    int cycle_count;
} TradeState;

int trade_loop_init(TradeState *state, const TradeConfig *cfg);
int trade_loop_cycle(TradeState *state, const TradeConfig *cfg, BinanceClient *client, Policy *policy);
void trade_loop_log(const TradeState *state, const TradeConfig *cfg, const PolicyAction *actions);

#endif
