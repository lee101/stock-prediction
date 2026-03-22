#include "trade_loop.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int trade_loop_init(TradeState *state, const TradeConfig *cfg) {
    memset(state, 0, sizeof(*state));
    state->cash = 10000.0;
    state->equity = state->cash;
    (void)cfg;
    return 0;
}

static void fetch_bars(TradeState *state, const TradeConfig *cfg, BinanceClient *client) {
    for (int i = 0; i < cfg->n_symbols; i++) {
        binance_get_klines(
            client,
            cfg->symbols[i],
            "1h",
            cfg->lookback_bars,
            state->bars[i],
            &state->bar_counts[i]
        );
    }
}

static void build_observation(
    const TradeState *state,
    const TradeConfig *cfg,
    double *obs,
    int *obs_len
) {
    int k = 0;
    for (int s = 0; s < cfg->n_symbols; s++) {
        int nb = state->bar_counts[s];
        if (nb == 0) continue;
        Kline *last = &((Kline *)state->bars[s])[nb - 1];
        obs[k++] = last->close;
        obs[k++] = last->high;
        obs[k++] = last->low;
        obs[k++] = last->volume;
        obs[k++] = (double)state->positions[s].qty;
        obs[k++] = (double)state->positions[s].bars_held;
    }
    *obs_len = k;
}

static void execute_actions(
    TradeState *state,
    const TradeConfig *cfg,
    BinanceClient *client,
    const PolicyAction *actions
) {
    for (int s = 0; s < cfg->n_symbols; s++) {
        const PolicyAction *a = &actions[s];

        if (a->sell_amount > 0.01 && state->positions[s].qty > 0.0) {
            if (cfg->dry_run) {
                fprintf(stderr, "[DRY] SELL %s qty=%.6f price=%.2f\n",
                        cfg->symbols[s], a->sell_amount, a->sell_price);
            } else {
                OrderResponse resp;
                binance_place_order(client, cfg->symbols[s],
                    ORDER_SELL, ORDER_LIMIT, a->sell_price, a->sell_amount, &resp);
            }
        }

        if (a->buy_amount > 0.01 && state->positions[s].qty <= 0.0 &&
            state->active_positions < cfg->max_positions)
        {
            if (cfg->dry_run) {
                fprintf(stderr, "[DRY] BUY %s qty=%.6f price=%.2f\n",
                        cfg->symbols[s], a->buy_amount, a->buy_price);
            } else {
                OrderResponse resp;
                binance_place_order(client, cfg->symbols[s],
                    ORDER_BUY, ORDER_LIMIT, a->buy_price, a->buy_amount, &resp);
            }
        }
    }
}

int trade_loop_cycle(TradeState *state, const TradeConfig *cfg, BinanceClient *client, Policy *policy) {
    fetch_bars(state, cfg, client);

    double obs[POLICY_MAX_OBS];
    int obs_len = 0;
    build_observation(state, cfg, obs, &obs_len);

    PolicyAction actions[MAX_TRADE_SYMBOLS];
    memset(actions, 0, sizeof(actions));
    int rc = policy_forward(policy, obs, obs_len, actions, cfg->n_symbols);
    if (rc != 0) {
        fprintf(stderr, "cycle %d: policy_forward failed\n", state->cycle_count);
        return -1;
    }

    execute_actions(state, cfg, client, actions);
    trade_loop_log(state, cfg, actions);

    state->cycle_count++;
    return 0;
}

void trade_loop_log(const TradeState *state, const TradeConfig *cfg, const PolicyAction *actions) {
    time_t now = time(NULL);
    struct tm *tm = gmtime(&now);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", tm);

    FILE *f = stderr;
    if (cfg->log_path[0]) {
        f = fopen(cfg->log_path, "a");
        if (!f) f = stderr;
    }

    fprintf(f, "[%s] cycle=%d equity=%.2f cash=%.2f positions=%d\n",
            ts, state->cycle_count, state->equity, state->cash, state->active_positions);

    for (int s = 0; s < cfg->n_symbols; s++) {
        if (actions[s].buy_amount > 0.01 || actions[s].sell_amount > 0.01) {
            fprintf(f, "  %s: bp=%.2f sp=%.2f ba=%.4f sa=%.4f pos=%.6f\n",
                    cfg->symbols[s],
                    actions[s].buy_price, actions[s].sell_price,
                    actions[s].buy_amount, actions[s].sell_amount,
                    state->positions[s].qty);
        }
    }

    if (f != stderr) fclose(f);
}
