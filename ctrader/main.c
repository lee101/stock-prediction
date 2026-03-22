#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

#include "binance_rest.h"
#include "policy_infer.h"
#include "trade_loop.h"
#include "market_sim.h"
#include "mktd_reader.h"

static volatile int g_running = 1;

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --api-key KEY       Binance API key\n"
        "  --secret-key KEY    Binance secret key\n"
        "  --model PATH        TorchScript model path\n"
        "  --symbols SYM,...   Comma-separated symbols\n"
        "  --interval SEC      Cycle interval (default 3600)\n"
        "  --max-leverage N    Max leverage (default 5.0)\n"
        "  --fee-rate N        Fee rate (default 0.001)\n"
        "  --max-hold N        Max hold hours (default 6)\n"
        "  --max-positions N   Max simultaneous positions (default 5)\n"
        "  --dry-run           Don't place real orders\n"
        "  --log PATH          Log file path\n"
        "  --testnet           Use Binance testnet\n"
        "  --backtest          Run offline backtest on MKTD data\n"
        "  --data-file PATH    MKTD binary data file for backtest\n"
        , prog);
}

static int parse_symbols(const char *csv, TradeConfig *cfg) {
    char buf[1024];
    strncpy(buf, csv, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    cfg->n_symbols = 0;
    char *tok = strtok(buf, ",");
    while (tok && cfg->n_symbols < MAX_TRADE_SYMBOLS) {
        strncpy(cfg->symbols[cfg->n_symbols], tok, BINANCE_MAX_SYMBOL - 1);
        cfg->n_symbols++;
        tok = strtok(NULL, ",");
    }
    return cfg->n_symbols;
}

static int run_backtest(const char *data_file, double fee_rate) {
    MktdData data;
    if (mktd_load(data_file, &data) != 0) {
        fprintf(stderr, "backtest: failed to load %s\n", data_file);
        return 1;
    }

    int S = data.num_symbols;
    int T = data.num_timesteps;
    double initial_cash = 10000.0;

    fprintf(stderr, "backtest: %d symbols, %d timesteps, fee=%.4f\n", S, T, fee_rate);

    double *per_sym_return = (double *)calloc((size_t)S, sizeof(double));

    for (int s = 0; s < S; s++) {
        if (T < 2) { per_sym_return[s] = 0.0; continue; }
        float first_close = data.closes[0 * S + s];
        float last_close = data.closes[(T - 1) * S + s];
        if (first_close > 0.0f) {
            per_sym_return[s] = (double)last_close / (double)first_close - 1.0 - 2.0 * fee_rate;
        }
    }

    /* portfolio-level equity curve: equal-weight buy-and-hold all symbols */
    double *equity_curve = (double *)malloc((size_t)(T + 1) * sizeof(double));
    equity_curve[0] = initial_cash;
    double alloc_per_sym = initial_cash / S;

    /* compute shares bought at t=0 for each symbol */
    double *shares = (double *)calloc((size_t)S, sizeof(double));
    double remaining_cash = initial_cash;
    for (int s = 0; s < S; s++) {
        float c0 = data.closes[0 * S + s];
        if (c0 > 0.0f) {
            double cost_per_share = (double)c0 * (1.0 + fee_rate);
            shares[s] = alloc_per_sym / cost_per_share;
            remaining_cash -= shares[s] * cost_per_share;
        }
    }

    for (int t = 0; t < T; t++) {
        double eq = remaining_cash;
        for (int s = 0; s < S; s++) {
            eq += shares[s] * (double)data.closes[t * S + s];
        }
        equity_curve[t + 1] = eq;
    }

    /* close all positions at final bar */
    double final_cash = remaining_cash;
    for (int s = 0; s < S; s++) {
        final_cash += shares[s] * (double)data.closes[(T - 1) * S + s] * (1.0 - fee_rate);
    }

    int n_eq = T + 1;
    double sortino = compute_sortino(equity_curve, n_eq);
    double max_dd = compute_max_drawdown(equity_curve, n_eq);
    double total_return = (final_cash - initial_cash) / initial_cash;
    int total_trades = 2 * S;

    printf("=== Backtest Results (buy-and-hold baseline) ===\n");
    printf("total_return:  %+.4f%%\n", total_return * 100.0);
    printf("sortino:       %.4f\n", sortino);
    printf("max_drawdown:  %.4f%%\n", max_dd * 100.0);
    printf("num_trades:    %d\n", total_trades);
    printf("final_equity:  %.2f\n", final_cash);
    printf("initial_cash:  %.2f\n", initial_cash);
    printf("\n--- Per-symbol returns ---\n");
    for (int s = 0; s < S; s++) {
        printf("  %-15s  %+.4f%%\n", data.symbol_names[s], per_sym_return[s] * 100.0);
    }

    free(equity_curve);
    free(shares);
    free(per_sym_return);
    mktd_free(&data);
    return 0;
}

int main(int argc, char **argv) {
    TradeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lookback_bars = 48;
    cfg.cycle_interval_sec = 3600;
    cfg.max_leverage = 5.0;
    cfg.fee_rate = 0.001;
    cfg.max_hold_hours = 6;
    cfg.max_positions = 5;
    cfg.force_close_slippage = 0.005;
    cfg.dry_run = 1;

    char api_key[128] = {0};
    char secret_key[128] = {0};
    char model_path[256] = {0};
    char data_file[512] = {0};
    int testnet = 0;
    int backtest_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--api-key") && i + 1 < argc) {
            strncpy(api_key, argv[++i], sizeof(api_key) - 1);
        } else if (!strcmp(argv[i], "--secret-key") && i + 1 < argc) {
            strncpy(secret_key, argv[++i], sizeof(secret_key) - 1);
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            strncpy(model_path, argv[++i], sizeof(model_path) - 1);
        } else if (!strcmp(argv[i], "--symbols") && i + 1 < argc) {
            parse_symbols(argv[++i], &cfg);
        } else if (!strcmp(argv[i], "--interval") && i + 1 < argc) {
            cfg.cycle_interval_sec = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--max-leverage") && i + 1 < argc) {
            cfg.max_leverage = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--fee-rate") && i + 1 < argc) {
            cfg.fee_rate = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--max-hold") && i + 1 < argc) {
            cfg.max_hold_hours = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--max-positions") && i + 1 < argc) {
            cfg.max_positions = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--dry-run")) {
            cfg.dry_run = 1;
        } else if (!strcmp(argv[i], "--live")) {
            cfg.dry_run = 0;
        } else if (!strcmp(argv[i], "--log") && i + 1 < argc) {
            strncpy(cfg.log_path, argv[++i], sizeof(cfg.log_path) - 1);
        } else if (!strcmp(argv[i], "--testnet")) {
            testnet = 1;
        } else if (!strcmp(argv[i], "--backtest")) {
            backtest_mode = 1;
        } else if (!strcmp(argv[i], "--data-file") && i + 1 < argc) {
            strncpy(data_file, argv[++i], sizeof(data_file) - 1);
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (backtest_mode) {
        if (!data_file[0]) {
            fprintf(stderr, "error: --backtest requires --data-file\n");
            return 1;
        }
        return run_backtest(data_file, cfg.fee_rate);
    }

    if (cfg.n_symbols == 0) {
        parse_symbols("BTCUSDT,ETHUSDT,SOLUSDT", &cfg);
    }

    fprintf(stderr, "ctrader: %d symbols, interval=%ds, leverage=%.1f, dry_run=%d\n",
            cfg.n_symbols, cfg.cycle_interval_sec, cfg.max_leverage, cfg.dry_run);

    BinanceClient client;
    binance_init(&client, api_key, secret_key);
    if (testnet) binance_set_testnet(&client);

    Policy policy;
    memset(&policy, 0, sizeof(policy));
    if (model_path[0]) {
        if (policy_load(&policy, model_path) != 0) {
            fprintf(stderr, "warning: model load failed, running without policy\n");
        }
    }

    TradeState state;
    trade_loop_init(&state, &cfg);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    while (g_running) {
        trade_loop_cycle(&state, &cfg, &client, &policy);
        if (!g_running) break;
        sleep(cfg.cycle_interval_sec);
    }

    fprintf(stderr, "shutting down after %d cycles\n", state.cycle_count);
    policy_unload(&policy);
    binance_cleanup(&client);
    return 0;
}
