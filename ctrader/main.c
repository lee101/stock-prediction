#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

#include "binance_rest.h"
#include "policy_infer.h"
#include "trade_loop.h"

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
    int testnet = 0;

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
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
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
