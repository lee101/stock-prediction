/*
 * ctrader/binance_bot/backtest_main.c
 *
 * End-to-end C backtest: reads an MKTD stocks daily bin, loads a .ctrdpol
 * policy exported from a pufferlib_market checkpoint, walks a 90-day window
 * forward, picks deterministic argmax actions, and simulates cash/position
 * bookkeeping in pure C.  Prints total return + num trades + equity curve
 * summary.
 *
 * Goal: match `python -m pufferlib_market.evaluate_holdout` on the same
 * (checkpoint, data, window) within ~1% total return, confirming the C
 * pipeline is faithful enough to drive live trading.
 *
 * Action encoding (for 1-bin alloc/level; matches TradingPolicy in train.py):
 *   0               -> FLAT   (close any position)
 *   1..S            -> LONG   on sym (action-1) with cash_allocation=100%
 *   S+1..2S         -> SHORT  on sym (action-1-S) with cash_allocation=100%
 *
 * Fees: taker fee_rate applied to buy_notional + sell_notional on each fill.
 * Simplifications for v0: full-allocation, market-on-close, no slippage/fill
 * buffer.  Those will be added once the headline PnL matches Python.
 */
#include "obs_builder.h"
#include "policy_mlp.h"
#include "../mktd_reader.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *data_path;
    const char *policy_path;
    int window_start;
    int window_steps;
    double initial_cash;
    double fee_rate;
    int verbose;
} BacktestArgs;

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --data FILE --policy FILE [--window-start N] [--window-steps N]\n"
        "          [--initial-cash N] [--fee-rate N] [--verbose]\n", prog);
}

static int parse_args(int argc, char **argv, BacktestArgs *a) {
    a->data_path = NULL; a->policy_path = NULL;
    a->window_start = 0; a->window_steps = 90;
    a->initial_cash = 10000.0; a->fee_rate = 0.001;
    a->verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        if (!strcmp(k, "--data") && i + 1 < argc) a->data_path = argv[++i];
        else if (!strcmp(k, "--policy") && i + 1 < argc) a->policy_path = argv[++i];
        else if (!strcmp(k, "--window-start") && i + 1 < argc) a->window_start = atoi(argv[++i]);
        else if (!strcmp(k, "--window-steps") && i + 1 < argc) a->window_steps = atoi(argv[++i]);
        else if (!strcmp(k, "--initial-cash") && i + 1 < argc) a->initial_cash = atof(argv[++i]);
        else if (!strcmp(k, "--fee-rate") && i + 1 < argc) a->fee_rate = atof(argv[++i]);
        else if (!strcmp(k, "--verbose")) a->verbose = 1;
        else { usage(argv[0]); return -1; }
    }
    if (!a->data_path || !a->policy_path) { usage(argv[0]); return -1; }
    return 0;
}

static double close_at(const MktdData *d, int t, int sym) {
    return (double)d->closes[(size_t)t * d->num_symbols + sym];
}

int main(int argc, char **argv) {
    BacktestArgs ag;
    if (parse_args(argc, argv, &ag) != 0) return 2;

    MktdData data;
    if (mktd_load(ag.data_path, &data) != 0) {
        fprintf(stderr, "backtest: mktd_load %s failed\n", ag.data_path);
        return 1;
    }
    fprintf(stderr, "data: %d symbols × %d timesteps (features_per_sym=%d)\n",
            data.num_symbols, data.num_timesteps, data.features_per_sym);

    CtrdpolPolicy pol;
    if (ctrdpol_load(&pol, ag.policy_path) != 0) {
        fprintf(stderr, "backtest: ctrdpol_load %s failed\n", ag.policy_path);
        mktd_free(&data);
        return 1;
    }
    fprintf(stderr, "policy: obs_dim=%u num_actions=%u num_symbols=%u\n",
            pol.hdr.obs_dim, pol.hdr.num_actions, pol.hdr.num_symbols);

    int S = data.num_symbols;
    int expected_obs = S * data.features_per_sym + 5 + S;
    if ((int)pol.hdr.obs_dim != expected_obs) {
        fprintf(stderr, "backtest: policy obs_dim=%u but data builds %d (S=%d F=%d)\n",
                pol.hdr.obs_dim, expected_obs, S, data.features_per_sym);
        ctrdpol_free(&pol); mktd_free(&data);
        return 1;
    }
    if ((int)pol.hdr.num_actions != 1 + 2 * S) {
        fprintf(stderr, "backtest: policy num_actions=%u expected %d for S=%d symbols\n",
                pol.hdr.num_actions, 1 + 2 * S, S);
        ctrdpol_free(&pol); mktd_free(&data);
        return 1;
    }

    int t0 = ag.window_start;
    int t1 = ag.window_start + ag.window_steps;
    if (t1 > data.num_timesteps) {
        fprintf(stderr, "backtest: window [%d,%d) exceeds num_timesteps=%d\n",
                t0, t1, data.num_timesteps);
        ctrdpol_free(&pol); mktd_free(&data);
        return 1;
    }

    CtrdpolPortfolioState pf;
    ctrdpol_init_portfolio(&pf, ag.initial_cash, ag.window_steps);

    float *obs = (float *)calloc(pol.hdr.obs_dim, sizeof(float));
    float *logits = (float *)calloc(pol.hdr.num_actions, sizeof(float));
    if (!obs || !logits) { fprintf(stderr, "oom\n"); return 1; }

    int num_trades = 0;
    double initial_equity = pf.equity;
    double peak = initial_equity;
    double max_dd = 0.0;

    for (int step = 0; step < ag.window_steps; step++) {
        int t = t0 + step;
        pf.step = step;
        if (ctrdpol_build_obs(&data, t, &pf, obs, (int)pol.hdr.obs_dim) != 0) {
            fprintf(stderr, "step %d: build_obs failed\n", step);
            break;
        }
        ctrdpol_forward(&pol, obs, logits);
        int action = ctrdpol_argmax(logits, (int)pol.hdr.num_actions);

        /* Mark-to-market current equity at this bar's close BEFORE acting. */
        double equity = pf.cash;
        if (pf.current_position >= 0) {
            int sym_idx = pf.current_position % S;
            int is_short = (pf.current_position >= S);
            double cur = close_at(&data, t, sym_idx);
            equity += (is_short ? -1.0 : 1.0) * pf.position_qty * cur;
        }
        pf.equity = equity;
        if (equity > peak) peak = equity;
        if (peak > 0) {
            double dd = (peak - equity) / peak;
            if (dd > max_dd) max_dd = dd;
        }

        /* Interpret action: 0=flat, 1..S=long, S+1..2S=short. */
        int target_pos = -1;
        if (action == 0) target_pos = -1;
        else if (action >= 1 && action <= S) target_pos = action - 1;            /* long sym */
        else if (action >= S + 1 && action <= 2 * S) target_pos = (action - 1);  /* short sym in [S,2S-1] */

        int is_short_target = (target_pos >= S);
        int sym_idx_target = (target_pos >= 0) ? (target_pos % S) : -1;

        /* If flipping or closing, liquidate current position first at current close. */
        if (pf.current_position != target_pos && pf.current_position >= 0) {
            int sym_idx = pf.current_position % S;
            int is_short = (pf.current_position >= S);
            double cur = close_at(&data, t, sym_idx);
            double notional = pf.position_qty * cur;
            double fee = notional * ag.fee_rate;
            if (is_short) {
                /* cover short: pay back at current price from cash */
                pf.cash -= notional + fee;
                /* short entry originally credited cash by entry_notional; here we
                 * only track net cash movement so the PnL is:
                 *   (entry_notional - exit_notional) - fees
                 * which is already reflected because entry added entry_notional to cash
                 * and the cur-close subtraction here removes exit_notional. */
            } else {
                pf.cash += notional - fee;
            }
            num_trades++;
            pf.current_position = -1;
            pf.position_qty = 0.0;
            pf.hold_hours = 0;
        }

        /* Open new target position if one is requested and we're flat. */
        if (target_pos >= 0 && pf.current_position == -1) {
            double cur = close_at(&data, t, sym_idx_target);
            if (cur > 0.0) {
                double notional = pf.cash;  /* full-allocation, matches 1/1 bins */
                double qty = notional / cur;
                double fee = notional * ag.fee_rate;
                if (is_short_target) {
                    /* short entry: cash += notional - fee, positive entry_qty represents shares shorted */
                    pf.cash = pf.cash + notional - fee; /* receive short proceeds minus fee */
                } else {
                    pf.cash = pf.cash - notional - fee;
                }
                pf.position_qty = qty;
                pf.entry_price = cur;
                pf.current_position = target_pos;
                pf.hold_hours = 0;
                num_trades++;
            }
        } else if (pf.current_position == target_pos && target_pos >= 0) {
            pf.hold_hours++;
        }

        if (ag.verbose) {
            fprintf(stderr, "  step=%3d t=%3d action=%d pos=%d qty=%.6f cash=%.2f equity=%.2f\n",
                    step, t, action, pf.current_position, pf.position_qty, pf.cash, equity);
        }
    }

    /* Force-close any open position at final bar close. */
    int t_final = t1 - 1;
    if (pf.current_position >= 0) {
        int sym_idx = pf.current_position % S;
        int is_short = (pf.current_position >= S);
        double cur = close_at(&data, t_final, sym_idx);
        double notional = pf.position_qty * cur;
        double fee = notional * ag.fee_rate;
        if (is_short) pf.cash -= notional + fee;
        else          pf.cash += notional - fee;
        num_trades++;
        pf.current_position = -1;
    }
    double final_equity = pf.cash;
    double total_return = (final_equity - initial_equity) / initial_equity;

    printf("BACKTEST RESULT\n");
    printf("  data           = %s\n", ag.data_path);
    printf("  policy         = %s\n", ag.policy_path);
    printf("  window         = [%d,%d)  (%d bars)\n", t0, t1, ag.window_steps);
    printf("  initial_equity = %.2f\n", initial_equity);
    printf("  final_equity   = %.2f\n", final_equity);
    printf("  total_return   = %+.4f (%+.2f%%)\n", total_return, total_return * 100.0);
    printf("  max_drawdown   = %.4f (%.2f%%)\n", max_dd, max_dd * 100.0);
    printf("  num_trades     = %d\n", num_trades);

    free(obs); free(logits);
    ctrdpol_free(&pol);
    mktd_free(&data);
    return 0;
}
