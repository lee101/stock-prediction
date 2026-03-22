#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../market_sim.h"

static int g_pass = 0, g_fail = 0;
static const double TOL = 1e-4;
static const double TOL_TIGHT = 1e-8;

#define CHECK_NEAR(a, b, tol, label) do { \
    double _a = (a), _b = (b), _t = (tol); \
    if (fabs(_a - _b) > _t) { \
        fprintf(stderr, "  FAIL %s: got %.10f expected %.10f (diff=%.2e, tol=%.2e)\n", \
                label, _a, _b, fabs(_a - _b), _t); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

#define CHECK_INT(a, b, label) do { \
    int _a = (a), _b = (b); \
    if (_a != _b) { \
        fprintf(stderr, "  FAIL %s: got %d expected %d\n", label, _a, _b); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

static int read_i32(FILE *f) {
    int v;
    if (fread(&v, 4, 1, f) != 1) { fprintf(stderr, "read error\n"); exit(1); }
    return v;
}

static double read_f64(FILE *f) {
    double v;
    if (fread(&v, 8, 1, f) != 1) { fprintf(stderr, "read error\n"); exit(1); }
    return v;
}

static void read_f64_array(FILE *f, double *out, int n) {
    if (fread(out, 8, n, f) != (size_t)n) { fprintf(stderr, "read error\n"); exit(1); }
}

int main(int argc, char **argv) {
    const char *path = "tests/parity_cases.bin";
    if (argc > 1) path = argv[1];

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }

    int n_cases = read_i32(f);
    fprintf(stderr, "=== sim parity tests: %d cases ===\n", n_cases);

    for (int ci = 0; ci < n_cases; ci++) {
        int n_bars = read_i32(f);

        double max_leverage = read_f64(f);
        double maker_fee = read_f64(f);
        double margin_hourly_rate = read_f64(f);
        double initial_cash = read_f64(f);
        double fill_buffer_pct = read_f64(f);
        double min_edge = read_f64(f);
        double intensity_scale = read_f64(f);
        int can_short = read_i32(f);
        int max_hold_bars = read_i32(f);

        double *high = (double *)malloc(n_bars * sizeof(double));
        double *low = (double *)malloc(n_bars * sizeof(double));
        double *close_arr = (double *)malloc(n_bars * sizeof(double));
        double *buy_price = (double *)malloc(n_bars * sizeof(double));
        double *sell_price = (double *)malloc(n_bars * sizeof(double));
        double *buy_amount = (double *)malloc(n_bars * sizeof(double));
        double *sell_amount = (double *)malloc(n_bars * sizeof(double));

        read_f64_array(f, high, n_bars);
        read_f64_array(f, low, n_bars);
        read_f64_array(f, close_arr, n_bars);
        read_f64_array(f, buy_price, n_bars);
        read_f64_array(f, sell_price, n_bars);
        read_f64_array(f, buy_amount, n_bars);
        read_f64_array(f, sell_amount, n_bars);

        double exp_total_return = read_f64(f);
        double exp_final_equity = read_f64(f);
        double exp_sortino = read_f64(f);
        double exp_max_drawdown = read_f64(f);
        double exp_margin_cost = read_f64(f);
        int exp_num_trades = read_i32(f);

        double *exp_equity_curve = (double *)malloc((n_bars + 1) * sizeof(double));
        read_f64_array(f, exp_equity_curve, n_bars + 1);

        SimConfig cfg = {
            .max_leverage = max_leverage,
            .can_short = can_short,
            .maker_fee = maker_fee,
            .margin_hourly_rate = margin_hourly_rate,
            .initial_cash = initial_cash,
            .fill_buffer_pct = fill_buffer_pct,
            .min_edge = min_edge,
            .max_hold_bars = max_hold_bars,
            .intensity_scale = intensity_scale,
        };

        SimResult result;
        double *equity_curve = (double *)malloc((n_bars + 1) * sizeof(double));

        simulate(NULL, high, low, close_arr,
                 buy_price, sell_price, buy_amount, sell_amount,
                 n_bars, &cfg, &result, equity_curve);

        fprintf(stderr, "case %2d n=%3d trades=%3d ret=%+.6f: ",
                ci, n_bars, result.num_trades, result.total_return);

        int case_fails = g_fail;

        CHECK_INT(result.num_trades, exp_num_trades, "num_trades");
        CHECK_NEAR(result.total_return, exp_total_return, TOL_TIGHT, "total_return");
        CHECK_NEAR(result.final_equity, exp_final_equity, TOL, "final_equity");
        CHECK_NEAR(result.margin_cost_total, exp_margin_cost, TOL, "margin_cost");
        CHECK_NEAR(result.sortino, exp_sortino, TOL, "sortino");
        CHECK_NEAR(result.max_drawdown, exp_max_drawdown, TOL_TIGHT, "max_drawdown");

        for (int j = 0; j <= n_bars; j++) {
            if (fabs(equity_curve[j] - exp_equity_curve[j]) > TOL) {
                fprintf(stderr, "  FAIL equity_curve[%d]: got %.6f expected %.6f\n",
                        j, equity_curve[j], exp_equity_curve[j]);
                g_fail++;
                break;
            }
        }
        if (g_fail == case_fails) {
            /* only count one pass for equity curve if no failures */
            g_pass++;
        }

        if (g_fail > case_fails) {
            fprintf(stderr, "  FAILED\n");
        } else {
            fprintf(stderr, "OK\n");
        }

        free(high); free(low); free(close_arr);
        free(buy_price); free(sell_price);
        free(buy_amount); free(sell_amount);
        free(exp_equity_curve); free(equity_curve);
    }

    fclose(f);
    fprintf(stderr, "\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
