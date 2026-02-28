#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double max_leverage;
    int can_short;
    double maker_fee;
    double margin_hourly_rate;
    double initial_cash;
    double fill_buffer_pct;
    double min_edge;
    int max_hold_bars;
    double intensity_scale;
} SimConfig;

typedef struct {
    double total_return;
    double sortino;
    double max_drawdown;
    double final_equity;
    int num_trades;
    double margin_cost_total;
} SimResult;

void simulate(
    const double *open, const double *high, const double *low, const double *close,
    const double *buy_price, const double *sell_price,
    const double *buy_amount, const double *sell_amount,
    int n_bars,
    const SimConfig *cfg,
    SimResult *result,
    double *equity_curve  /* caller-allocated, size n_bars+1 */
) {
    double cash = cfg->initial_cash;
    double inventory = 0.0;
    double margin_cost_total = 0.0;
    int bars_in_position = 0;
    int num_trades = 0;
    double iscale = cfg->intensity_scale;
    double fill_buf = cfg->fill_buffer_pct;

    equity_curve[0] = cash;

    for (int i = 0; i < n_bars; i++) {
        double c = close[i];
        double h = high[i];
        double l = low[i];
        double bp = buy_price[i];
        double sp = sell_price[i];
        double ba = buy_amount[i] * iscale;
        double sa = sell_amount[i] * iscale;
        if (ba > 100.0) ba = 100.0;
        if (sa > 100.0) sa = 100.0;
        ba /= 100.0;
        sa /= 100.0;

        double equity = cash + inventory * c;

        // margin interest
        if (cash < 0) {
            double interest = (-cash) * cfg->margin_hourly_rate;
            cash -= interest;
            margin_cost_total += interest;
        }
        if (inventory < 0) {
            double bv = (-inventory) * c;
            double interest = bv * cfg->margin_hourly_rate;
            cash -= interest;
            margin_cost_total += interest;
        }

        // force close
        if (cfg->max_hold_bars > 0 && inventory > 0 && bars_in_position >= cfg->max_hold_bars) {
            double fp = c * 0.999;
            cash += inventory * fp * (1.0 - cfg->maker_fee);
            num_trades++;
            inventory = 0.0;
            bars_in_position = 0;
            equity_curve[i + 1] = cash;
            continue;
        }

        // min edge
        double edge = (bp > 0 && sp > 0) ? (sp - bp) / bp : 0.0;
        if (cfg->min_edge > 0 && edge < cfg->min_edge) {
            if (inventory > 0) bars_in_position++;
            equity_curve[i + 1] = cash + inventory * c;
            continue;
        }

        // sell first
        int sold_this_bar = 0;
        if (sa > 0 && sp > 0 && h >= sp * (1.0 + fill_buf)) {
            double sell_qty = 0;
            if (inventory > 0) {
                sell_qty = sa * inventory;
                if (sell_qty > inventory) sell_qty = inventory;
            } else if (cfg->can_short) {
                double max_sv = cfg->max_leverage * (equity > 0 ? equity : 0);
                double q1 = sa * max_sv / (sp * (1.0 + cfg->maker_fee));
                double q2 = max_sv / (sp * (1.0 + cfg->maker_fee));
                sell_qty = q1 < q2 ? q1 : q2;
            }
            if (sell_qty < 0) sell_qty = 0;
            if (sell_qty > 0) {
                cash += sell_qty * sp * (1.0 - cfg->maker_fee);
                inventory -= sell_qty;
                num_trades++;
                sold_this_bar = 1;
                if (inventory <= 0) bars_in_position = 0;
            }
        }

        // buy (no same-bar roundtrip)
        if (!sold_this_bar && ba > 0 && bp > 0 && l <= bp * (1.0 - fill_buf)) {
            double max_bv = cfg->max_leverage * (equity > 0 ? equity : 0) - inventory * bp;
            if (max_bv > 0) {
                double buy_qty = ba * max_bv / (bp * (1.0 + cfg->maker_fee));
                if (buy_qty > 0) {
                    cash -= buy_qty * bp * (1.0 + cfg->maker_fee);
                    inventory += buy_qty;
                    num_trades++;
                }
            }
        }

        if (inventory > 0) bars_in_position++;
        else bars_in_position = 0;

        equity_curve[i + 1] = cash + inventory * c;
    }

    // close remaining position
    if (n_bars > 0 && inventory != 0) {
        double lc = close[n_bars - 1];
        if (inventory > 0)
            cash += inventory * lc * (1.0 - cfg->maker_fee);
        else
            cash -= (-inventory) * lc * (1.0 + cfg->maker_fee);
        inventory = 0;
        equity_curve[n_bars] = cash;
    }

    // compute metrics
    int n_eq = n_bars + 1;
    double *ret = (double *)malloc(n_bars * sizeof(double));
    int n_neg = 0;
    double sum_ret = 0, sum_neg_sq = 0;

    double sum_neg = 0;
    for (int i = 0; i < n_bars; i++) {
        double denom = fabs(equity_curve[i]) + 1e-10;
        ret[i] = (equity_curve[i + 1] - equity_curve[i]) / denom;
        sum_ret += ret[i];
        if (ret[i] < 0) {
            sum_neg += ret[i];
            sum_neg_sq += ret[i] * ret[i];
            n_neg++;
        }
    }

    double mean_ret = n_bars > 0 ? sum_ret / n_bars : 0;
    double mean_neg = n_neg > 0 ? sum_neg / n_neg : 0;
    double var_neg = n_neg > 0 ? (sum_neg_sq / n_neg - mean_neg * mean_neg) : 0;
    if (var_neg < 0) var_neg = 0;
    double std_neg = n_neg > 0 ? sqrt(var_neg) : 1e-10;
    double sortino = (mean_ret / (std_neg + 1e-10)) * sqrt(8760.0);

    // max drawdown
    double running_max = equity_curve[0];
    double max_dd = 0;
    for (int i = 0; i < n_eq; i++) {
        if (equity_curve[i] > running_max) running_max = equity_curve[i];
        double dd = (equity_curve[i] - running_max) / (running_max + 1e-10);
        if (dd < max_dd) max_dd = dd;
    }

    result->total_return = n_bars > 0 ? (equity_curve[n_bars] / equity_curve[0] - 1.0) : 0;
    result->sortino = sortino;
    result->max_drawdown = max_dd;
    result->final_equity = equity_curve[n_bars > 0 ? n_bars : 0];
    result->num_trades = num_trades;
    result->margin_cost_total = margin_cost_total;

    free(ret);
}

/* batch: run N simulations with different configs on same data */
void simulate_batch(
    const double *open, const double *high, const double *low, const double *close,
    const double *buy_price, const double *sell_price,
    const double *buy_amount, const double *sell_amount,
    int n_bars,
    const SimConfig *configs,
    SimResult *results,
    int n_configs
) {
    double *eq = (double *)malloc((n_bars + 1) * sizeof(double));
    for (int i = 0; i < n_configs; i++) {
        simulate(open, high, low, close, buy_price, sell_price,
                 buy_amount, sell_amount, n_bars, &configs[i], &results[i], eq);
    }
    free(eq);
}
