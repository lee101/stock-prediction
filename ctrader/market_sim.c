#include "market_sim.h"
#include <stdlib.h>
#include <string.h>

static double clamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

double compute_sortino(const double *equity_curve, int n_eq) {
    if (n_eq < 2) return 0.0;
    int n_bars = n_eq - 1;
    double sum_ret = 0.0, sum_neg = 0.0, sum_neg_sq = 0.0;
    int n_neg = 0;

    for (int i = 0; i < n_bars; i++) {
        double denom = fabs(equity_curve[i]) + 1e-10;
        double r = (equity_curve[i + 1] - equity_curve[i]) / denom;
        sum_ret += r;
        if (r < 0.0) {
            sum_neg += r;
            sum_neg_sq += r * r;
            n_neg++;
        }
    }

    double mean_ret = sum_ret / n_bars;
    double mean_neg = n_neg > 0 ? sum_neg / n_neg : 0.0;
    double var_neg = n_neg > 0 ? (sum_neg_sq / n_neg - mean_neg * mean_neg) : 0.0;
    if (var_neg < 0.0) var_neg = 0.0;
    double std_neg = n_neg > 0 ? sqrt(var_neg) : 1e-10;
    return (mean_ret / (std_neg + 1e-10)) * sqrt(ANNUALIZE_FACTOR);
}

double compute_max_drawdown(const double *equity_curve, int n_eq) {
    double running_max = equity_curve[0];
    double max_dd = 0.0;
    for (int i = 0; i < n_eq; i++) {
        if (equity_curve[i] > running_max) running_max = equity_curve[i];
        double dd = (equity_curve[i] - running_max) / (running_max + 1e-10);
        if (dd < max_dd) max_dd = dd;
    }
    return max_dd;
}

void simulate(
    const double *open, const double *high, const double *low, const double *close,
    const double *buy_price, const double *sell_price,
    const double *buy_amount, const double *sell_amount,
    int n_bars,
    const SimConfig *cfg,
    SimResult *result,
    double *equity_curve
) {
    double cash = cfg->initial_cash;
    double inventory = 0.0;
    double margin_cost_total = 0.0;
    int bars_in_position = 0;
    int num_trades = 0;
    double iscale = cfg->intensity_scale;
    double fill_buf = cfg->fill_buffer_pct;

    (void)open;
    equity_curve[0] = cash;

    for (int i = 0; i < n_bars; i++) {
        double c = close[i];
        double h = high[i];
        double l = low[i];
        double bp = buy_price[i];
        double sp = sell_price[i];
        double ba = clamp(buy_amount[i] * iscale, 0.0, 100.0) / 100.0;
        double sa = clamp(sell_amount[i] * iscale, 0.0, 100.0) / 100.0;

        double equity = cash + inventory * c;

        if (cash < 0.0) {
            double interest = (-cash) * cfg->margin_hourly_rate;
            cash -= interest;
            margin_cost_total += interest;
        }
        if (inventory < 0.0) {
            double bv = (-inventory) * c;
            double interest = bv * cfg->margin_hourly_rate;
            cash -= interest;
            margin_cost_total += interest;
        }

        if (cfg->max_hold_bars > 0 && inventory > 0.0 && bars_in_position >= cfg->max_hold_bars) {
            double fp = c * 0.999;
            cash += inventory * fp * (1.0 - cfg->maker_fee);
            num_trades++;
            inventory = 0.0;
            bars_in_position = 0;
            equity_curve[i + 1] = cash;
            continue;
        }

        double edge = (bp > 0.0 && sp > 0.0) ? (sp - bp) / bp : 0.0;
        if (cfg->min_edge > 0.0 && edge < cfg->min_edge) {
            if (inventory > 0.0) bars_in_position++;
            equity_curve[i + 1] = cash + inventory * c;
            continue;
        }

        int sold_this_bar = 0;
        if (sa > 0.0 && sp > 0.0 && h >= sp * (1.0 + fill_buf)) {
            double sell_qty = 0.0;
            if (inventory > 0.0) {
                sell_qty = sa * inventory;
                if (sell_qty > inventory) sell_qty = inventory;
            } else if (cfg->can_short) {
                double max_sv = cfg->max_leverage * (equity > 0.0 ? equity : 0.0);
                double q1 = sa * max_sv / (sp * (1.0 + cfg->maker_fee));
                double q2 = max_sv / (sp * (1.0 + cfg->maker_fee));
                sell_qty = q1 < q2 ? q1 : q2;
            }
            if (sell_qty < 0.0) sell_qty = 0.0;
            if (sell_qty > 0.0) {
                cash += sell_qty * sp * (1.0 - cfg->maker_fee);
                inventory -= sell_qty;
                num_trades++;
                sold_this_bar = 1;
                if (inventory <= 0.0) bars_in_position = 0;
            }
        }

        if (!sold_this_bar && ba > 0.0 && bp > 0.0 && l <= bp * (1.0 - fill_buf)) {
            double max_bv = cfg->max_leverage * (equity > 0.0 ? equity : 0.0) - inventory * bp;
            if (max_bv > 0.0) {
                double buy_qty = ba * max_bv / (bp * (1.0 + cfg->maker_fee));
                if (buy_qty > 0.0) {
                    cash -= buy_qty * bp * (1.0 + cfg->maker_fee);
                    inventory += buy_qty;
                    num_trades++;
                }
            }
        }

        if (inventory > 0.0) bars_in_position++;
        else bars_in_position = 0;

        equity_curve[i + 1] = cash + inventory * c;
    }

    if (n_bars > 0 && inventory != 0.0) {
        double lc = close[n_bars - 1];
        if (inventory > 0.0)
            cash += inventory * lc * (1.0 - cfg->maker_fee);
        else
            cash -= (-inventory) * lc * (1.0 + cfg->maker_fee);
        inventory = 0.0;
        equity_curve[n_bars] = cash;
    }

    int n_eq = n_bars + 1;
    result->sortino = compute_sortino(equity_curve, n_eq);
    result->max_drawdown = compute_max_drawdown(equity_curve, n_eq);
    result->total_return = n_bars > 0 ? (equity_curve[n_bars] / equity_curve[0] - 1.0) : 0.0;
    result->final_equity = equity_curve[n_bars > 0 ? n_bars : 0];
    result->num_trades = num_trades;
    result->margin_cost_total = margin_cost_total;
}

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

/* multi-symbol portfolio simulation */

static inline int idx(int bar, int sym, int n_sym) {
    return bar * n_sym + sym;
}

void simulate_multi(
    int n_bars,
    int n_symbols,
    const double *close,
    const double *high,
    const double *low,
    const double *buy_price,
    const double *sell_price,
    const double *buy_amount,
    const double *sell_amount,
    const MultiSimConfig *cfg,
    MultiSimResult *result,
    double *equity_curve
) {
    if (n_symbols > MAX_SYMBOLS) return;

    Position positions[MAX_SYMBOLS];
    memset(positions, 0, sizeof(positions));
    memset(result, 0, sizeof(*result));

    double cash = cfg->initial_cash;
    int active_count = 0;
    double iscale = cfg->intensity_scale;
    double fill_buf = cfg->fill_buffer_pct;
    int num_trades = 0;
    double margin_cost_total = 0.0;

    equity_curve[0] = cash;

    for (int t = 0; t < n_bars; t++) {
        /* compute portfolio equity */
        double pos_value = 0.0;
        for (int s = 0; s < n_symbols; s++) {
            if (positions[s].qty != 0.0) {
                pos_value += positions[s].qty * close[idx(t, s, n_symbols)];
            }
        }
        double equity = cash + pos_value;

        /* margin interest on borrowed cash */
        if (cash < 0.0) {
            double interest = (-cash) * cfg->margin_hourly_rate;
            cash -= interest;
            margin_cost_total += interest;
        }
        /* margin interest on short positions */
        for (int s = 0; s < n_symbols; s++) {
            if (positions[s].qty < 0.0) {
                double bv = (-positions[s].qty) * close[idx(t, s, n_symbols)];
                double interest = bv * cfg->margin_hourly_rate;
                cash -= interest;
                margin_cost_total += interest;
            }
        }

        /* force-close positions exceeding max hold */
        if (cfg->max_hold_bars > 0) {
            for (int s = 0; s < n_symbols; s++) {
                if (positions[s].qty == 0.0) continue;
                if (positions[s].bars_held >= cfg->max_hold_bars) {
                    double c = close[idx(t, s, n_symbols)];
                    double fee = cfg->sym_cfgs[s].fee_rate;
                    double slip = cfg->force_close_slippage;
                    if (positions[s].qty > 0.0) {
                        double fp = c * (1.0 - slip);
                        cash += positions[s].qty * fp * (1.0 - fee);
                    } else {
                        double fp = c * (1.0 + slip);
                        cash -= (-positions[s].qty) * fp * (1.0 + fee);
                    }
                    num_trades++;
                    result->trades_per_symbol[s]++;
                    positions[s].qty = 0.0;
                    positions[s].bars_held = 0;
                    active_count--;
                }
            }
        }

        /* process sells/exits first */
        for (int s = 0; s < n_symbols; s++) {
            if (positions[s].qty <= 0.0) continue;

            double sp_val = sell_price[idx(t, s, n_symbols)];
            double sa_val = clamp(sell_amount[idx(t, s, n_symbols)] * iscale, 0.0, 100.0) / 100.0;
            double h_val = high[idx(t, s, n_symbols)];
            double fee = cfg->sym_cfgs[s].fee_rate;

            if (sa_val > 0.0 && sp_val > 0.0 && h_val >= sp_val * (1.0 + fill_buf)) {
                double sell_qty = sa_val * positions[s].qty;
                if (sell_qty > positions[s].qty) sell_qty = positions[s].qty;
                cash += sell_qty * sp_val * (1.0 - fee);
                positions[s].qty -= sell_qty;
                num_trades++;
                result->trades_per_symbol[s]++;
                if (positions[s].qty <= 1e-12) {
                    positions[s].qty = 0.0;
                    positions[s].bars_held = 0;
                    active_count--;
                }
            }
        }

        /* process short sells for short-allowed symbols */
        for (int s = 0; s < n_symbols; s++) {
            if (!cfg->sym_cfgs[s].can_short) continue;
            if (positions[s].qty != 0.0) continue;
            if (active_count >= cfg->max_positions) break;

            double sp_val = sell_price[idx(t, s, n_symbols)];
            double sa_val = clamp(sell_amount[idx(t, s, n_symbols)] * iscale, 0.0, 100.0) / 100.0;
            double h_val = high[idx(t, s, n_symbols)];
            double fee = cfg->sym_cfgs[s].fee_rate;

            if (sa_val > 0.0 && sp_val > 0.0 && h_val >= sp_val * (1.0 + fill_buf)) {
                double alloc = equity > 0.0 ? equity / cfg->max_positions : 0.0;
                double max_sv = cfg->max_leverage * alloc;
                double sell_qty = sa_val * max_sv / (sp_val * (1.0 + fee));
                if (sell_qty > 0.0) {
                    cash += sell_qty * sp_val * (1.0 - fee);
                    positions[s].qty = -sell_qty;
                    positions[s].entry_price = sp_val;
                    positions[s].bars_held = 0;
                    num_trades++;
                    result->trades_per_symbol[s]++;
                    active_count++;
                }
            }
        }

        /* process short covers */
        for (int s = 0; s < n_symbols; s++) {
            if (positions[s].qty >= 0.0) continue;

            double bp_val = buy_price[idx(t, s, n_symbols)];
            double ba_val = clamp(buy_amount[idx(t, s, n_symbols)] * iscale, 0.0, 100.0) / 100.0;
            double l_val = low[idx(t, s, n_symbols)];
            double fee = cfg->sym_cfgs[s].fee_rate;

            if (ba_val > 0.0 && bp_val > 0.0 && l_val <= bp_val * (1.0 - fill_buf)) {
                double cover_qty = ba_val * (-positions[s].qty);
                if (cover_qty > (-positions[s].qty)) cover_qty = -positions[s].qty;
                cash -= cover_qty * bp_val * (1.0 + fee);
                positions[s].qty += cover_qty;
                num_trades++;
                result->trades_per_symbol[s]++;
                if (positions[s].qty >= -1e-12) {
                    positions[s].qty = 0.0;
                    positions[s].bars_held = 0;
                    active_count--;
                }
            }
        }

        /* process buys */
        for (int s = 0; s < n_symbols; s++) {
            if (!cfg->sym_cfgs[s].can_long) continue;
            if (positions[s].qty != 0.0) continue;
            if (active_count >= cfg->max_positions) break;

            double bp_val = buy_price[idx(t, s, n_symbols)];
            double ba_val = clamp(buy_amount[idx(t, s, n_symbols)] * iscale, 0.0, 100.0) / 100.0;
            double l_val = low[idx(t, s, n_symbols)];
            double fee = cfg->sym_cfgs[s].fee_rate;

            if (ba_val > 0.0 && bp_val > 0.0 && l_val <= bp_val * (1.0 - fill_buf)) {
                double alloc = equity > 0.0 ? equity / cfg->max_positions : 0.0;
                double max_bv = cfg->max_leverage * alloc;
                double buy_qty = ba_val * max_bv / (bp_val * (1.0 + fee));
                if (buy_qty > 0.0) {
                    cash -= buy_qty * bp_val * (1.0 + fee);
                    positions[s].qty = buy_qty;
                    positions[s].entry_price = bp_val;
                    positions[s].bars_held = 0;
                    num_trades++;
                    result->trades_per_symbol[s]++;
                    active_count++;
                }
            }
        }

        /* increment hold counters */
        for (int s = 0; s < n_symbols; s++) {
            if (positions[s].qty != 0.0)
                positions[s].bars_held++;
        }

        /* record equity */
        pos_value = 0.0;
        for (int s = 0; s < n_symbols; s++) {
            if (positions[s].qty != 0.0)
                pos_value += positions[s].qty * close[idx(t, s, n_symbols)];
        }
        equity_curve[t + 1] = cash + pos_value;
    }

    /* close remaining positions at last close */
    if (n_bars > 0) {
        for (int s = 0; s < n_symbols; s++) {
            if (positions[s].qty == 0.0) continue;
            double lc = close[idx(n_bars - 1, s, n_symbols)];
            double fee = cfg->sym_cfgs[s].fee_rate;
            if (positions[s].qty > 0.0)
                cash += positions[s].qty * lc * (1.0 - fee);
            else
                cash -= (-positions[s].qty) * lc * (1.0 + fee);
            positions[s].qty = 0.0;
        }
        equity_curve[n_bars] = cash;
    }

    int n_eq = n_bars + 1;
    result->sortino = compute_sortino(equity_curve, n_eq);
    result->max_drawdown = compute_max_drawdown(equity_curve, n_eq);
    result->total_return = n_bars > 0 ? (equity_curve[n_bars] / equity_curve[0] - 1.0) : 0.0;
    result->final_equity = equity_curve[n_bars > 0 ? n_bars : 0];
    result->num_trades = num_trades;
    result->margin_cost_total = margin_cost_total;
}
