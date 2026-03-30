#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "worksteal_sim.h"

#define MAX_SYM 256
#define IDX(sym, bar) ((sym) * n_bars + (bar))

typedef struct {
    int active;
    int direction; /* 0=long, 1=short */
    double entry_price;
    int entry_day;
    double quantity;
    double cost_basis;
    double peak_price;
    double target_exit;
    double stop_price;
    double margin_borrowed;
} CPosition;

typedef struct {
    int symbol;
    int direction;
    double score;
    double fill_price;
    double bar_high;
    double bar_low;
} Candidate;

static double rolling_max_high(const double *highs, const int *valid, int n_bars, int sym, int end, int lookback) {
    double mx = -1e30;
    int start = end - lookback + 1;
    if (start < 0) start = 0;
    for (int i = start; i <= end; i++) {
        if (valid[IDX(sym, i)]) {
            double h = highs[IDX(sym, i)];
            if (h > mx) mx = h;
        }
    }
    return mx > -1e29 ? mx : 0.0;
}

static double rolling_min_low(const double *lows, const int *valid, int n_bars, int sym, int end, int lookback) {
    double mn = 1e30;
    int start = end - lookback + 1;
    if (start < 0) start = 0;
    for (int i = start; i <= end; i++) {
        if (valid[IDX(sym, i)]) {
            double l = lows[IDX(sym, i)];
            if (l < mn) mn = l;
        }
    }
    return mn < 1e29 ? mn : 0.0;
}

static double compute_sma(const double *closes, const int *valid, int n_bars, int sym, int end, int period) {
    double sum = 0;
    int count = 0;
    int start = end - period + 1;
    if (start < 0) start = 0;
    for (int i = start; i <= end; i++) {
        if (valid[IDX(sym, i)]) {
            sum += closes[IDX(sym, i)];
            count++;
        }
    }
    return count > 0 ? sum / count : 0.0;
}

static int count_valid_bars(const int *valid, int n_bars, int sym, int end) {
    int c = 0;
    for (int i = 0; i <= end; i++) {
        if (valid[IDX(sym, i)]) c++;
    }
    return c;
}

static double compute_margin_interest(double margin_borrowed, int entry_day, int current_day, double annual_rate) {
    if (margin_borrowed <= 0) return 0.0;
    int days = current_day - entry_day;
    if (days < 1) days = 1;
    double daily = annual_rate / 365.0;
    return margin_borrowed * daily * days;
}

static double position_entry_fee(const CPosition *pos, double fee_rate) {
    return pos->quantity * pos->entry_price * fee_rate;
}

static double mark_to_market_position_value(
    const CPosition *pos,
    double price,
    int current_day,
    double annual_rate,
    double fee_rate
) {
    double interest = compute_margin_interest(pos->margin_borrowed, pos->entry_day, current_day, annual_rate);
    if (pos->direction == 0) {
        return pos->quantity * price - pos->margin_borrowed - interest;
    }
    return pos->cost_basis
        + pos->quantity * (pos->entry_price - price)
        - position_entry_fee(pos, fee_rate)
        - interest;
}

static void close_position_accounting(
    const CPosition *pos,
    double exit_price,
    int current_day,
    double annual_rate,
    double fee_rate,
    double *cash_delta,
    double *pnl
) {
    double interest = compute_margin_interest(pos->margin_borrowed, pos->entry_day, current_day, annual_rate);
    double exit_fee = pos->quantity * exit_price * fee_rate;
    if (pos->direction == 0) {
        double proceeds = pos->quantity * exit_price * (1.0 - fee_rate);
        *cash_delta = proceeds - pos->margin_borrowed - interest;
        *pnl = *cash_delta - pos->cost_basis;
        return;
    }
    *cash_delta = pos->cost_basis
        + pos->quantity * (pos->entry_price - exit_price)
        - position_entry_fee(pos, fee_rate)
        - exit_fee
        - interest;
    *pnl = *cash_delta - pos->cost_basis;
}

static int cmp_candidates(const void *a, const void *b) {
    const Candidate *ca = (const Candidate *)a;
    const Candidate *cb = (const Candidate *)b;
    if (cb->score > ca->score) return 1;
    if (cb->score < ca->score) return -1;
    return 0;
}

void worksteal_simulate(
    const double *timestamps,
    const int *valid_mask,
    const double *opens,
    const double *highs,
    const double *lows,
    const double *closes,
    const double *fee_rates,
    int n_bars,
    int n_symbols,
    const WorkStealConfig *cfg,
    SimResult *result,
    double *equity_curve
) {
    if (n_bars <= 0 || n_symbols <= 0) {
        memset(result, 0, sizeof(SimResult));
        return;
    }

    CPosition positions[MAX_SYM];
    int last_exit_day[MAX_SYM];
    memset(positions, 0, sizeof(positions));
    for (int s = 0; s < MAX_SYM; s++) last_exit_day[s] = -9999;

    double cash = cfg->initial_cash;
    int n_active = 0;
    int total_trades = 0;
    int n_wins = 0;
    int n_exits = 0;
    int eq_len = 0;
    double peak_equity = cfg->initial_cash;
    double max_dd_positive = 0.0; /* positive drawdown for early exit check */

    Candidate cands[MAX_SYM * 2];

    for (int d = 0; d < n_bars; d++) {
        /* compute inventory value for equity */
        double inv_value = 0.0;
        for (int s = 0; s < n_symbols; s++) {
            if (!positions[s].active) continue;
            if (!valid_mask[IDX(s, d)]) continue;
            double c = closes[IDX(s, d)];
            inv_value += mark_to_market_position_value(&positions[s], c, d, cfg->margin_annual_rate, fee_rates[s]);
        }
        (void)(cash + inv_value);

        /* 1. Check exits */
        for (int s = 0; s < n_symbols; s++) {
            if (!positions[s].active) continue;
            if (!valid_mask[IDX(s, d)]) continue;

            CPosition *pos = &positions[s];
            double c = closes[IDX(s, d)];
            double h = highs[IDX(s, d)];
            double l = lows[IDX(s, d)];

            double exit_price = 0;
            int do_exit = 0;
            double pnl = 0;

            if (pos->direction == 0) { /* long */
                if (h > pos->peak_price) pos->peak_price = h;

                if (h >= pos->target_exit) {
                    exit_price = pos->target_exit;
                    do_exit = 1;
                } else if (l <= pos->stop_price) {
                    exit_price = pos->stop_price;
                    do_exit = 1;
                } else if (cfg->trailing_stop_pct > 0) {
                    double trail = pos->peak_price * (1.0 - cfg->trailing_stop_pct);
                    if (l <= trail) {
                        exit_price = trail;
                        do_exit = 1;
                    }
                }
            } else { /* short */
                if (l < pos->peak_price) pos->peak_price = l;

                if (l <= pos->target_exit) {
                    exit_price = pos->target_exit;
                    do_exit = 1;
                } else if (h >= pos->stop_price) {
                    exit_price = pos->stop_price;
                    do_exit = 1;
                } else if (cfg->trailing_stop_pct > 0) {
                    double trail = pos->peak_price * (1.0 + cfg->trailing_stop_pct);
                    if (h >= trail) {
                        exit_price = trail;
                        do_exit = 1;
                    }
                }
                /* margin call */
                if (!do_exit) {
                    double unrealized = pos->quantity * (c - pos->entry_price);
                    if (unrealized > pos->cost_basis * 0.8) {
                        exit_price = c;
                        do_exit = 1;
                    }
                }
            }

            /* max hold */
            if (!do_exit && cfg->max_hold_days > 0) {
                int held = d - pos->entry_day;
                if (held >= cfg->max_hold_days) {
                    exit_price = c;
                    do_exit = 1;
                }
            }

            if (do_exit) {
                double fee_rate = fee_rates[s];
                double cash_delta = 0.0;
                close_position_accounting(
                    pos,
                    exit_price,
                    d,
                    cfg->margin_annual_rate,
                    fee_rate,
                    &cash_delta,
                    &pnl
                );
                cash += cash_delta;

                total_trades++;
                n_exits++;
                if (pnl > 0) n_wins++;
                last_exit_day[s] = d;
                pos->active = 0;
                n_active--;
            }
        }

        /* 2. Check entries (work-stealing) */
        if (n_active < cfg->max_positions) {
            int nc = 0;

            for (int s = 0; s < n_symbols; s++) {
                if (positions[s].active) continue;
                if (!valid_mask[IDX(s, d)]) continue;
                int n_valid = count_valid_bars(valid_mask, n_bars, s, d);
                if (n_valid < cfg->lookback_days) continue;
                if (cfg->reentry_cooldown_days > 0 && (d - last_exit_day[s]) < cfg->reentry_cooldown_days) continue;

                double c = closes[IDX(s, d)];
                double h_bar = highs[IDX(s, d)];
                double l_bar = lows[IDX(s, d)];

                /* momentum filter */
                if (cfg->momentum_period > 0 && n_valid > cfg->momentum_period) {
                    /* find the bar momentum_period bars back */
                    int cnt = 0;
                    double past_close = c;
                    for (int j = d; j >= 0; j--) {
                        if (valid_mask[IDX(s, j)]) {
                            cnt++;
                            if (cnt == cfg->momentum_period + 1) {
                                past_close = closes[IDX(s, j)];
                                break;
                            }
                        }
                    }
                    if (past_close > 0) {
                        double mom = (c - past_close) / past_close;
                        if (mom < cfg->momentum_min) continue;
                    }
                }

                /* SMA filter */
                if (cfg->sma_filter_period > 0) {
                    double sma = compute_sma(closes, valid_mask, n_bars, s, d, cfg->sma_filter_period);
                    if (c < sma) continue;
                }

                /* long candidate */
                double ref_high = rolling_max_high(highs, valid_mask, n_bars, s, d, cfg->lookback_days);
                if (ref_high <= 0) continue;
                double buy_target = ref_high * (1.0 - cfg->dip_pct);
                double dist_long = (c - buy_target) / ref_high;

                if (dist_long <= cfg->proximity_pct) {
                    double fill = buy_target;
                    if (fill < l_bar) fill = l_bar;
                    cands[nc].symbol = s;
                    cands[nc].direction = 0;
                    cands[nc].score = -dist_long;
                    cands[nc].fill_price = fill;
                    cands[nc].bar_high = h_bar;
                    cands[nc].bar_low = l_bar;
                    nc++;
                }

                /* short candidate */
                if (cfg->enable_shorts) {
                    double ref_low = rolling_min_low(lows, valid_mask, n_bars, s, d, cfg->lookback_days);
                    if (ref_low <= 0) continue;
                    double short_target = ref_low * (1.0 + cfg->short_pump_pct);
                    double dist_short = (short_target - c) / ref_low;
                    if (dist_short <= cfg->proximity_pct) {
                        double fill = short_target;
                        if (fill > h_bar) fill = h_bar;
                        cands[nc].symbol = s;
                        cands[nc].direction = 1;
                        cands[nc].score = -dist_short;
                        cands[nc].fill_price = fill;
                        cands[nc].bar_high = h_bar;
                        cands[nc].bar_low = l_bar;
                        nc++;
                    }
                }
            }

            qsort(cands, nc, sizeof(Candidate), cmp_candidates);

            int slots = cfg->max_positions - n_active;
            double base_eq = cfg->initial_cash;
            for (int k = 0; k < nc && slots > 0; k++) {
                int s = cands[k].symbol;
                if (positions[s].active) continue;

                double fee_rate = fee_rates[s];
                double max_alloc = base_eq * cfg->max_position_pct * cfg->max_leverage;
                double fill = cands[k].fill_price;

                if (cands[k].direction == 0) { /* long */
                    double alloc = max_alloc;
                    double qty = alloc / (fill * (1.0 + fee_rate));
                    if (qty <= 0) continue;

                    double actual_cost = qty * fill * (1.0 + fee_rate);
                    double available_cash = cash;
                    double borrowed = actual_cost - available_cash;
                    if (borrowed < 0) borrowed = 0;
                    double equity_used = actual_cost - borrowed;
                    cash -= equity_used;

                    positions[s].active = 1;
                    positions[s].direction = 0;
                    positions[s].entry_price = fill;
                    positions[s].entry_day = d;
                    positions[s].quantity = qty;
                    positions[s].cost_basis = equity_used;
                    positions[s].peak_price = cands[k].bar_high;
                    positions[s].target_exit = fill * (1.0 + cfg->profit_target_pct);
                    positions[s].stop_price = fill * (1.0 - cfg->stop_loss_pct);
                    positions[s].margin_borrowed = borrowed;
                    n_active++;
                    slots--;
                    total_trades++;
                } else { /* short */
                    double alloc_limit = base_eq * cfg->max_position_pct;
                    double alloc = max_alloc < alloc_limit ? max_alloc : alloc_limit;
                    double qty = alloc / fill;
                    if (qty <= 0) continue;
                    double margin_required = alloc * 0.5;
                    if (cash < margin_required) continue;

                    cash -= margin_required;
                    double borrowed = qty * fill;

                    positions[s].active = 1;
                    positions[s].direction = 1;
                    positions[s].entry_price = fill;
                    positions[s].entry_day = d;
                    positions[s].quantity = qty;
                    positions[s].cost_basis = margin_required;
                    positions[s].peak_price = cands[k].bar_low;
                    positions[s].target_exit = fill * (1.0 - cfg->profit_target_pct);
                    positions[s].stop_price = fill * (1.0 + cfg->stop_loss_pct);
                    positions[s].margin_borrowed = borrowed;
                    n_active++;
                    slots--;
                    total_trades++;
                }
            }
        }

        /* 3. Compute equity */
        double inv = 0.0;
        for (int s = 0; s < n_symbols; s++) {
            if (!positions[s].active) continue;
            if (!valid_mask[IDX(s, d)]) {
                inv += mark_to_market_position_value(
                    &positions[s],
                    positions[s].entry_price,
                    d,
                    cfg->margin_annual_rate,
                    fee_rates[s]
                );
                continue;
            }
            double c = closes[IDX(s, d)];
            inv += mark_to_market_position_value(
                &positions[s],
                c,
                d,
                cfg->margin_annual_rate,
                fee_rates[s]
            );
        }
        double equity = cash + inv;
        equity_curve[eq_len++] = equity;

        if (equity > peak_equity) peak_equity = equity;

        /* track positive drawdown for early exit */
        if (peak_equity > 0) {
            double dd_pos = (peak_equity - equity) / peak_equity;
            if (dd_pos > max_dd_positive) max_dd_positive = dd_pos;
        }

        /* drawdown-vs-profit early exit (matches Python evaluate_drawdown_vs_profit_early_exit) */
        if (n_bars >= 20 && eq_len >= 2) {
            double progress = (double)eq_len / (double)n_bars;
            if (progress >= 0.5) {
                double total_ret = (equity - cfg->initial_cash) / cfg->initial_cash;
                if (max_dd_positive > total_ret) {
                    /* force close all positions */
                    for (int s = 0; s < n_symbols; s++) {
                        if (!positions[s].active) continue;
                        double cp;
                        if (valid_mask[IDX(s, d)])
                            cp = closes[IDX(s, d)];
                        else
                            cp = positions[s].entry_price;
                        double fr = fee_rates[s];
                        double cash_delta = 0.0;
                        double pnl = 0.0;
                        close_position_accounting(
                            &positions[s],
                            cp,
                            d,
                            cfg->margin_annual_rate,
                            fr,
                            &cash_delta,
                            &pnl
                        );
                        cash += cash_delta;
                        total_trades++;
                        n_exits++;
                        if (pnl > 0) n_wins++;
                        positions[s].active = 0;
                    }
                    n_active = 0;
                    equity_curve[eq_len - 1] = cash;
                    break;
                }
            }
        }

        /* max drawdown early exit */
        if (cfg->max_drawdown_exit > 0 && eq_len > 1 && peak_equity > 0) {
            double dd = (equity - peak_equity) / peak_equity;
            if (dd < -cfg->max_drawdown_exit) {
                /* force close all */
                for (int s = 0; s < n_symbols; s++) {
                    if (!positions[s].active) continue;
                    double cp;
                    if (valid_mask[IDX(s, d)])
                        cp = closes[IDX(s, d)];
                    else
                        cp = positions[s].entry_price;
                    double fr = fee_rates[s];
                    double cash_delta = 0.0;
                    double pnl = 0.0;
                    close_position_accounting(
                        &positions[s],
                        cp,
                        d,
                        cfg->margin_annual_rate,
                        fr,
                        &cash_delta,
                        &pnl
                    );
                    cash += cash_delta;
                    total_trades++;
                    n_exits++;
                    if (pnl > 0) n_wins++;
                    positions[s].active = 0;
                }
                n_active = 0;
                equity_curve[eq_len - 1] = cash;
                break;
            }
        }
    }

    /* compute metrics */
    result->n_days = eq_len;
    result->total_trades = total_trades;

    if (eq_len < 2) {
        result->total_return = 0;
        result->sortino = 0;
        result->sharpe = 0;
        result->max_drawdown = 0;
        result->win_rate = 0;
        result->final_equity = cfg->initial_cash;
        result->mean_daily_return = 0;
        return;
    }

    double *rets = (double *)malloc((eq_len - 1) * sizeof(double));
    int n_ret = eq_len - 1;
    double sum_ret = 0;
    for (int i = 0; i < n_ret; i++) {
        double denom = equity_curve[i];
        if (denom < 1e-8) denom = 1e-8;
        rets[i] = (equity_curve[i + 1] - equity_curve[i]) / denom;
        sum_ret += rets[i];
    }

    double mean_ret = sum_ret / n_ret;
    result->mean_daily_return = mean_ret;

    /* downside std */
    double sum_neg = 0, sum_neg_sq = 0;
    int n_neg = 0;
    for (int i = 0; i < n_ret; i++) {
        if (rets[i] < 0) {
            sum_neg += rets[i];
            sum_neg_sq += rets[i] * rets[i];
            n_neg++;
        }
    }
    double downside_std;
    if (n_neg > 1) {
        double mean_neg = sum_neg / n_neg;
        double var_neg = sum_neg_sq / n_neg - mean_neg * mean_neg;
        if (var_neg < 0) var_neg = 0;
        downside_std = sqrt(var_neg);
    } else {
        downside_std = 1e-8;
    }

    /* full std for sharpe */
    double sum_sq = 0;
    for (int i = 0; i < n_ret; i++) {
        double d = rets[i] - mean_ret;
        sum_sq += d * d;
    }
    double std_ret = n_ret > 1 ? sqrt(sum_sq / n_ret) : 1e-8;

    result->sortino = mean_ret / fmax(downside_std, 1e-8) * sqrt(365.0);
    result->sharpe = mean_ret / fmax(std_ret, 1e-8) * sqrt(365.0);

    result->total_return = (equity_curve[eq_len - 1] - equity_curve[0]) / equity_curve[0];
    result->final_equity = equity_curve[eq_len - 1];

    /* max drawdown */
    double rmax = equity_curve[0];
    double mdd = 0;
    for (int i = 0; i < eq_len; i++) {
        if (equity_curve[i] > rmax) rmax = equity_curve[i];
        double dd = (equity_curve[i] - rmax) / rmax;
        if (dd < mdd) mdd = dd;
    }
    result->max_drawdown = mdd;

    result->win_rate = n_exits > 0 ? (double)n_wins / n_exits * 100.0 : 0.0;

    free(rets);
}

void worksteal_simulate_batch(
    const double *timestamps,
    const int *valid_mask,
    const double *opens,
    const double *highs,
    const double *lows,
    const double *closes,
    const double *fee_rates,
    int n_bars,
    int n_symbols,
    const WorkStealConfig *configs,
    SimResult *results,
    int n_configs
) {
    double *eq = (double *)malloc(n_bars * sizeof(double));
    for (int i = 0; i < n_configs; i++) {
        worksteal_simulate(timestamps, valid_mask, opens, highs, lows, closes,
                           fee_rates, n_bars, n_symbols, &configs[i], &results[i], eq);
    }
    free(eq);
}
