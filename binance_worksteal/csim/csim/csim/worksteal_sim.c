#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "worksteal_sim.h"

#define MAX_SYM 64
#define IDX(sym, bar) ((sym) * n_bars + (bar))

typedef struct {
    int active;
    int direction;
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

static inline double rolling_max_high(
    const double * restrict highs,
    const int * restrict valid,
    int n_bars, int sym, int end, int lookback
) {
    double mx = -1e30;
    int start = end - lookback + 1;
    if (start < 0) start = 0;
    const int base = sym * n_bars;
    for (int i = start; i <= end; i++) {
        if (valid[base + i]) {
            double h = highs[base + i];
            if (h > mx) mx = h;
        }
    }
    return mx > -1e29 ? mx : 0.0;
}

static inline double rolling_min_low(
    const double * restrict lows,
    const int * restrict valid,
    int n_bars, int sym, int end, int lookback
) {
    double mn = 1e30;
    int start = end - lookback + 1;
    if (start < 0) start = 0;
    const int base = sym * n_bars;
    for (int i = start; i <= end; i++) {
        if (valid[base + i]) {
            double l = lows[base + i];
            if (l < mn) mn = l;
        }
    }
    return mn < 1e29 ? mn : 0.0;
}

static inline double compute_sma(
    const double * restrict closes,
    const int * restrict valid,
    int n_bars, int sym, int end, int period
) {
    double sum = 0;
    int count = 0;
    int start = end - period + 1;
    if (start < 0) start = 0;
    const int base = sym * n_bars;
    for (int i = start; i <= end; i++) {
        if (valid[base + i]) {
            sum += closes[base + i];
            count++;
        }
    }
    return count > 0 ? sum / count : 0.0;
}

static inline double compute_margin_interest(double margin_borrowed, int entry_day, int current_day, double annual_rate) {
    if (margin_borrowed <= 0) return 0.0;
    int days = current_day - entry_day;
    if (days < 1) days = 1;
    return margin_borrowed * (annual_rate / 365.0) * days;
}

static inline void force_close_all(
    CPosition *positions, int n_symbols,
    const int * restrict valid_mask, const double * restrict closes,
    const double * restrict fee_rates,
    int n_bars, int d,
    double *cash, int *total_trades, int *n_exits, int *n_wins
) {
    for (int s = 0; s < n_symbols; s++) {
        if (!positions[s].active) continue;
        double cp;
        const int sidx = s * n_bars + d;
        if (valid_mask[sidx])
            cp = closes[sidx];
        else
            cp = positions[s].entry_price;
        double fr = fee_rates[s];
        double pnl;
        if (positions[s].direction == 0) {
            double proceeds = positions[s].quantity * cp * (1.0 - fr);
            pnl = proceeds - positions[s].cost_basis;
            *cash += proceeds;
        } else {
            pnl = positions[s].quantity * (positions[s].entry_price - cp);
            *cash += positions[s].cost_basis + pnl;
        }
        (*total_trades)++;
        (*n_exits)++;
        if (pnl > 0) (*n_wins)++;
        positions[s].active = 0;
    }
}

static inline void insertion_sort_candidates(Candidate *c, int n) {
    for (int i = 1; i < n; i++) {
        Candidate key = c[i];
        int j = i - 1;
        while (j >= 0 && c[j].score < key.score) {
            c[j + 1] = c[j];
            j--;
        }
        c[j + 1] = key;
    }
}

static void worksteal_simulate_internal(
    const double * restrict timestamps,
    const int * restrict valid_mask,
    const double * restrict opens,
    const double * restrict highs,
    const double * restrict lows,
    const double * restrict closes,
    const double * restrict fee_rates,
    int n_bars,
    int n_symbols,
    const WorkStealConfig * restrict cfg,
    SimResult * restrict result,
    double * restrict equity_curve,
    int * restrict cumvalid
) {
    if (n_bars <= 0 || n_symbols <= 0) {
        memset(result, 0, sizeof(SimResult));
        return;
    }

    /* precompute cumulative valid counts per symbol */
    for (int s = 0; s < n_symbols; s++) {
        const int base = s * n_bars;
        int running = 0;
        for (int d = 0; d < n_bars; d++) {
            running += valid_mask[base + d];
            cumvalid[base + d] = running;
        }
    }

    CPosition positions[MAX_SYM];
    int last_exit_day[MAX_SYM];
    memset(positions, 0, sizeof(CPosition) * MAX_SYM);
    for (int s = 0; s < MAX_SYM; s++) last_exit_day[s] = -9999;

    double cash = cfg->initial_cash;
    int n_active = 0;
    int total_trades = 0;
    int n_wins = 0;
    int n_exits = 0;
    int eq_len = 0;
    double peak_equity = cfg->initial_cash;
    double max_dd_positive = 0.0;

    const double trailing_pct = cfg->trailing_stop_pct;
    const int has_trailing = trailing_pct > 0;
    const int max_hold = cfg->max_hold_days;
    const double margin_rate = cfg->margin_annual_rate;
    const double dip_pct = cfg->dip_pct;
    const double prox_pct = cfg->proximity_pct;
    const double tp_pct = cfg->profit_target_pct;
    const double sl_pct = cfg->stop_loss_pct;
    const double max_pos_pct = cfg->max_position_pct;
    const double max_lev = cfg->max_leverage;
    const double init_cash = cfg->initial_cash;
    const int lookback = cfg->lookback_days;
    const int sma_period = cfg->sma_filter_period;
    const int mom_period = cfg->momentum_period;
    const double mom_min = cfg->momentum_min;
    const int cooldown = cfg->reentry_cooldown_days;
    const int enable_shorts = cfg->enable_shorts;
    const double short_pump = cfg->short_pump_pct;
    const double dd_exit = cfg->max_drawdown_exit;

    Candidate cands[MAX_SYM * 2];

    for (int d = 0; d < n_bars; d++) {
        /* 1. Check exits */
        for (int s = 0; s < n_symbols; s++) {
            if (!positions[s].active) continue;

            const int idx = s * n_bars + d;
            if (!valid_mask[idx]) continue;

            CPosition *pos = &positions[s];
            const double c = closes[idx];
            const double h = highs[idx];
            const double l = lows[idx];

            double exit_price = 0;
            int do_exit = 0;

            if (pos->direction == 0) {
                if (h > pos->peak_price) pos->peak_price = h;

                if (h >= pos->target_exit) {
                    exit_price = pos->target_exit;
                    do_exit = 1;
                } else if (l <= pos->stop_price) {
                    exit_price = pos->stop_price;
                    do_exit = 1;
                } else if (has_trailing) {
                    double trail = pos->peak_price * (1.0 - trailing_pct);
                    if (l <= trail) {
                        exit_price = trail;
                        do_exit = 1;
                    }
                }
            } else {
                if (l < pos->peak_price) pos->peak_price = l;

                if (l <= pos->target_exit) {
                    exit_price = pos->target_exit;
                    do_exit = 1;
                } else if (h >= pos->stop_price) {
                    exit_price = pos->stop_price;
                    do_exit = 1;
                } else if (has_trailing) {
                    double trail = pos->peak_price * (1.0 + trailing_pct);
                    if (h >= trail) {
                        exit_price = trail;
                        do_exit = 1;
                    }
                }
                if (!do_exit) {
                    double unrealized = pos->quantity * (c - pos->entry_price);
                    if (unrealized > pos->cost_basis * 0.8) {
                        exit_price = c;
                        do_exit = 1;
                    }
                }
            }

            if (!do_exit && max_hold > 0) {
                if ((d - pos->entry_day) >= max_hold) {
                    exit_price = c;
                    do_exit = 1;
                }
            }

            if (do_exit) {
                double fr = fee_rates[s];
                double interest = compute_margin_interest(pos->margin_borrowed, pos->entry_day, d, margin_rate);
                double pnl;

                if (pos->direction == 0) {
                    double proceeds = pos->quantity * exit_price * (1.0 - fr);
                    pnl = proceeds - pos->cost_basis - interest;
                    cash += proceeds;
                } else {
                    pnl = pos->quantity * (pos->entry_price - exit_price)
                        - pos->quantity * exit_price * fr
                        - pos->quantity * pos->entry_price * fr
                        - interest;
                    cash += pos->cost_basis + pnl;
                }

                total_trades++;
                n_exits++;
                if (pnl > 0) n_wins++;
                last_exit_day[s] = d;
                pos->active = 0;
                n_active--;
            }
        }

        /* 2. Check entries */
        if (n_active < cfg->max_positions) {
            int nc = 0;

            for (int s = 0; s < n_symbols; s++) {
                if (positions[s].active) continue;
                const int base = s * n_bars;
                if (!valid_mask[base + d]) continue;
                int n_valid = cumvalid[base + d];
                if (n_valid < lookback) continue;
                if (cooldown > 0 && (d - last_exit_day[s]) < cooldown) continue;

                const double c = closes[base + d];
                const double h_bar = highs[base + d];
                const double l_bar = lows[base + d];

                if (mom_period > 0 && n_valid > mom_period) {
                    int cnt = 0;
                    double past_close = c;
                    for (int j = d; j >= 0; j--) {
                        if (valid_mask[base + j]) {
                            cnt++;
                            if (cnt == mom_period + 1) {
                                past_close = closes[base + j];
                                break;
                            }
                        }
                    }
                    if (past_close > 0) {
                        double mom = (c - past_close) / past_close;
                        if (mom < mom_min) continue;
                    }
                }

                if (sma_period > 0) {
                    double sma = compute_sma(closes, valid_mask, n_bars, s, d, sma_period);
                    if (c < sma) continue;
                }

                double ref_high = rolling_max_high(highs, valid_mask, n_bars, s, d, lookback);
                if (ref_high <= 0) continue;
                double buy_target = ref_high * (1.0 - dip_pct);
                double dist_long = (c - buy_target) / ref_high;

                if (dist_long <= prox_pct) {
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

                if (enable_shorts) {
                    double ref_low = rolling_min_low(lows, valid_mask, n_bars, s, d, lookback);
                    if (ref_low <= 0) continue;
                    double short_target = ref_low * (1.0 + short_pump);
                    double dist_short = (short_target - c) / ref_low;
                    if (dist_short <= prox_pct) {
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

            if (nc > 1) insertion_sort_candidates(cands, nc);

            int slots = cfg->max_positions - n_active;
            double base_eq = init_cash;
            for (int k = 0; k < nc && slots > 0; k++) {
                int s = cands[k].symbol;
                if (positions[s].active) continue;

                double fr = fee_rates[s];
                double max_alloc = base_eq * max_pos_pct * max_lev;
                double fill = cands[k].fill_price;

                if (cands[k].direction == 0) {
                    double alloc = max_alloc < cash ? max_alloc : cash;
                    double qty = alloc / (fill * (1.0 + fr));
                    if (qty <= 0) continue;

                    double actual_cost = qty * fill * (1.0 + fr);
                    double borrowed = actual_cost - cash;
                    if (borrowed < 0) borrowed = 0;
                    double deduct = actual_cost < cash ? actual_cost : cash;
                    cash -= deduct;

                    positions[s].active = 1;
                    positions[s].direction = 0;
                    positions[s].entry_price = fill;
                    positions[s].entry_day = d;
                    positions[s].quantity = qty;
                    positions[s].cost_basis = actual_cost;
                    positions[s].peak_price = cands[k].bar_high;
                    positions[s].target_exit = fill * (1.0 + tp_pct);
                    positions[s].stop_price = fill * (1.0 - sl_pct);
                    positions[s].margin_borrowed = borrowed;
                    n_active++;
                    slots--;
                    total_trades++;
                } else {
                    double alloc_limit = base_eq * max_pos_pct;
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
                    positions[s].target_exit = fill * (1.0 - tp_pct);
                    positions[s].stop_price = fill * (1.0 + sl_pct);
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
            const int idx = s * n_bars + d;
            if (!valid_mask[idx]) {
                if (positions[s].direction == 0)
                    inv += positions[s].quantity * positions[s].entry_price;
                continue;
            }
            double c = closes[idx];
            double interest = compute_margin_interest(positions[s].margin_borrowed,
                positions[s].entry_day, d, margin_rate);
            if (positions[s].direction == 0) {
                inv += positions[s].quantity * c - interest;
            } else {
                inv += positions[s].quantity * (positions[s].entry_price - c) - interest;
            }
        }
        double equity = cash + inv;
        equity_curve[eq_len++] = equity;

        if (equity > peak_equity) peak_equity = equity;

        if (peak_equity > 0) {
            double dd_pos = (peak_equity - equity) / peak_equity;
            if (dd_pos > max_dd_positive) max_dd_positive = dd_pos;
        }

        /* drawdown-vs-profit early exit */
        if (n_bars >= 20 && eq_len >= 2) {
            double progress = (double)eq_len / (double)n_bars;
            if (progress >= 0.5) {
                double total_ret = (equity - init_cash) / init_cash;
                if (max_dd_positive > total_ret) {
                    force_close_all(positions, n_symbols, valid_mask, closes,
                                    fee_rates, n_bars, d, &cash, &total_trades,
                                    &n_exits, &n_wins);
                    n_active = 0;
                    equity_curve[eq_len - 1] = cash;
                    break;
                }
            }
        }

        /* max drawdown early exit */
        if (dd_exit > 0 && eq_len > 1 && peak_equity > 0) {
            double dd = (equity - peak_equity) / peak_equity;
            if (dd < -dd_exit) {
                force_close_all(positions, n_symbols, valid_mask, closes,
                                fee_rates, n_bars, d, &cash, &total_trades,
                                &n_exits, &n_wins);
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
        result->final_equity = init_cash;
        result->mean_daily_return = 0;
        return;
    }

    int n_ret = eq_len - 1;
    double sum_ret = 0;
    double sum_neg = 0, sum_neg_sq = 0;
    int n_neg = 0;
    double rmax = equity_curve[0];
    double mdd = 0;

    /* single pass: returns + downside stats + max drawdown */
    for (int i = 0; i < n_ret; i++) {
        double denom = equity_curve[i];
        if (denom < 1e-8) denom = 1e-8;
        double r = (equity_curve[i + 1] - equity_curve[i]) / denom;
        sum_ret += r;
        if (r < 0) {
            sum_neg += r;
            sum_neg_sq += r * r;
            n_neg++;
        }

        double ev = equity_curve[i + 1];
        if (ev > rmax) rmax = ev;
        double dd = (ev - rmax) / rmax;
        if (dd < mdd) mdd = dd;
    }
    /* also check first element for max drawdown */
    if (equity_curve[0] > rmax) rmax = equity_curve[0];
    {
        double dd0 = (equity_curve[0] - rmax) / rmax;
        if (dd0 < mdd) mdd = dd0;
    }

    double mean_ret = sum_ret / n_ret;
    result->mean_daily_return = mean_ret;

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
    double prev_eq = equity_curve[0];
    for (int i = 0; i < n_ret; i++) {
        double denom = prev_eq;
        if (denom < 1e-8) denom = 1e-8;
        double r = (equity_curve[i + 1] - prev_eq) / denom;
        double diff = r - mean_ret;
        sum_sq += diff * diff;
        prev_eq = equity_curve[i + 1];
    }
    double std_ret = n_ret > 1 ? sqrt(sum_sq / n_ret) : 1e-8;

    double sqrt365 = sqrt(365.0);
    result->sortino = mean_ret / fmax(downside_std, 1e-8) * sqrt365;
    result->sharpe = mean_ret / fmax(std_ret, 1e-8) * sqrt365;
    result->total_return = (equity_curve[eq_len - 1] - equity_curve[0]) / equity_curve[0];
    result->final_equity = equity_curve[eq_len - 1];
    result->max_drawdown = mdd;
    result->win_rate = n_exits > 0 ? (double)n_wins / n_exits * 100.0 : 0.0;
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
    int *cumvalid = (int *)malloc(n_symbols * n_bars * sizeof(int));
    if (!cumvalid) {
        memset(result, 0, sizeof(SimResult));
        return;
    }
    worksteal_simulate_internal(timestamps, valid_mask, opens, highs, lows, closes,
                                fee_rates, n_bars, n_symbols, cfg, result, equity_curve, cumvalid);
    free(cumvalid);
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
#ifdef _OPENMP
    #pragma omp parallel
    {
        double *eq = (double *)malloc(n_bars * sizeof(double));
        int *cumvalid = (int *)malloc(n_symbols * n_bars * sizeof(int));
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_configs; i++) {
            worksteal_simulate_internal(timestamps, valid_mask, opens, highs, lows, closes,
                                        fee_rates, n_bars, n_symbols, &configs[i], &results[i],
                                        eq, cumvalid);
        }
        free(eq);
        free(cumvalid);
    }
#else
    double *eq = (double *)malloc(n_bars * sizeof(double));
    int *cumvalid = (int *)malloc(n_symbols * n_bars * sizeof(int));
    for (int i = 0; i < n_configs; i++) {
        worksteal_simulate_internal(timestamps, valid_mask, opens, highs, lows, closes,
                                    fee_rates, n_bars, n_symbols, &configs[i], &results[i],
                                    eq, cumvalid);
    }
    free(eq);
    free(cumvalid);
#endif
}

int worksteal_get_num_threads(void) {
#ifdef _OPENMP
    int n;
    #pragma omp parallel
    {
        #pragma omp single
        n = omp_get_num_threads();
    }
    return n;
#else
    return 1;
#endif
}
