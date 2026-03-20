#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "worksteal_sim.h"

#define MAX_SYM 64
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

static double compute_atr(const double *highs, const double *lows, const double *closes,
                           const int *valid, int n_bars, int sym, int end, int period) {
    if (period <= 0) return 0.0;
    /* Collect last (period+1) valid bar indices to compute period TR values */
    int indices[256];
    int n_idx = 0;
    int need = period + 1;
    for (int i = end; i >= 0 && n_idx < need; i--) {
        if (valid[IDX(sym, i)]) {
            indices[n_idx++] = i;
        }
    }
    if (n_idx < 2) return highs[IDX(sym, end)] - lows[IDX(sym, end)];
    /* indices are in reverse order; compute TR for the last min(n_idx-1, period) bars */
    int n_tr = n_idx - 1;
    if (n_tr > period) n_tr = period;
    double sum_tr = 0.0;
    for (int k = 0; k < n_tr; k++) {
        int cur = indices[k];      /* more recent */
        int prev = indices[k + 1]; /* previous bar */
        double h = highs[IDX(sym, cur)];
        double l = lows[IDX(sym, cur)];
        double pc = closes[IDX(sym, prev)];
        double tr = h - l;
        double d2 = fabs(h - pc);
        double d3 = fabs(l - pc);
        if (d2 > tr) tr = d2;
        if (d3 > tr) tr = d3;
        sum_tr += tr;
    }
    return sum_tr / n_tr;
}

/* Compute market breadth: fraction of symbols with negative day-over-day return */
static double compute_market_breadth(const double *closes, const int *valid,
                                      int n_bars, int n_symbols, int d) {
    int n_neg = 0;
    int n_total = 0;
    for (int s = 0; s < n_symbols; s++) {
        if (!valid[IDX(s, d)]) continue;
        /* find previous valid bar for this symbol */
        int prev = -1;
        for (int j = d - 1; j >= 0; j--) {
            if (valid[IDX(s, j)]) { prev = j; break; }
        }
        if (prev < 0) continue;
        n_total++;
        if (closes[IDX(s, d)] < closes[IDX(s, prev)])
            n_neg++;
    }
    return n_total > 0 ? (double)n_neg / n_total : 0.0;
}

/* Check risk-off conditions:
   1. SMA check: if <50% of symbols are above their N-period SMA, risk-off
   2. Momentum check: if avg N-day momentum < threshold, risk-off */
static int check_risk_off(const double *closes, const int *valid,
                           int n_bars, int n_symbols, int d,
                           const WorkStealConfig *cfg) {
    if (cfg->risk_off_sma_period <= 0 && cfg->risk_off_momentum_period <= 0)
        return 0;

    int risk_off = 0;

    /* SMA check */
    if (cfg->risk_off_sma_period > 0) {
        int above_sma = 0;
        int total_checked = 0;
        for (int s = 0; s < n_symbols; s++) {
            if (!valid[IDX(s, d)]) continue;
            int n_valid = count_valid_bars(valid, n_bars, s, d);
            if (n_valid < cfg->risk_off_sma_period) continue;
            total_checked++;
            double sma = compute_sma(closes, valid, n_bars, s, d, cfg->risk_off_sma_period);
            if (closes[IDX(s, d)] > sma) above_sma++;
        }
        if (total_checked > 0 && (double)above_sma / total_checked < 0.5)
            risk_off = 1;
    }

    /* Momentum check */
    if (cfg->risk_off_momentum_period > 0) {
        double mom_sum = 0.0;
        int mom_count = 0;
        for (int s = 0; s < n_symbols; s++) {
            if (!valid[IDX(s, d)]) continue;
            int cnt = 0;
            double past_close = 0.0;
            for (int j = d; j >= 0; j--) {
                if (valid[IDX(s, j)]) {
                    cnt++;
                    if (cnt == cfg->risk_off_momentum_period + 1) {
                        past_close = closes[IDX(s, j)];
                        break;
                    }
                }
            }
            if (cnt < cfg->risk_off_momentum_period + 1) continue;
            if (past_close > 0) {
                mom_sum += (closes[IDX(s, d)] - past_close) / past_close;
                mom_count++;
            }
        }
        if (mom_count > 0) {
            double avg_mom = mom_sum / mom_count;
            if (avg_mom < cfg->risk_off_momentum_threshold)
                risk_off = 1;
        }
    }

    return risk_off;
}

static void force_close_all(CPosition *positions, int n_symbols,
                            const int *valid_mask, const double *closes,
                            const double *fee_rates, int n_bars, int d,
                            double *cash, int *total_trades, int *n_exits,
                            int *n_wins, int *n_active) {
    for (int s = 0; s < n_symbols; s++) {
        if (!positions[s].active) continue;
        double cp;
        if (valid_mask[IDX(s, d)])
            cp = closes[IDX(s, d)];
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
    *n_active = 0;
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
    double max_dd_positive = 0.0;

    Candidate cands[MAX_SYM * 2];

    for (int d = 0; d < n_bars; d++) {
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
                double interest = compute_margin_interest(pos->margin_borrowed, pos->entry_day, d, cfg->margin_annual_rate);

                if (pos->direction == 0) {
                    double proceeds = pos->quantity * exit_price * (1.0 - fee_rate);
                    pnl = proceeds - pos->cost_basis - interest;
                    cash += proceeds;
                } else {
                    pnl = pos->quantity * (pos->entry_price - exit_price)
                        - pos->quantity * exit_price * fee_rate
                        - pos->quantity * pos->entry_price * fee_rate
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

        /* 2. Market breadth + risk-off check */
        int skip_entries = 0;
        double breadth_threshold = cfg->market_breadth_filter;

        if (cfg->risk_off_sma_period > 0 || cfg->risk_off_momentum_period > 0) {
            int risk_off = check_risk_off(closes, valid_mask, n_bars, n_symbols, d, cfg);
            if (risk_off && cfg->risk_off_breadth_threshold > 0) {
                breadth_threshold = cfg->risk_off_breadth_threshold;
            }
        }

        if (breadth_threshold > 0) {
            double breadth = compute_market_breadth(closes, valid_mask, n_bars, n_symbols, d);
            if (breadth > breadth_threshold)
                skip_entries = 1;
        }

        /* 3. Check entries (work-stealing) */
        if (n_active < cfg->max_positions && !skip_entries) {
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

                /* Compute dip_pct: use ATR-adaptive if configured */
                double effective_dip = cfg->dip_pct;
                if (cfg->atr_period > 0 && cfg->atr_dip_mult > 0) {
                    double atr = compute_atr(highs, lows, closes, valid_mask, n_bars, s, d, cfg->atr_period);
                    if (c > 0 && atr > 0) {
                        double atr_dip = (atr * cfg->atr_dip_mult) / c;
                        if (atr_dip > effective_dip) effective_dip = atr_dip;
                    }
                }

                /* long candidate */
                double ref_high = rolling_max_high(highs, valid_mask, n_bars, s, d, cfg->lookback_days);
                if (ref_high <= 0) continue;
                double buy_target = ref_high * (1.0 - effective_dip);
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
                    double alloc = max_alloc < cash ? max_alloc : cash;
                    double qty = alloc / (fill * (1.0 + fee_rate));
                    if (qty < 1e-8) continue;

                    double actual_cost = qty * fill * (1.0 + fee_rate);
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
                    if (qty < 1e-8) continue;
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

        /* 4. Compute equity */
        double inv = 0.0;
        for (int s = 0; s < n_symbols; s++) {
            if (!positions[s].active) continue;
            if (!valid_mask[IDX(s, d)]) {
                if (positions[s].direction == 0)
                    inv += positions[s].quantity * positions[s].entry_price;
                continue;
            }
            double c = closes[IDX(s, d)];
            double interest = compute_margin_interest(positions[s].margin_borrowed,
                positions[s].entry_day, d, cfg->margin_annual_rate);
            if (positions[s].direction == 0) {
                inv += positions[s].quantity * c - interest;
            } else {
                double unrealized = positions[s].quantity * (positions[s].entry_price - c) - interest;
                inv += unrealized;
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
                double total_ret = (equity - cfg->initial_cash) / cfg->initial_cash;
                if (max_dd_positive > total_ret) {
                    force_close_all(positions, n_symbols, valid_mask, closes,
                                    fee_rates, n_bars, d, &cash,
                                    &total_trades, &n_exits, &n_wins, &n_active);
                    equity_curve[eq_len - 1] = cash;
                    break;
                }
            }
        }

        /* max drawdown early exit */
        if (cfg->max_drawdown_exit > 0 && eq_len > 1 && peak_equity > 0) {
            double dd = (equity - peak_equity) / peak_equity;
            if (dd < -cfg->max_drawdown_exit) {
                force_close_all(positions, n_symbols, valid_mask, closes,
                                fee_rates, n_bars, d, &cash,
                                &total_trades, &n_exits, &n_wins, &n_active);
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
    double sum_neg_sq = 0;
    int n_neg = 0;
    for (int i = 0; i < n_ret; i++) {
        if (rets[i] < 0) {
            sum_neg_sq += rets[i] * rets[i];
            n_neg++;
        }
    }
    double downside_std;
    if (n_neg > 1) {
        double var_neg = sum_neg_sq / n_neg;
        /* Use population std of negative returns (matching numpy .std() default) */
        double sum_neg = 0;
        for (int i = 0; i < n_ret; i++) {
            if (rets[i] < 0) sum_neg += rets[i];
        }
        double mean_neg = sum_neg / n_neg;
        var_neg = sum_neg_sq / n_neg - mean_neg * mean_neg;
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
