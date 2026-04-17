#include "market_sim.h"
#include <stdlib.h>
#include <string.h>

static double clamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static double safe_periods_per_year(double periods_per_year) {
    return periods_per_year > 0.0 ? periods_per_year : ANNUALIZE_FACTOR;
}

double compute_annualized_return(double total_return, int n_periods, double periods_per_year) {
    if (n_periods <= 0) return 0.0;
    if (total_return <= -1.0) return -1.0;
    double ppy = safe_periods_per_year(periods_per_year);
    double growth = 1.0 + total_return;
    return pow(growth, ppy / (double)n_periods) - 1.0;
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
    if (!equity_curve || n_eq <= 0) return 0.0;
    double running_max = equity_curve[0];
    double max_dd = 0.0;
    for (int i = 0; i < n_eq; i++) {
        if (equity_curve[i] > running_max) running_max = equity_curve[i];
        double dd = (running_max - equity_curve[i]) / (running_max + 1e-10);
        if (dd > max_dd) max_dd = dd;
    }
    return max_dd;
}

static void clamp_target_weights(
    const double *raw,
    double *clamped,
    int n_symbols,
    int can_short,
    double max_gross_leverage
) {
    double gross = 0.0;
    for (int s = 0; s < n_symbols; s++) {
        double w = raw[s];
        if (!can_short && w < 0.0) {
            w = 0.0;
        }
        clamped[s] = w;
        gross += fabs(w);
    }

    double gross_limit = max_gross_leverage > 0.0 ? max_gross_leverage : 1.0;
    if (gross > gross_limit && gross > 0.0) {
        double scale = gross_limit / gross;
        for (int s = 0; s < n_symbols; s++) {
            clamped[s] *= scale;
        }
    }
}

static void score_to_target_weights(
    const double *raw_scores,
    int n_symbols,
    const WeightSimConfig *cfg,
    double *out_weights
) {
    if (cfg->can_short) {
        double gross = 0.0;
        for (int s = 0; s < n_symbols; s++) {
            out_weights[s] = tanh(raw_scores[s]);
            gross += fabs(out_weights[s]);
        }
        {
            double gross_limit = cfg->max_gross_leverage > 0.0 ? cfg->max_gross_leverage : 1.0;
            if (gross > gross_limit && gross > 0.0) {
                double scale = gross_limit / gross;
                for (int s = 0; s < n_symbols; s++) {
                    out_weights[s] *= scale;
                }
            }
        }
        return;
    }

    double max_score = raw_scores[0];
    for (int s = 1; s < n_symbols; s++) {
        if (raw_scores[s] > max_score) max_score = raw_scores[s];
    }

    double exp_sum = 0.0;
    for (int s = 0; s < n_symbols; s++) {
        double shifted = raw_scores[s] - max_score;
        if (shifted < -30.0) shifted = -30.0;
        if (shifted > 30.0) shifted = 30.0;
        out_weights[s] = exp(shifted);
        exp_sum += out_weights[s];
    }

    double gross_scale = cfg->max_gross_leverage > 0.0 ? cfg->max_gross_leverage : 1.0;
    if (exp_sum <= 0.0) {
        for (int s = 0; s < n_symbols; s++) out_weights[s] = 0.0;
        return;
    }
    for (int s = 0; s < n_symbols; s++) {
        out_weights[s] = (out_weights[s] / exp_sum) * gross_scale;
    }
}

int weight_env_obs_dim(const WeightEnv *env) {
    if (!env) return 0;
    return env->env_cfg.lookback * env->n_symbols + env->n_symbols + 4;
}

int weight_env_init(
    WeightEnv *env,
    const double *close,
    int n_bars,
    int n_symbols,
    const WeightEnvConfig *env_cfg,
    const WeightSimConfig *sim_cfg
) {
    if (!env) return -1;
    memset(env, 0, sizeof(*env));
    if (!close || !env_cfg || !sim_cfg) return -1;
    if (n_bars <= 2 || n_symbols <= 0 || n_symbols > MAX_SYMBOLS) return -1;

    env->close = close;
    env->n_bars = n_bars;
    env->n_symbols = n_symbols;
    env->env_cfg = *env_cfg;
    env->sim_cfg = *sim_cfg;

    if (env->env_cfg.lookback <= 1) env->env_cfg.lookback = 48;
    if (env->env_cfg.episode_steps <= 0) env->env_cfg.episode_steps = 168;
    if (env->env_cfg.reward_scale == 0.0) env->env_cfg.reward_scale = 100.0;
    if (env->sim_cfg.initial_cash <= 0.0) env->sim_cfg.initial_cash = 10000.0;
    if (env->sim_cfg.periods_per_year <= 0.0) env->sim_cfg.periods_per_year = ANNUALIZE_FACTOR;
    if (env->sim_cfg.max_gross_leverage <= 0.0) env->sim_cfg.max_gross_leverage = 1.0;

    if (n_bars < env->env_cfg.lookback + env->env_cfg.episode_steps + 1) return -1;

    env->equity_curve = (double *)calloc((size_t)env->env_cfg.episode_steps + 1, sizeof(double));
    env->returns = (double *)calloc((size_t)env->env_cfg.episode_steps, sizeof(double));
    if (!env->equity_curve || !env->returns) {
        free(env->equity_curve);
        free(env->returns);
        memset(env, 0, sizeof(*env));
        return -1;
    }
    return 0;
}

void weight_env_free(WeightEnv *env) {
    if (!env) return;
    free(env->equity_curve);
    free(env->returns);
    memset(env, 0, sizeof(*env));
}

int weight_env_reset(WeightEnv *env, int start_index) {
    if (!env || !env->close || !env->equity_curve || !env->returns) return -1;

    int min_start = env->env_cfg.lookback;
    int max_start = env->n_bars - env->env_cfg.episode_steps - 1;
    if (max_start < min_start) return -1;
    if (start_index < min_start) start_index = min_start;
    if (start_index > max_start) start_index = max_start;

    env->start_index = start_index;
    env->t = start_index;
    env->steps = 0;
    env->equity = env->sim_cfg.initial_cash;
    env->peak_equity = env->sim_cfg.initial_cash;
    env->recent_return = 0.0;
    env->total_turnover = 0.0;
    env->total_fees = 0.0;
    env->total_borrow_cost = 0.0;
    memset(env->current_weights, 0, sizeof(env->current_weights));
    memset(env->equity_curve, 0, ((size_t)env->env_cfg.episode_steps + 1) * sizeof(double));
    memset(env->returns, 0, (size_t)env->env_cfg.episode_steps * sizeof(double));
    env->equity_curve[0] = env->equity;
    return 0;
}

int weight_env_get_obs(const WeightEnv *env, double *out_obs, int obs_len) {
    if (!env || !out_obs) return -1;
    int need = weight_env_obs_dim(env);
    if (obs_len < need) return -1;

    int lookback = env->env_cfg.lookback;
    int n_symbols = env->n_symbols;
    int offset = 0;

    for (int i = 0; i < lookback * n_symbols; i++) out_obs[i] = 0.0;
    offset += lookback * n_symbols;

    int window_start = env->t - lookback;
    for (int row = 1; row < lookback; row++) {
        for (int s = 0; s < n_symbols; s++) {
            double prev_px = env->close[(window_start + row - 1) * n_symbols + s];
            double cur_px = env->close[(window_start + row) * n_symbols + s];
            prev_px = prev_px > 1e-12 ? prev_px : 1e-12;
            cur_px = cur_px > 1e-12 ? cur_px : 1e-12;
            out_obs[(row * n_symbols) + s] = log(cur_px) - log(prev_px);
        }
    }

    for (int s = 0; s < n_symbols; s++) {
        out_obs[offset + s] = env->current_weights[s];
    }
    offset += n_symbols;

    {
        double drawdown = 1.0 - (env->equity / (env->peak_equity + 1e-10));
        out_obs[offset++] = env->recent_return;
        out_obs[offset++] = drawdown;
        out_obs[offset++] = env->equity / (env->sim_cfg.initial_cash + 1e-10) - 1.0;
        out_obs[offset++] = (double)env->steps / (double)env->env_cfg.episode_steps;
    }

    return 0;
}

int weight_env_step(
    WeightEnv *env,
    const double *raw_scores,
    int raw_len,
    WeightEnvStepInfo *out_info
) {
    if (!env || !raw_scores || raw_len < env->n_symbols) return -1;
    if (env->steps >= env->env_cfg.episode_steps || env->t >= env->n_bars - 1) return -1;

    double next_weights[MAX_SYMBOLS];
    score_to_target_weights(raw_scores, env->n_symbols, &env->sim_cfg, next_weights);

    double turnover = 0.0;
    for (int s = 0; s < env->n_symbols; s++) {
        turnover += fabs(next_weights[s] - env->current_weights[s]);
    }
    double fees = env->equity * turnover * env->sim_cfg.fee_rate;

    double long_exposure = 0.0;
    double short_exposure = 0.0;
    for (int s = 0; s < env->n_symbols; s++) {
        if (next_weights[s] > 0.0) long_exposure += next_weights[s];
        if (next_weights[s] < 0.0) short_exposure += -next_weights[s];
    }
    double borrow_base = 0.0;
    if (long_exposure > 1.0) borrow_base += long_exposure - 1.0;
    borrow_base += short_exposure;
    double borrow_cost = env->equity * borrow_base * env->sim_cfg.borrow_rate_per_period;

    double gross_return = 0.0;
    for (int s = 0; s < env->n_symbols; s++) {
        double prev_px = env->close[env->t * env->n_symbols + s];
        double next_px = env->close[(env->t + 1) * env->n_symbols + s];
        if (prev_px <= 1e-12) continue;
        gross_return += next_weights[s] * ((next_px - prev_px) / prev_px);
    }

    double new_equity = env->equity * (1.0 + gross_return) - fees - borrow_cost;
    if (new_equity < 0.0) new_equity = 0.0;
    double period_return = env->equity > 1e-12 ? (new_equity / env->equity - 1.0) : 0.0;

    for (int s = 0; s < env->n_symbols; s++) env->current_weights[s] = next_weights[s];
    env->equity = new_equity;
    if (env->equity > env->peak_equity) env->peak_equity = env->equity;
    env->recent_return = period_return;
    env->total_turnover += turnover;
    env->total_fees += fees;
    env->total_borrow_cost += borrow_cost;
    env->returns[env->steps] = period_return;
    env->steps += 1;
    env->t += 1;
    env->equity_curve[env->steps] = env->equity;

    int done = env->steps >= env->env_cfg.episode_steps || env->t >= env->n_bars - 1;
    if (out_info) {
        memset(out_info, 0, sizeof(*out_info));
        out_info->reward = period_return * env->env_cfg.reward_scale;
        out_info->turnover = turnover;
        out_info->fees = fees;
        out_info->borrow_cost = borrow_cost;
        out_info->equity = env->equity;
        out_info->period_return = period_return;
        out_info->done = done;

        if (done) {
            int n_eq = env->steps + 1;
            out_info->summary.total_return = env->equity / env->sim_cfg.initial_cash - 1.0;
            out_info->summary.annualized_return = compute_annualized_return(
                out_info->summary.total_return,
                env->steps,
                env->sim_cfg.periods_per_year
            );
            out_info->summary.sortino = compute_sortino(env->equity_curve, n_eq);
            out_info->summary.max_drawdown = compute_max_drawdown(env->equity_curve, n_eq);
            out_info->summary.final_equity = env->equity;
            out_info->summary.total_turnover = env->total_turnover;
            out_info->summary.total_fees = env->total_fees;
            out_info->summary.total_borrow_cost = env->total_borrow_cost;
        }
    }
    return 0;
}

void simulate_target_weights(
    int n_bars,
    int n_symbols,
    const double *close,
    const double *target_weights,
    const WeightSimConfig *cfg,
    WeightSimResult *result,
    double *equity_curve
) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    if (!cfg || !close || !target_weights || !equity_curve || n_bars <= 0 || n_symbols <= 0 || n_symbols > MAX_SYMBOLS) {
        return;
    }

    double equity = cfg->initial_cash > 0.0 ? cfg->initial_cash : 10000.0;
    equity_curve[0] = equity;

    if (n_bars == 1) {
        result->final_equity = equity;
        return;
    }

    double prev_weights[MAX_SYMBOLS];
    double next_weights[MAX_SYMBOLS];
    memset(prev_weights, 0, sizeof(prev_weights));

    double total_turnover = 0.0;
    double total_fees = 0.0;
    double total_borrow = 0.0;

    for (int t = 0; t < n_bars - 1; t++) {
        clamp_target_weights(
            &target_weights[t * n_symbols],
            next_weights,
            n_symbols,
            cfg->can_short,
            cfg->max_gross_leverage
        );

        double turnover = 0.0;
        double gross_long = 0.0;
        double gross_short = 0.0;
        double portfolio_growth = 1.0;

        for (int s = 0; s < n_symbols; s++) {
            turnover += fabs(next_weights[s] - prev_weights[s]);
            if (next_weights[s] > 0.0) gross_long += next_weights[s];
            else gross_short += -next_weights[s];
        }

        double cash_weight = 1.0;
        for (int s = 0; s < n_symbols; s++) {
            cash_weight -= next_weights[s];
        }

        portfolio_growth = cash_weight;
        for (int s = 0; s < n_symbols; s++) {
            double p0 = close[t * n_symbols + s];
            double p1 = close[(t + 1) * n_symbols + s];
            double rel = 1.0;
            if (p0 > 0.0 && p1 > 0.0) {
                rel = p1 / p0;
            }
            portfolio_growth += next_weights[s] * rel;
        }

        double fees = equity * turnover * (cfg->fee_rate > 0.0 ? cfg->fee_rate : 0.0);
        double borrow_exposure = fmax(gross_long - 1.0, 0.0) + gross_short;
        double borrow_cost = equity * borrow_exposure * (cfg->borrow_rate_per_period > 0.0 ? cfg->borrow_rate_per_period : 0.0);
        double equity_after_costs = equity - fees - borrow_cost;
        if (equity_after_costs < 0.0) equity_after_costs = 0.0;

        equity = equity_after_costs * portfolio_growth;
        if (!isfinite(equity) || equity < 0.0) {
            equity = 0.0;
        }

        equity_curve[t + 1] = equity;
        total_turnover += turnover;
        total_fees += fees;
        total_borrow += borrow_cost;
        memcpy(prev_weights, next_weights, (size_t)n_symbols * sizeof(double));
    }

    result->final_equity = equity_curve[n_bars - 1];
    result->total_return = cfg->initial_cash > 0.0
        ? (result->final_equity / cfg->initial_cash) - 1.0
        : 0.0;
    result->annualized_return = compute_annualized_return(
        result->total_return,
        n_bars - 1,
        cfg->periods_per_year
    );
    result->sortino = compute_sortino(equity_curve, n_bars);
    result->max_drawdown = compute_max_drawdown(equity_curve, n_bars);
    result->total_turnover = total_turnover;
    result->total_fees = total_fees;
    result->total_borrow_cost = total_borrow;
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
