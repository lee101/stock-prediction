/*
 * trading_env.c  –  Pure C market-simulation environment for PufferLib RL.
 *
 * Reads pre-exported binary data (Chronos2 forecasts + OHLCV prices).
 * Each agent replays a random window through historical data, making
 * hourly buy/sell/hold decisions.  Reward = hourly portfolio-value change.
 */

#include "../include/trading_env.h"
#include <fcntl.h>
#include <sys/stat.h>

/* ------------------------------------------------------------------ */
/*  Data loading                                                       */
/* ------------------------------------------------------------------ */

MarketData* market_data_load(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "market_data_load: cannot open %s\n", path);
        return NULL;
    }

    /* get file size */
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    /* read entire file into a buffer */
    void* buf = malloc(file_size);
    if (!buf) { fclose(fp); return NULL; }
    if ((long)fread(buf, 1, file_size, fp) != file_size) {
        free(buf); fclose(fp); return NULL;
    }
    fclose(fp);

    /* parse header */
    DataHeader* hdr = (DataHeader*)buf;
    if (memcmp(hdr->magic, "MKTD", 4) != 0) {
        fprintf(stderr, "market_data_load: bad magic in %s\n", path);
        free(buf); return NULL;
    }

    MarketData* md = (MarketData*)calloc(1, sizeof(MarketData));
    md->num_symbols   = (int)hdr->num_symbols;
    md->num_timesteps = (int)hdr->num_timesteps;
    md->file_buf      = buf;
    md->file_size     = (size_t)file_size;

    if (md->num_symbols > MAX_SYMBOLS) {
        fprintf(stderr, "market_data_load: too many symbols %d (max %d)\n",
                md->num_symbols, MAX_SYMBOLS);
        free(buf); free(md); return NULL;
    }

    /* symbol names */
    char* ptr = (char*)buf + HEADER_SIZE;
    for (int i = 0; i < md->num_symbols; i++) {
        memcpy(md->sym_names[i], ptr + i * SYM_NAME_LEN, SYM_NAME_LEN);
        md->sym_names[i][SYM_NAME_LEN - 1] = '\0';
    }
    ptr += md->num_symbols * SYM_NAME_LEN;

    /* feature array */
    md->features = (float*)ptr;
    ptr += (size_t)md->num_timesteps * md->num_symbols * FEATURES_PER_SYM * sizeof(float);

    /* price array */
    md->prices = (float*)ptr;
    ptr += (size_t)md->num_timesteps * md->num_symbols * PRICE_FEATS * sizeof(float);

    /* optional tradable mask (uint8[T][S]) appended after prices (v2+) */
    md->tradable = NULL;
    size_t mask_bytes = (size_t)md->num_timesteps * md->num_symbols * sizeof(unsigned char);
    if (hdr->version >= 2) {
        if ((size_t)(ptr - (char*)buf) + mask_bytes <= md->file_size) {
            md->tradable = (unsigned char*)ptr;
        } else {
            fprintf(stderr, "market_data_load: v%u missing tradable mask in %s\n",
                    hdr->version, path);
            free(buf); free(md); return NULL;
        }
    } else {
        if ((size_t)(ptr - (char*)buf) + mask_bytes <= md->file_size) {
            /* Allow a mask to be present even for v1 files, for forward compatibility. */
            md->tradable = (unsigned char*)ptr;
        }
    }

    fprintf(stderr, "market_data_load: %d symbols, %d timesteps from %s\n",
            md->num_symbols, md->num_timesteps, path);
    return md;
}

void market_data_free(MarketData* md) {
    if (!md) return;
    if (md->file_buf) free(md->file_buf);
    free(md);
}

/* ------------------------------------------------------------------ */
/*  Observation                                                        */
/* ------------------------------------------------------------------ */

static void fill_observations(TradingEnv* env) {
    const MarketData* md = env->data;
    const AgentState* ag = &env->agent;
    const int S = md->num_symbols;
    int t = ag->data_offset + ag->step;

    /* clamp to valid range */
    if (t < 0) t = 0;
    if (t >= md->num_timesteps) t = md->num_timesteps - 1;

    /* per-symbol features */
    memcpy(env->observations,
           &md->features[t * S * FEATURES_PER_SYM],
           S * FEATURES_PER_SYM * sizeof(float));

    /* portfolio state */
    int base = S * FEATURES_PER_SYM;
    float pos_val = 0.0f;
    float unreal_pnl = 0.0f;

    if (ag->position_sym >= 0) {
        int sym = ag->position_sym % S;
        float cur_price = get_price(md, t, sym, P_CLOSE);
        int is_short = (ag->position_sym >= S);
        if (is_short) {
            /* Short position is a liability: equity = cash - qty*price.
               cash already includes the short-sale proceeds from open_short(). */
            pos_val = -(ag->position_qty * cur_price);
            unreal_pnl = ag->position_qty * (ag->entry_price - cur_price);
        } else {
            pos_val = ag->position_qty * cur_price;
            unreal_pnl = ag->position_qty * (cur_price - ag->entry_price);
        }
    }

    env->observations[base + 0] = ag->cash / INITIAL_CASH;
    env->observations[base + 1] = pos_val / INITIAL_CASH;
    env->observations[base + 2] = unreal_pnl / INITIAL_CASH;
    env->observations[base + 3] = (float)ag->hold_hours / (float)(env->max_steps > 0 ? env->max_steps : 1);
    env->observations[base + 4] = (float)ag->step / (float)(env->max_steps > 0 ? env->max_steps : 1);

    /* one-hot position encoding */
    for (int i = 0; i < S; i++) {
        env->observations[base + 5 + i] = 0.0f;
    }
    if (ag->position_sym >= 0 && ag->position_sym < S) {
        env->observations[base + 5 + ag->position_sym] = 1.0f;
    } else if (ag->position_sym >= S && ag->position_sym < 2 * S) {
        env->observations[base + 5 + (ag->position_sym - S)] = -1.0f;
    }
}

/* ------------------------------------------------------------------ */
/*  Portfolio valuation                                                */
/* ------------------------------------------------------------------ */

static float compute_equity(const TradingEnv* env, int t) {
    const AgentState* ag = &env->agent;
    const MarketData* md = env->data;
    const int S = md->num_symbols;
    float equity = ag->cash;

    if (ag->position_sym >= 0) {
        int sym = ag->position_sym % S;
        float cur_price = get_price(md, t, sym, P_CLOSE);
        int is_short = (ag->position_sym >= S);
        if (is_short) {
            /* Short position equity: cash - qty*price.
               (cash already contains the short-sale proceeds.) */
            equity -= ag->position_qty * cur_price;
        } else {
            equity += ag->position_qty * cur_price;
        }
    }
    return equity;
}

/* ------------------------------------------------------------------ */
/*  Trade execution                                                    */
/* ------------------------------------------------------------------ */

static void close_position(TradingEnv* env, int t) {
    AgentState* ag = &env->agent;
    const MarketData* md = env->data;
    const int S = md->num_symbols;

    if (ag->position_sym < 0) return;

    int sym = ag->position_sym % S;
    float cur_price = get_price(md, t, sym, P_CLOSE);
    int is_short = (ag->position_sym >= S);

    if (is_short) {
        /* Buy back borrowed shares. Note: cash includes the short-sale proceeds from open_short(). */
        float cost = ag->position_qty * cur_price * (1.0f + env->fee_rate);
        ag->cash -= cost;
    } else {
        float proceeds = ag->position_qty * cur_price * (1.0f - env->fee_rate);
        ag->cash += proceeds;
    }

    float pnl;
    if (is_short) {
        pnl = ag->entry_price - cur_price; /* per share */
    } else {
        pnl = cur_price - ag->entry_price;
    }
    if (pnl > 0) ag->winning_trades++;
    ag->num_trades++;

    ag->position_sym = -1;
    ag->position_qty = 0.0f;
    ag->entry_price = 0.0f;
    ag->hold_hours = 0;
}

static void open_long(TradingEnv* env, int t, int sym) {
    AgentState* ag = &env->agent;
    const MarketData* md = env->data;
    float price = get_price(md, t, sym, P_CLOSE);

    if (price <= 0.0f || ag->cash <= 0.0f) return;

    float buy_budget = ag->cash * env->max_leverage;
    float qty = buy_budget / (price * (1.0f + env->fee_rate));
    float cost = qty * price * (1.0f + env->fee_rate);

    ag->cash -= cost;
    ag->position_sym = sym;
    ag->position_qty = qty;
    ag->entry_price = price;
    ag->hold_hours = 0;
}

static void open_short(TradingEnv* env, int t, int sym) {
    AgentState* ag = &env->agent;
    const MarketData* md = env->data;
    const int S = md->num_symbols;
    float price = get_price(md, t, sym, P_CLOSE);

    if (price <= 0.0f || ag->cash <= 0.0f) return;

    float sell_budget = ag->cash * env->max_leverage;
    float qty = sell_budget / (price * (1.0f + env->fee_rate));

    /* short proceeds go to cash (minus borrow fee) */
    ag->cash += qty * price * (1.0f - env->fee_rate);
    ag->position_sym = S + sym;  /* S..2S-1 = short */
    ag->position_qty = qty;
    ag->entry_price = price;
    ag->hold_hours = 0;
}

/* ------------------------------------------------------------------ */
/*  Core PufferLib functions                                           */
/* ------------------------------------------------------------------ */

void c_reset(TradingEnv* env) {
    AgentState* ag = &env->agent;
    const MarketData* md = env->data;

    /* random starting offset (leave room for max_steps) */
    int max_offset = md->num_timesteps - env->max_steps - 1;
    if (max_offset < 0) max_offset = 0;
    ag->data_offset = (max_offset > 0) ? (rand() % max_offset) : 0;
    ag->step = 0;

    ag->cash = INITIAL_CASH;
    ag->position_sym = -1;
    ag->position_qty = 0.0f;
    ag->entry_price = 0.0f;
    ag->hold_hours = 0;
    ag->peak_equity = INITIAL_CASH;
    ag->initial_equity = INITIAL_CASH;
    ag->num_trades = 0;
    ag->winning_trades = 0;
    ag->sum_neg_sq = 0.0f;
    ag->sum_ret = 0.0f;
    ag->ret_count = 0;

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    fill_observations(env);
}

void c_step(TradingEnv* env) {
    AgentState* ag = &env->agent;
    const MarketData* md = env->data;
    const int S = md->num_symbols;
    int action = env->actions[0];
    int t = ag->data_offset + ag->step;

    /* clamp timestep */
    if (t >= md->num_timesteps) t = md->num_timesteps - 1;

    /* current position info */
    int cur_sym = -1;
    int cur_tradable = 1;
    if (ag->position_sym >= 0) {
        cur_sym = ag->position_sym % S;
        cur_tradable = is_tradable(md, t, cur_sym);
    }

    /* compute equity before action */
    float equity_before = compute_equity(env, t);

    /* decode action:
     *   0         = go flat (close position)
     *   1..S      = go long symbol (action-1)
     *   S+1..2S   = go short symbol (action-S-1)
     */
    if (action == 0) {
        /* close any open position */
        if (ag->position_sym >= 0) {
            if (cur_tradable) {
                close_position(env, t);
            } else {
                /* can't close when market closed: treat as hold */
                ag->hold_hours++;
            }
        }
    } else if (action >= 1 && action <= S) {
        int target_sym = action - 1;
        int target_tradable = is_tradable(md, t, target_sym);
        /* if already in this long position, hold */
        if (ag->position_sym == target_sym) {
            ag->hold_hours++;
        } else {
            /* If target market is closed, do nothing (avoid accidental closes). */
            if (!target_tradable) {
                if (ag->position_sym >= 0) ag->hold_hours++;
            } else if (ag->position_sym >= 0 && !cur_tradable) {
                /* can't close current position right now */
                ag->hold_hours++;
            } else {
                /* close current, open new long */
                if (ag->position_sym >= 0) close_position(env, t);
                open_long(env, t, target_sym);
            }
        }
    } else if (action >= S + 1 && action <= 2 * S) {
        int target_sym = action - S - 1;
        int short_id = S + target_sym;
        int target_tradable = is_tradable(md, t, target_sym);
        /* if already in this short position, hold */
        if (ag->position_sym == short_id) {
            ag->hold_hours++;
        } else {
            if (!target_tradable) {
                if (ag->position_sym >= 0) ag->hold_hours++;
            } else if (ag->position_sym >= 0 && !cur_tradable) {
                ag->hold_hours++;
            } else {
                if (ag->position_sym >= 0) close_position(env, t);
                open_short(env, t, target_sym);
            }
        }
    }
    /* else: invalid action, treat as hold */

    /* advance time */
    ag->step++;
    int t_new = ag->data_offset + ag->step;
    if (t_new >= md->num_timesteps) t_new = md->num_timesteps - 1;

    /* compute equity after market move */
    float equity_after = compute_equity(env, t_new);

    /* reward = percentage return (clipped) */
    float ret = 0.0f;
    if (equity_before > 1e-6f) {
        ret = (equity_after - equity_before) / equity_before;
    }

    /* track for sortino */
    ag->sum_ret += ret;
    ag->ret_count++;
    if (ret < 0.0f) {
        ag->sum_neg_sq += ret * ret;
    }

    /* update peak for drawdown */
    if (equity_after > ag->peak_equity) {
        ag->peak_equity = equity_after;
    }

    /* reward shaping (configurable via env params) */
    float reward = ret * env->reward_scale;

    /* clip extreme rewards to prevent gradient explosion */
    float clip = env->reward_clip;
    if (reward > clip) reward = clip;
    if (reward < -clip) reward = -clip;

    /* small penalty for sitting in cash (opportunity cost) */
    if (ag->position_sym < 0) {
        reward -= env->cash_penalty;
    }

    /* drawdown penalty: penalize equity dropping below peak */
    if (env->drawdown_penalty > 0.0f && ag->peak_equity > 0.0f) {
        float dd = (ag->peak_equity - equity_after) / ag->peak_equity;
        if (dd > 0.0f) {
            reward -= env->drawdown_penalty * dd;
        }
    }

    env->rewards[0] = reward;

    /* check terminal */
    int done = (ag->step >= env->max_steps) ||
               (t_new >= md->num_timesteps - 1) ||
               (equity_after < INITIAL_CASH * 0.01f);  /* bankrupt */

    if (done) {
        /* close position for final accounting */
        if (ag->position_sym >= 0) {
            close_position(env, t_new);
        }
        float final_equity = ag->cash;
        float total_return = (final_equity - ag->initial_equity) / ag->initial_equity;
        float max_dd = (ag->peak_equity > 0.0f)
            ? (ag->peak_equity - equity_after) / ag->peak_equity
            : 0.0f;

        /* sortino ratio */
        float sortino = 0.0f;
        if (ag->ret_count > 1 && ag->sum_neg_sq > 0.0f) {
            float mean_ret = ag->sum_ret / ag->ret_count;
            float downside_dev = sqrtf(ag->sum_neg_sq / ag->ret_count);
            float ppy = env->periods_per_year;
            if (ppy <= 0.0f) ppy = 8760.0f;
            sortino = mean_ret / downside_dev * sqrtf(ppy);  /* annualised */
        }

        env->log.total_return += total_return;
        env->log.sortino += sortino;
        env->log.max_drawdown += max_dd;
        env->log.num_trades += (float)ag->num_trades;
        env->log.win_rate += (ag->num_trades > 0)
            ? (float)ag->winning_trades / ag->num_trades
            : 0.0f;
        env->log.avg_hold_hours += (ag->num_trades > 0)
            ? (float)ag->step / ag->num_trades
            : 0.0f;
        env->log.n += 1.0f;

        env->terminals[0] = 1;

        /* auto-reset for next episode */
        c_reset(env);
    } else {
        env->terminals[0] = 0;
        fill_observations(env);
    }
}

void c_close(TradingEnv* env) {
    /* env does not own data - caller must free MarketData separately */
    (void)env;
}
