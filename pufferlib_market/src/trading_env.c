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
#include <stdint.h>

/* Fast inline PRNG (xorshift64*) to replace glibc rand() in the hotpath.
 * glibc rand() is ~10x slower than this and goes through a PLT indirection;
 * xorshift64* inlines to ~5 instructions and has better statistical quality
 * than LCG rand() for sim-fill gating and episode-offset randomization. */
static __thread uint64_t g_trading_rng_state = 0x9E3779B97F4A7C15ULL;
static inline uint64_t fast_rand_u64(void) {
    uint64_t x = g_trading_rng_state;
    if (UNLIKELY(x == 0)) x = 0x9E3779B97F4A7C15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_trading_rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static inline float fast_rand_float(void) {
    /* uniform [0,1) using top 24 bits -> float mantissa */
    return (float)((fast_rand_u64() >> 40) * (1.0f / 16777216.0f));
}
static inline int fast_rand_int(int n) {
    if (n <= 1) return 0;
    /* Lemire's nearly-divisionless bounded int */
    return (int)(((fast_rand_u64() >> 32) * (uint64_t)n) >> 32);
}

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
    md->any_tradable  = NULL;

    /* Read features_per_sym from header (v3+); fall back to FEATURES_PER_SYM for v1/v2. */
    if (hdr->features_per_sym > 0) {
        md->features_per_sym = (int)hdr->features_per_sym;
    } else {
        md->features_per_sym = FEATURES_PER_SYM;
    }

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
    /* cppcheck-suppress invalidPointerCast ; file buffer is aligned by mmap/malloc */
    md->features = (float*)(void*)ptr;
    ptr += (size_t)md->num_timesteps * md->num_symbols * md->features_per_sym * sizeof(float);

    /* price array */
    /* cppcheck-suppress invalidPointerCast ; file buffer is aligned by mmap/malloc */
    md->prices = (float*)(void*)ptr;
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

    if (md->tradable != NULL) {
        md->any_tradable = (unsigned char*)malloc((size_t)md->num_timesteps);
        if (!md->any_tradable) {
            fprintf(stderr, "market_data_load: failed to allocate any_tradable cache\n");
            free(buf); free(md); return NULL;
        }
        for (int t = 0; t < md->num_timesteps; t++) {
            const unsigned char* row = md->tradable + ((size_t)t * md->num_symbols);
            unsigned char any_open = 0;
            for (int s = 0; s < md->num_symbols; s++) {
                if (row[s] != 0) {
                    any_open = 1;
                    break;
                }
            }
            md->any_tradable[t] = any_open;
        }
    }

    fprintf(stderr, "market_data_load: %d symbols, %d timesteps from %s\n",
            md->num_symbols, md->num_timesteps, path);
    return md;
}

void market_data_free(MarketData* md) {
    if (!md) return;
    if (md->any_tradable) free(md->any_tradable);
    if (md->file_buf) free(md->file_buf);
    free(md);
}

/* ------------------------------------------------------------------ */
/*  Observation                                                        */
/* ------------------------------------------------------------------ */

static void fill_observations(TradingEnv* __restrict__ env) {
    const MarketData* __restrict__ md = env->data;
    const AgentState* __restrict__ ag = &env->agent;
    const int S = md->num_symbols;
    int t = ag->data_offset + ag->step;

    /* clamp to valid range */
    if (UNLIKELY(t < 0)) t = 0;
    if (UNLIKELY(t >= md->num_timesteps)) t = md->num_timesteps - 1;

    /* Observation lag: agent sees features from t-lag to prevent look-ahead bias.
       lag=1 matches the prior hardcoded behavior. Production uses lag>=2
       because Chronos2 forecast + Gemini call + order placement straddles
       more than one bar. */
    int lag = env->decision_lag >= 1 ? env->decision_lag : 1;
    int t_obs = t - lag;
    if (UNLIKELY(t_obs < 0)) t_obs = 0;

    const int F = md->features_per_sym;
    const float* __restrict__ feat_src = &md->features[t_obs * S * F];
    __builtin_prefetch(feat_src + S * F, 0, 1);

    /* per-symbol features (lagged by 1 bar) — compiler will vectorize with AVX2 */
    float* __restrict__ obs_dst = env->observations;
    memcpy(obs_dst, feat_src, (size_t)S * F * sizeof(float));

    /* portfolio state uses LAGGED prices (t-1) for position valuation shown
       to the agent, matching what a real trader would see before bar t closes */
    int base = S * F;
    float pos_val = 0.0f;
    float unreal_pnl = 0.0f;

    if (ag->position_sym >= 0) {
        int sym = ag->position_sym % S;
        float cur_price = get_price(md, t_obs, sym, P_CLOSE);
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

    obs_dst[base + 0] = ag->cash / INITIAL_CASH;
    obs_dst[base + 1] = pos_val / INITIAL_CASH;
    obs_dst[base + 2] = unreal_pnl / INITIAL_CASH;
    obs_dst[base + 3] = (float)ag->hold_hours / (float)(env->max_steps > 0 ? env->max_steps : 1);
    obs_dst[base + 4] = (float)ag->step / (float)(env->max_steps > 0 ? env->max_steps : 1);

    /* one-hot position encoding — zero then set hot entry */
    memset(obs_dst + base + 5, 0, (size_t)S * sizeof(float));
    if (ag->position_sym >= 0 && ag->position_sym < S) {
        obs_dst[base + 5 + ag->position_sym] = 1.0f;
    } else if (ag->position_sym >= S && ag->position_sym < 2 * S) {
        obs_dst[base + 5 + (ag->position_sym - S)] = -1.0f;
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

static float borrow_fee(const TradingEnv* env, int t) {
    const AgentState* ag = &env->agent;
    const MarketData* md = env->data;
    const int S = md->num_symbols;
    if (env->short_borrow_apr <= 0.0f || ag->position_qty <= 0.0f) {
        return 0.0f;
    }
    float ppy = env->periods_per_year;
    if (ppy <= 0.0f) ppy = 8760.0f;

    int sym = ag->position_sym % S;
    float cur_price = get_price(md, t, sym, P_CLOSE);
    if (cur_price <= 0.0f) return 0.0f;

    if (ag->position_sym >= S) {
        /* Short position: charge on full notional */
        return ag->position_qty * cur_price * env->short_borrow_apr / ppy;
    } else if (env->max_leverage > 1.0f) {
        /* Leveraged long: charge on borrowed portion only.
           borrowed = position_value - cash_at_entry ≈ position_value * (1 - 1/leverage) */
        float pos_val = ag->position_qty * cur_price;
        float borrowed = pos_val * (1.0f - 1.0f / env->max_leverage);
        if (borrowed <= 0.0f) return 0.0f;
        return borrowed * env->short_borrow_apr / ppy;
    }
    return 0.0f;
}

static int should_drawdown_profit_early_exit(
    const TradingEnv* env,
    float equity_after,
    float* out_progress,
    float* out_total_return
) {
    const AgentState* ag = &env->agent;
    int total_steps = env->max_steps;
    int min_steps = env->drawdown_profit_early_exit_min_steps;
    float progress_fraction = env->drawdown_profit_early_exit_progress_fraction;

    if (out_progress) *out_progress = 0.0f;
    if (out_total_return) *out_total_return = 0.0f;

    if (!env->enable_drawdown_profit_early_exit) {
        return 0;
    }
    if (total_steps < 2) {
        return 0;
    }
    if (total_steps < min_steps) {
        return 0;
    }
    if (!isfinite(ag->initial_equity) || ag->initial_equity <= 0.0f) {
        return 0;
    }

    float progress = (float)(ag->step + 1) / (float)total_steps;
    if (progress > 1.0f) progress = 1.0f;
    if (out_progress) *out_progress = progress;
    if (progress < progress_fraction) {
        return 0;
    }

    float total_return = (equity_after - ag->initial_equity) / ag->initial_equity;
    if (out_total_return) *out_total_return = total_return;
    return ag->max_drawdown > total_return;
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
    /* Use OPEN price for closing — order placed at start of bar */
    float cur_price = get_price(md, t, sym, P_OPEN);
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

static int resolve_limit_fill_price(const TradingEnv* env, const MarketData* md, int t, int sym, float target_price, int is_buy, float* fill_price) {
    float low = get_price(md, t, sym, P_LOW);
    float high = get_price(md, t, sym, P_HIGH);
    if (low > high) {
        float tmp = low;
        low = high;
        high = tmp;
    }
    if (target_price < low || target_price > high) {
        return 0;
    }
    /* Apply adverse slippage: buys fill higher, sells fill lower */
    float slippage = env->fill_slippage_bps / 10000.0f;
    if (is_buy) {
        *fill_price = target_price * (1.0f + slippage);
        if (*fill_price > high) return 0;  /* slipped out of range */
    } else {
        *fill_price = target_price * (1.0f - slippage);
        if (*fill_price < low) return 0;   /* slipped out of range */
    }
    return 1;
}

static float action_allocation_pct(const TradingEnv* env, int alloc_idx) {
    int bins = env->action_allocation_bins;
    if (bins <= 1) return 1.0f;
    if (alloc_idx < 0) alloc_idx = 0;
    if (alloc_idx >= bins) alloc_idx = bins - 1;
    float pct = (float)(alloc_idx + 1) / (float)bins;
    if (pct < 0.01f) pct = 0.01f;
    if (pct > 1.0f) pct = 1.0f;
    return pct;
}

static float action_level_offset_bps(const TradingEnv* env, int level_idx) {
    int bins = env->action_level_bins;
    float max_bps = env->action_max_offset_bps;
    if (bins <= 1 || max_bps <= 0.0f) return 0.0f;
    if (level_idx < 0) level_idx = 0;
    if (level_idx >= bins) level_idx = bins - 1;
    float frac = (float)level_idx / (float)(bins - 1);  /* [0,1] */
    return (2.0f * frac - 1.0f) * max_bps;              /* [-max,+max] */
}

static void open_long(TradingEnv* env, int t, int sym, float allocation_pct, float level_offset_bps) {
    AgentState* ag = &env->agent;
    const MarketData* md = env->data;
    /* Use OPEN price as reference since order is placed at bar start */
    float ref_price = get_price(md, t, sym, P_OPEN);

    if (ref_price <= 0.0f || ag->cash <= 0.0f) return;

    float target_price = ref_price * (1.0f + level_offset_bps / 10000.0f);
    float fill_price = 0.0f;
    if (target_price <= 0.0f || !resolve_limit_fill_price(env, md, t, sym, target_price, /*is_buy=*/1, &fill_price)) {
        return;
    }

    if (allocation_pct <= 0.0f) return;
    if (allocation_pct > 1.0f) allocation_pct = 1.0f;
    float buy_budget = ag->cash * env->max_leverage * allocation_pct;
    if (buy_budget <= 0.0f) return;

    float denom = fill_price * (1.0f + env->fee_rate);
    if (denom <= 0.0f) return;
    float qty = buy_budget / denom;
    if (qty <= 0.0f) return;
    float cost = qty * denom;
    if (cost <= 0.0f) return;

    ag->cash -= cost;
    ag->position_sym = sym;
    ag->position_qty = qty;
    ag->entry_price = fill_price;
    ag->hold_hours = 0;
}

static void open_short(TradingEnv* env, int t, int sym, float allocation_pct, float level_offset_bps) {
    AgentState* ag = &env->agent;
    const MarketData* md = env->data;
    const int S = md->num_symbols;
    /* Use OPEN price as reference since order is placed at bar start */
    float ref_price = get_price(md, t, sym, P_OPEN);

    if (ref_price <= 0.0f || ag->cash <= 0.0f) return;

    float target_price = ref_price * (1.0f + level_offset_bps / 10000.0f);
    float fill_price = 0.0f;
    if (target_price <= 0.0f || !resolve_limit_fill_price(env, md, t, sym, target_price, /*is_buy=*/0, &fill_price)) {
        return;
    }

    if (allocation_pct <= 0.0f) return;
    if (allocation_pct > 1.0f) allocation_pct = 1.0f;
    float sell_budget = ag->cash * env->max_leverage * allocation_pct;
    if (sell_budget <= 0.0f) return;

    float denom = fill_price * (1.0f + env->fee_rate);
    if (denom <= 0.0f) return;
    float qty = sell_budget / denom;
    if (qty <= 0.0f) return;

    /* short proceeds go to cash (minus fee) */
    ag->cash += qty * fill_price * (1.0f - env->fee_rate);
    ag->position_sym = S + sym;  /* S..2S-1 = short */
    ag->position_qty = qty;
    ag->entry_price = fill_price;
    ag->hold_hours = 0;
}

/* ------------------------------------------------------------------ */
/*  Core PufferLib functions                                           */
/* ------------------------------------------------------------------ */

static void reset_agent_state(TradingEnv* env) {
    AgentState* ag = &env->agent;
    const MarketData* md = env->data;

    /* Starting offset: use forced_offset if set, otherwise random. */
    int max_offset = md->num_timesteps - env->max_steps - 1;
    if (env->forced_offset >= 0) {
        ag->data_offset = env->forced_offset;
        if (ag->data_offset > max_offset && max_offset >= 0)
            ag->data_offset = max_offset;
    } else if (max_offset <= 0) {
        ag->data_offset = 0;
    } else {
        ag->data_offset = fast_rand_int(max_offset + 1);
    }
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
    ag->prev_ret = 0.0f;
    ag->max_drawdown = 0.0f;
}

void c_reset(TradingEnv* env) {
    reset_agent_state(env);

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    fill_observations(env);
}

__attribute__((hot)) void c_step(TradingEnv* env) {
    AgentState* ag = &env->agent;
    const MarketData* md = env->data;
    const int S = md->num_symbols;
    int action = env->actions[0];
    int t = ag->data_offset + ag->step;
    int trade_events = 0;

    /* Clear terminal flag; if episode ends this step, we'll set it to 1 below.
       This is required so the vectorized wrapper can see terminal transitions. */
    env->terminals[0] = 0;

    /* clamp timestep */
    if (UNLIKELY(t >= md->num_timesteps)) t = md->num_timesteps - 1;

    /* Force-close if max_hold_hours exceeded (before action decode) */
    if (UNLIKELY(env->max_hold_hours > 0 && ag->position_sym >= 0 &&
        ag->hold_hours >= env->max_hold_hours)) {
        int cs = ag->position_sym % S;
        if (is_tradable(md, t, cs)) {
            close_position(env, t);
            trade_events += 1;
        }
        /* If not tradable, will close next tradable step */
    }

    /* current position info */
    int cur_tradable = 1;
    if (ag->position_sym >= 0) {
        cur_tradable = is_tradable(md, t, ag->position_sym % S);
    }
    int any_tradable = (md->any_tradable != NULL) ? (md->any_tradable[t] != 0) : 1;

    __builtin_prefetch(&md->prices[t * S * PRICE_FEATS], 0, 1);

    /* compute equity before action */
    float equity_before = compute_equity(env, t);

    int alloc_bins = env->action_allocation_bins;
    int level_bins = env->action_level_bins;
    if (alloc_bins < 1) alloc_bins = 1;
    if (level_bins < 1) level_bins = 1;
    int per_symbol_actions = alloc_bins * level_bins;
    int side_block = S * per_symbol_actions;

    /* decode action:
     *   0            = go flat (close position)
     *   1..side_block         = long actions
     *   side_block+1..2*side_block = short actions
     * with each side action decoded into (symbol, allocation_bin, level_bin).
     */
    if (action == 0) {
        /* close any open position */
        if (ag->position_sym >= 0) {
            if (cur_tradable) {
                close_position(env, t);
                trade_events += 1;
            } else {
                /* can't close when market closed: treat as hold */
                ag->hold_hours++;
            }
        }
    } else if (action >= 1 && action <= 2 * side_block) {
        int action_idx = action - 1;
        int is_short_target = (action_idx >= side_block);
        if (is_short_target) action_idx -= side_block;

        int target_sym = action_idx / per_symbol_actions;
        int rem = action_idx % per_symbol_actions;
        int alloc_idx = rem / level_bins;
        int level_idx = rem % level_bins;
        float target_alloc = action_allocation_pct(env, alloc_idx);
        float level_bps = action_level_offset_bps(env, level_idx);

        int target_pos_id = is_short_target ? (S + target_sym) : target_sym;
        int target_tradable = is_tradable(md, t, target_sym);

        /* if already in this exact position, hold */
        if (ag->position_sym == target_pos_id) {
            ag->hold_hours++;
        } else {
            /* If target market is closed, do nothing (avoid accidental closes). */
            if (!target_tradable) {
                if (ag->position_sym >= 0) ag->hold_hours++;
            } else if (ag->position_sym >= 0 && !cur_tradable) {
                /* can't close current position right now */
                ag->hold_hours++;
            } else {
                if (ag->position_sym >= 0) {
                    close_position(env, t);
                    trade_events += 1;
                }
                /* fill_probability gate: simulate unfilled orders due to low liquidity */
                int fill_ok = 1;
                if (env->fill_probability < 1.0f) {
                    float roll = fast_rand_float();
                    fill_ok = (roll <= env->fill_probability);
                }
                if (fill_ok) {
                    if (is_short_target) {
                        open_short(env, t, target_sym, target_alloc, level_bps);
                    } else {
                        open_long(env, t, target_sym, target_alloc, level_bps);
                    }
                    if (ag->position_sym == target_pos_id) {
                        trade_events += 1;
                    }
                }
                /* else: order not filled, agent stays flat this step */
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
    float bfee = borrow_fee(env, t_new);
    if (bfee > 0.0f) {
        ag->cash -= bfee;
        equity_after -= bfee;
    }

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
    float current_drawdown = 0.0f;
    if (ag->peak_equity > 0.0f) {
        current_drawdown = (ag->peak_equity - equity_after) / ag->peak_equity;
        if (current_drawdown > ag->max_drawdown) {
            ag->max_drawdown = current_drawdown;
        }
    }

    /* reward shaping (configurable via env params) */
    float reward = ret * env->reward_scale;

    /* clip extreme rewards to prevent gradient explosion */
    float clip = env->reward_clip;
    if (reward > clip) reward = clip;
    if (reward < -clip) reward = -clip;

    /* small penalty for sitting in cash (opportunity cost) */
    if (ag->position_sym < 0 && any_tradable) {
        reward -= env->cash_penalty;
    }

    /* drawdown penalty: penalize equity dropping below peak */
    if (env->drawdown_penalty > 0.0f && current_drawdown > 0.0f) {
        reward -= env->drawdown_penalty * current_drawdown;
    }

    /* downside penalty: penalize negative returns more than positive returns.
       This encourages higher Sortino by shrinking downside deviation. */
    if (env->downside_penalty > 0.0f && ret < 0.0f) {
        reward -= env->downside_penalty * (ret * ret);
    }

    /* smooth downside penalty: uses a softplus approximation of max(0, -ret)
       so the shaping term is differentiable around zero returns. */
    if (env->smooth_downside_penalty > 0.0f) {
        float temp = env->smooth_downside_temperature;
        if (temp <= 1e-6f) temp = 1e-6f;
        float x = -ret / temp;
        float softplus_x;
        if (x > 20.0f) {
            softplus_x = x;
        } else if (x < -20.0f) {
            softplus_x = expf(x);
        } else {
            softplus_x = log1pf(expf(x));
        }
        float smooth_neg = temp * softplus_x;
        reward -= env->smooth_downside_penalty * (smooth_neg * smooth_neg);
    }

    /* trade penalty: discourage churn (opens/closes) without changing accounting. */
    if (env->trade_penalty > 0.0f && trade_events > 0) {
        reward -= env->trade_penalty * (float)trade_events;
    }

    /* smoothness penalty: penalize large changes in returns (encourages steady PnL curve) */
    if (env->smoothness_penalty > 0.0f && ag->ret_count > 0) {
        float ret_diff = ret - ag->prev_ret;
        reward -= env->smoothness_penalty * (ret_diff * ret_diff);
    }

    int early_exit = 0;
    float early_exit_progress = 0.0f;
    float early_exit_total_return = 0.0f;
    if (should_drawdown_profit_early_exit(
        env,
        equity_after,
        &early_exit_progress,
        &early_exit_total_return
    )) {
        early_exit = 1;
    }
    ag->prev_ret = ret;
    env->rewards[0] = reward;

    /* check terminal */
    int done = (ag->step >= env->max_steps) ||
               (t_new >= md->num_timesteps - 1) ||
               early_exit ||
               (equity_after < INITIAL_CASH * 0.01f);  /* bankrupt */

    if (UNLIKELY(done)) {
        if (early_exit && env->drawdown_profit_early_exit_verbose) {
            fprintf(
                stderr,
                "[pufferlib_market.binding] early stopping at %.1f%%: max drawdown %.2f%% exceeds profit %.2f%%.\n",
                early_exit_progress * 100.0f,
                ag->max_drawdown * 100.0f,
                early_exit_total_return * 100.0f
            );
        }
        /* close position for final accounting */
        if (ag->position_sym >= 0) {
            close_position(env, t_new);
        }
        float final_equity = ag->cash;
        float total_return = (final_equity - ag->initial_equity) / ag->initial_equity;
        float max_dd = ag->max_drawdown;

        /* sortino ratio */
        float sortino = 0.0f;
        if (ag->ret_count > 1 && ag->sum_neg_sq > 0.0f) {
            float mean_ret = ag->sum_ret / ag->ret_count;
            float downside_dev = sqrtf(ag->sum_neg_sq / ag->ret_count);
            float ppy = env->periods_per_year;
            if (ppy <= 0.0f) ppy = 8760.0f;
            if (downside_dev > 1e-8f) {
                sortino = mean_ret / downside_dev * sqrtf(ppy);  /* annualised */
            }
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

        /* auto-reset for next episode: reset the agent state + refresh observations,
           but do NOT clear terminals/rewards within this step. */
        reset_agent_state(env);
        fill_observations(env);
    } else {
        int t_next = t_new + 1;
        if (LIKELY(t_next < md->num_timesteps)) {
            __builtin_prefetch(&md->features[t_next * S * md->features_per_sym], 0, 1);
            __builtin_prefetch(&md->prices[t_next * S * PRICE_FEATS], 0, 1);
        }
        fill_observations(env);
    }
}

void c_close(TradingEnv* env) {
    /* env does not own data - caller must free MarketData separately */
    (void)env;
}
