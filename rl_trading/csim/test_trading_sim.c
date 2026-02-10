#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "trading_sim.h"

#define N_SYM 3
#define N_BARS 100
#define N_FEAT 5
#define EPS_LEN 20
#define ASSERTF(cond, fmt, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); assert(0); } \
} while(0)

static float g_close[N_SYM * N_BARS];
static float g_high[N_SYM * N_BARS];
static float g_low[N_SYM * N_BARS];
static float g_features[N_SYM * N_BARS * N_FEAT];

static void fill_data(void) {
    for (int s = 0; s < N_SYM; s++) {
        for (int b = 0; b < N_BARS; b++) {
            float base = 100.0f + (float)s * 50.0f + (float)b * 0.1f;
            g_close[s * N_BARS + b] = base;
            g_high[s * N_BARS + b] = base + 1.0f;
            g_low[s * N_BARS + b] = base - 1.0f;
            for (int f = 0; f < N_FEAT; f++) {
                g_features[(s * N_BARS + b) * N_FEAT + f] = (float)f * 0.01f + (float)b * 0.001f;
            }
        }
    }
}

static TradingSim make_env(void) {
    TradingSim env;
    memset(&env, 0, sizeof(env));
    env.n_symbols = N_SYM;
    env.n_bars = N_BARS;
    env.n_features = N_FEAT;
    env.obs_dim = N_SYM * N_FEAT + N_SYM + 3;
    env.close = g_close;
    env.high = g_high;
    env.low = g_low;
    env.features = g_features;
    env.initial_cash = 10000.0f;
    env.fee_rate = 0.0f;
    env.max_hold_bars = 6;
    env.episode_length = EPS_LEN;
    env.data_start = 5;
    env.data_end = N_BARS;
    allocate(&env);
    return env;
}

static void test_reset(void) {
    TradingSim env = make_env();
    c_reset(&env);
    ASSERTF(env.current_bar >= env.data_start, "bar=%d < start=%d", env.current_bar, env.data_start);
    ASSERTF(env.cash == 10000.0f, "cash=%f", env.cash);
    ASSERTF(env.holding_symbol == -1, "holding=%d", env.holding_symbol);
    ASSERTF(env.inventory == 0.0f, "inv=%f", env.inventory);
    ASSERTF(env.bars_held == 0, "bars_held=%d", env.bars_held);
    for (int i = 0; i < env.obs_dim; i++) {
        ASSERTF(isfinite(env.observations[i]), "obs[%d] not finite", i);
    }
    free_allocated(&env);
    printf("  PASS test_reset\n");
}

static void test_hold(void) {
    TradingSim env = make_env();
    c_reset(&env);
    float cash_before = env.cash;
    env.actions[0] = 0.0f;
    c_step(&env);
    ASSERTF(env.holding_symbol == -1, "should be flat");
    ASSERTF(env.cash == cash_before, "cash changed on hold");
    ASSERTF(env.trade_count == 0, "trades=%d", env.trade_count);
    free_allocated(&env);
    printf("  PASS test_hold\n");
}

static void test_buy_sell(void) {
    TradingSim env = make_env();
    env.fee_rate = 0.0f;
    c_reset(&env);
    env.start_bar = 10;
    env.current_bar = 10;
    env.prev_equity = env.initial_cash;
    compute_observations(&env);

    env.actions[0] = 1.0f;
    c_step(&env);
    ASSERTF(env.holding_symbol == 0, "should hold sym 0, got %d", env.holding_symbol);
    ASSERTF(env.inventory > 0.0f, "inv=%f", env.inventory);
    ASSERTF(env.cash == 0.0f, "cash=%f should be 0", env.cash);

    env.actions[0] = (float)(N_SYM + 1);
    c_step(&env);
    ASSERTF(env.holding_symbol == -1, "should be flat after sell");
    ASSERTF(env.cash > 0.0f, "cash=%f", env.cash);
    ASSERTF(env.trade_count == 1, "trades=%d", env.trade_count);
    free_allocated(&env);
    printf("  PASS test_buy_sell\n");
}

static void test_buy_fees(void) {
    TradingSim env = make_env();
    env.fee_rate = 0.001f;
    c_reset(&env);
    env.start_bar = 10;
    env.current_bar = 10;
    env.prev_equity = env.initial_cash;
    compute_observations(&env);
    float cash0 = env.cash;

    env.actions[0] = 1.0f;
    c_step(&env);
    ASSERTF(env.holding_symbol == 0, "hold sym 0");
    float buy_price = ts_get_close(&env, 0, 10);
    float expected_inv = (cash0 - cash0 * 0.001f) / buy_price;
    ASSERTF(fabsf(env.inventory - expected_inv) < 0.01f,
            "inv=%f expected=%f", env.inventory, expected_inv);

    env.actions[0] = (float)(N_SYM + 1);
    c_step(&env);
    ASSERTF(env.holding_symbol == -1, "flat");
    ASSERTF(env.cash < cash0, "should lose money to fees, cash=%f", env.cash);
    free_allocated(&env);
    printf("  PASS test_buy_fees\n");
}

static void test_switch_symbol(void) {
    TradingSim env = make_env();
    env.fee_rate = 0.0f;
    c_reset(&env);
    env.start_bar = 10;
    env.current_bar = 10;
    env.prev_equity = env.initial_cash;
    compute_observations(&env);

    env.actions[0] = 1.0f;
    c_step(&env);
    ASSERTF(env.holding_symbol == 0, "hold 0");

    env.actions[0] = 2.0f;
    c_step(&env);
    ASSERTF(env.holding_symbol == 1, "hold 1 after switch, got %d", env.holding_symbol);
    ASSERTF(env.trade_count == 1, "close of sym0 = 1 trade, got %d", env.trade_count);
    free_allocated(&env);
    printf("  PASS test_switch_symbol\n");
}

static void test_max_hold(void) {
    TradingSim env = make_env();
    env.fee_rate = 0.0f;
    env.max_hold_bars = 3;
    c_reset(&env);
    env.start_bar = 10;
    env.current_bar = 10;
    env.prev_equity = env.initial_cash;
    compute_observations(&env);

    env.actions[0] = 1.0f;
    c_step(&env);
    ASSERTF(env.holding_symbol == 0, "hold 0");
    env.actions[0] = 0.0f;
    c_step(&env);
    ASSERTF(env.holding_symbol == 0, "still hold after 2 bars");
    env.actions[0] = 0.0f;
    c_step(&env);
    ASSERTF(env.holding_symbol == -1, "force closed after 3 bars, got sym=%d held=%d",
            env.holding_symbol, env.bars_held);
    free_allocated(&env);
    printf("  PASS test_max_hold\n");
}

static void test_episode_truncation(void) {
    TradingSim env = make_env();
    env.episode_length = 5;
    c_reset(&env);
    int start = env.current_bar;
    for (int i = 0; i < 5; i++) {
        env.actions[0] = 0.0f;
        c_step(&env);
    }
    ASSERTF(env.current_bar != start + 5, "should have auto-reset");
    ASSERTF(env.log.n >= 1.0f, "log.n=%f", env.log.n);
    free_allocated(&env);
    printf("  PASS test_episode_truncation\n");
}

static void test_obs_finite(void) {
    TradingSim env = make_env();
    c_reset(&env);
    for (int step = 0; step < 50; step++) {
        env.actions[0] = (float)(rand() % (2 * N_SYM + 1));
        c_step(&env);
        for (int i = 0; i < env.obs_dim; i++) {
            ASSERTF(isfinite(env.observations[i]), "step=%d obs[%d]=%f", step, i, env.observations[i]);
        }
        ASSERTF(isfinite(env.rewards[0]), "step=%d reward=%f", step, env.rewards[0]);
    }
    free_allocated(&env);
    printf("  PASS test_obs_finite\n");
}

static void test_sell_wrong_symbol(void) {
    TradingSim env = make_env();
    c_reset(&env);
    env.start_bar = 10;
    env.current_bar = 10;
    env.prev_equity = env.initial_cash;
    compute_observations(&env);

    env.actions[0] = 1.0f;
    c_step(&env);
    ASSERTF(env.holding_symbol == 0, "hold 0");

    env.actions[0] = (float)(N_SYM + 2);
    c_step(&env);
    ASSERTF(env.holding_symbol == 0, "still hold 0 after sell wrong sym");
    free_allocated(&env);
    printf("  PASS test_sell_wrong_symbol\n");
}

static void test_equity_tracking(void) {
    TradingSim env = make_env();
    env.fee_rate = 0.0f;
    c_reset(&env);
    float eq0 = ts_equity(&env);
    ASSERTF(fabsf(eq0 - 10000.0f) < 0.01f, "eq=%f", eq0);

    env.start_bar = 10;
    env.current_bar = 10;
    env.prev_equity = env.initial_cash;
    compute_observations(&env);
    env.actions[0] = 1.0f;
    c_step(&env);
    float eq1 = ts_equity(&env);
    ASSERTF(eq1 > 0.0f, "equity=%f", eq1);
    ASSERTF(fabsf(eq1 - 10000.0f) < 50.0f, "equity drifted too far: %f", eq1);
    free_allocated(&env);
    printf("  PASS test_equity_tracking\n");
}

int main(void) {
    fill_data();
    srand(42);
    printf("Running trading_sim tests...\n");
    test_reset();
    test_hold();
    test_buy_sell();
    test_buy_fees();
    test_switch_symbol();
    test_max_hold();
    test_episode_truncation();
    test_obs_finite();
    test_sell_wrong_symbol();
    test_equity_tracking();
    printf("All tests passed.\n");
    return 0;
}
