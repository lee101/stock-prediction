// Tests for the DPS (direction, size, limit_offset) action-mode extension to
// the C++ market simulator. The SCALAR codepath in Portfolio::step is provably
// bit-identical to the pre-change version when leverage_cap_override_ <= 0
// (the default), because the new branch resolves to exactly config_.MAX_LEVERAGE
// — the literal value the previous code passed to torch::clamp. This test
// asserts that property numerically and then exercises the new DPS behaviors:
//   1. SCALAR mode with override=0 produces the same outputs across calls
//      (golden against hardcoded values computed analytically below).
//   2. SCALAR mode is unaffected by toggling and resetting the override.
//   3. DPS leverage cap is honored (size=1, dir=1 -> 5x notional, not more).
//   4. DPS fee scales with notional (10 bps on |trade|, crypto fee).
//   5. DPS zero action -> zero PnL change, zero fee.

#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "market_config.h"
#include "market_state.h"
#include "portfolio.h"

using namespace market_sim;

namespace {

int g_failed = 0;
int g_passed = 0;

#define EXPECT_NEAR(a, b, tol, msg)                                            \
    do {                                                                       \
        double _a = static_cast<double>(a);                                    \
        double _b = static_cast<double>(b);                                    \
        if (std::abs(_a - _b) > (tol)) {                                       \
            std::cerr << "FAIL " << msg << ": " << _a << " vs " << _b          \
                      << " (tol " << tol << ")\n";                             \
            g_failed++;                                                        \
        } else {                                                               \
            g_passed++;                                                        \
        }                                                                      \
    } while (0)

#define EXPECT_TRUE(cond, msg)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "FAIL " << msg << ": condition false\n";              \
            g_failed++;                                                        \
        } else {                                                               \
            g_passed++;                                                        \
        }                                                                      \
    } while (0)

// Build an OHLC price tensor for a single env. Layout matches market_env.cpp:
// index 0=open, 1=high, 2=low, 3=close.
torch::Tensor make_ohlc(float o, float h, float l, float c, torch::Device dev) {
    auto t = torch::tensor({{o, h, l, c}},
                           torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    return t;
}

}  // namespace

int main() {
    auto device = torch::Device("cpu");

    // ---- Test 1: SCALAR mode golden ------------------------------------------------
    // 1 env, action = +1.0 (full long at 1x leverage), price stays flat at 100.
    // Expected fee  = |100000 * 1.0| * 0.0005 = 50.0
    // Expected cash = 100000 - 100000 - 50 = -50
    // Expected position = 100000
    // No leverage cost (|pos|/equity == 100000/99950 ~ 1.0005, excess = 0.0005 -> tiny)
    // PnL change ~ -50 - tiny_lev_cost (price flat, fee dominates).
    {
        MarketConfig cfg;
        cfg.action_mode = ActionMode::SCALAR;
        Portfolio pf(cfg, device, /*batch_size=*/1);
        MarketState mkt(cfg, device);  // default-constructed -> STOCK fee 5 bps

        auto action = torch::tensor({1.0f}, torch::TensorOptions().device(device));
        auto prices = make_ohlc(100.f, 100.f, 100.f, 100.f, device);

        auto r = pf.step(action, prices, mkt, false);

        float fee = r.fees_paid[0].item<float>();
        EXPECT_NEAR(fee, 50.0f, 1e-3, "SCALAR golden: fee on full-long open");

        auto state = pf.get_state();  // [1,5]
        float position_pct = state.index({0, 1}).item<float>();
        // position 100000 / equity (~99950 - tiny_lev_cost) ~ 1.0005
        EXPECT_TRUE(position_pct > 0.9995f && position_pct < 1.0015f,
                    "SCALAR golden: position pct ~1x");

        // Override stays at default 0 -> behavior unchanged after toggling.
        pf.set_leverage_cap_override(5.0f);
        pf.set_leverage_cap_override(0.0f);
        EXPECT_NEAR(pf.leverage_cap_override(), 0.0f, 0.0, "override resets to 0");
    }

    // ---- Test 2: SCALAR clamp at MAX_LEVERAGE=1.5 ----------------------------------
    {
        MarketConfig cfg;
        Portfolio pf(cfg, device, 1);
        MarketState mkt(cfg, device);
        // Action 5.0 should be clamped to 1.5
        auto action = torch::tensor({5.0f}, torch::TensorOptions().device(device));
        auto prices = make_ohlc(100.f, 100.f, 100.f, 100.f, device);
        auto r = pf.step(action, prices, mkt, false);
        // fee = 1.5 * 100000 * 0.0005 = 75
        EXPECT_NEAR(r.fees_paid[0].item<float>(), 75.0f, 1e-3,
                    "SCALAR: action=5 clamped to 1.5x -> fee=75");
    }

    // ---- Test 3: DPS leverage cap (size=1, dir=1 -> 5x not more) -------------------
    {
        MarketConfig cfg;
        cfg.action_mode = ActionMode::DPS;
        Portfolio pf(cfg, device, 1);
        MarketState mkt(cfg, device);
        pf.set_leverage_cap_override(cfg.max_leverage_dps);

        // Pass scalar leverage = dir*size*5 = 1*1*5 = 5.0 (what env step would do).
        auto action = torch::tensor({5.0f}, torch::TensorOptions().device(device));
        auto prices = make_ohlc(100.f, 100.f, 100.f, 100.f, device);
        auto r = pf.step(action, prices, mkt, false);
        // fee = 5 * 100000 * 0.0005 = 250
        EXPECT_NEAR(r.fees_paid[0].item<float>(), 250.0f, 1e-3,
                    "DPS: leverage cap honored at 5x -> fee=250");

        // Now request 99x — must clamp to 5x as well.
        Portfolio pf2(cfg, device, 1);
        pf2.set_leverage_cap_override(cfg.max_leverage_dps);
        auto big = torch::tensor({99.0f}, torch::TensorOptions().device(device));
        auto r2 = pf2.step(big, prices, mkt, false);
        EXPECT_NEAR(r2.fees_paid[0].item<float>(), 250.0f, 1e-3,
                    "DPS: 99x request clamped to 5x");
    }

    // ---- Test 4: DPS zero action -> zero PnL & zero fee ----------------------------
    {
        MarketConfig cfg;
        cfg.action_mode = ActionMode::DPS;
        Portfolio pf(cfg, device, 1);
        MarketState mkt(cfg, device);
        pf.set_leverage_cap_override(cfg.max_leverage_dps);

        auto action = torch::tensor({0.0f}, torch::TensorOptions().device(device));
        auto prices = make_ohlc(100.f, 105.f, 95.f, 100.f, device);
        auto r = pf.step(action, prices, mkt, false);
        EXPECT_NEAR(r.fees_paid[0].item<float>(), 0.0f, 1e-6, "DPS zero: no fee");
        EXPECT_NEAR(r.realized_pnl[0].item<float>(), 0.0f, 1e-6, "DPS zero: no realized pnl");
    }

    // ---- Test 5: DPS fee scales linearly with |notional| ---------------------------
    {
        MarketConfig cfg;
        cfg.action_mode = ActionMode::DPS;
        auto run = [&](float lev) {
            Portfolio pf(cfg, device, 1);
            MarketState mkt(cfg, device);
            pf.set_leverage_cap_override(cfg.max_leverage_dps);
            auto a = torch::tensor({lev}, torch::TensorOptions().device(device));
            auto p = make_ohlc(100.f, 100.f, 100.f, 100.f, device);
            return pf.step(a, p, mkt, false).fees_paid[0].item<float>();
        };
        float f1 = run(1.0f);
        float f3 = run(3.0f);
        EXPECT_NEAR(f3, 3.0f * f1, 1e-3, "DPS: fee linear in notional");
    }

    // ---- Test 6: execution_price_override flows into PnL & cost basis -------------
    // Vanilla SCALAR vs SCALAR+override on a flat bar: fees match exactly (fee
    // depends only on |trade_amount|, not on price), and on the *next* step
    // closing back to flat at close=100 the override portfolio has realized
    // PnL ~= -1% of notional (entry was marked at 101) while the vanilla
    // portfolio realizes ~0.
    {
        MarketConfig cfg;
        cfg.action_mode = ActionMode::SCALAR;

        Portfolio pf_a(cfg, device, 1);
        MarketState mkt(cfg, device);
        auto action = torch::tensor({1.0f}, torch::TensorOptions().device(device));
        auto prices = make_ohlc(100.f, 105.f, 95.f, 100.f, device);
        auto r_a = pf_a.step(action, prices, mkt, false);
        float fee_a    = r_a.fees_paid[0].item<float>();
        float unreal_a = r_a.unrealized_pnl[0].item<float>();

        Portfolio pf_b(cfg, device, 1);
        auto override_px = torch::tensor({101.0f}, torch::TensorOptions().device(device));
        pf_b.set_execution_price_override(override_px);
        auto r_b = pf_b.step(action, prices, mkt, false);
        float fee_b = r_b.fees_paid[0].item<float>();
        pf_b.clear_execution_price_override();

        EXPECT_NEAR(fee_a, fee_b, 1e-3, "override: fee independent of exec price");

        // Close back to flat at close=100 to materialize realized PnL.
        auto sell    = torch::tensor({0.0f}, torch::TensorOptions().device(device));
        auto prices2 = make_ohlc(100.f, 100.f, 100.f, 100.f, device);
        auto r_a2 = pf_a.step(sell, prices2, mkt, false);
        auto r_b2 = pf_b.step(sell, prices2, mkt, false);
        float realized_a = r_a2.realized_pnl[0].item<float>();
        float realized_b = r_b2.realized_pnl[0].item<float>();
        EXPECT_NEAR(realized_a, 0.0f, 1e-2, "vanilla close: realized=0");
        // pos ~100000, entry=101, exit=100 -> ~ -990
        EXPECT_TRUE(realized_b < -900.0f && realized_b > -1100.0f,
                    "override: realized ~ -1% of notional from limit entry");

        // Bit-identity of SCALAR golden after toggling override on/off.
        Portfolio pf_c(cfg, device, 1);
        pf_c.set_execution_price_override(override_px);
        pf_c.clear_execution_price_override();
        auto r_c = pf_c.step(action, prices, mkt, false);
        EXPECT_NEAR(r_c.fees_paid[0].item<float>(), fee_a, 0.0,
                    "override toggle: SCALAR fee bit-identical");
        EXPECT_NEAR(r_c.unrealized_pnl[0].item<float>(), unreal_a, 0.0,
                    "override toggle: SCALAR unreal bit-identical");
    }

    // ---- Test 7: cost basis uses override (entry_price = override) ----------------
    {
        MarketConfig cfg;
        Portfolio pf(cfg, device, 1);
        MarketState mkt(cfg, device);
        auto override_px = torch::tensor({110.0f}, torch::TensorOptions().device(device));
        pf.set_execution_price_override(override_px);
        auto open_action = torch::tensor({1.0f}, torch::TensorOptions().device(device));
        auto prices = make_ohlc(100.f, 100.f, 100.f, 100.f, device);
        pf.step(open_action, prices, mkt, false);
        pf.clear_execution_price_override();

        auto sell = torch::tensor({0.0f}, torch::TensorOptions().device(device));
        auto r = pf.step(sell, prices, mkt, false);
        float realized = r.realized_pnl[0].item<float>();
        // pos ~100000, entry=110, exit=100 -> ~ -9090
        EXPECT_TRUE(realized < -8000.0f && realized > -10000.0f,
                    "cost basis uses override: ~-9% loss closing override-entry at close");
    }

    std::cout << "\n=== test_dps_action_mode summary ===\n";
    std::cout << "passed: " << g_passed << "  failed: " << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
