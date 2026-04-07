#pragma once
namespace gpu_trading_env {
struct EnvParams {
    float fee_bps;
    float buffer_bps;
    float max_quote_offset_bps;
    float max_leverage;
    float maint_margin;
    float liq_penalty;
    float init_cash;
    int   T_total;
    int   episode_len;
};
} // namespace gpu_trading_env
