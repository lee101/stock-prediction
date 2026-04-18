// Python binding for the SoA fused env_step CUDA kernel.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "env_params.h"

namespace gpu_trading_env {

void launch_env_step(
    const at::Tensor& ohlc,
    at::Tensor& pos_qty,
    at::Tensor& pos_entry_px,
    at::Tensor& cash,
    at::Tensor& equity,
    at::Tensor& dd_peak,
    at::Tensor& drawdown,
    at::Tensor& t_idx,
    at::Tensor& done_flag,
    const at::Tensor& action,
    at::Tensor& reward,
    at::Tensor& cost,
    EnvParams p
);

struct MultiEnvParams {
    float fee_bps;
    float buffer_bps;
    float max_leverage;
    float maint_margin;
    float liq_penalty;
    float init_cash;
    int   T_total;
    int   S;
    int   episode_len;
};

void launch_env_step_multisym(
    const at::Tensor& prices,
    at::Tensor& pos_sym,
    at::Tensor& pos_side,
    at::Tensor& pos_qty,
    at::Tensor& pos_entry_px,
    at::Tensor& cash,
    at::Tensor& equity,
    at::Tensor& dd_peak,
    at::Tensor& drawdown,
    at::Tensor& t_idx,
    at::Tensor& done_flag,
    const at::Tensor& action,
    at::Tensor& reward,
    at::Tensor& cost,
    MultiEnvParams p
);

struct PortfolioBracketParams {
    float fee_bps;
    float fill_buffer_bps;
    float max_leverage;
    float annual_margin_rate;
    int   trading_days_per_year;
    int   S;
};

void launch_portfolio_bracket_step(
    const at::Tensor& cash_in,
    const at::Tensor& pos_in,
    const at::Tensor& actions,
    const at::Tensor& prev_close,
    const at::Tensor& bar_open,
    const at::Tensor& bar_high,
    const at::Tensor& bar_low,
    const at::Tensor& bar_close,
    const at::Tensor& tradable,
    at::Tensor& cash_out,
    at::Tensor& pos_out,
    at::Tensor& reward,
    at::Tensor& info_eq_prev,
    at::Tensor& info_new_eq,
    at::Tensor& info_fees,
    at::Tensor& info_margin,
    at::Tensor& info_borrowed,
    PortfolioBracketParams p
);

static void env_step_py(
    at::Tensor ohlc,
    at::Tensor pos_qty,
    at::Tensor pos_entry_px,
    at::Tensor cash,
    at::Tensor equity,
    at::Tensor dd_peak,
    at::Tensor drawdown,
    at::Tensor t_idx,
    at::Tensor done_flag,
    at::Tensor action,
    at::Tensor reward,
    at::Tensor cost,
    double fee_bps,
    double buffer_bps,
    double max_quote_offset_bps,
    double max_leverage,
    double maint_margin,
    double liq_penalty,
    double init_cash,
    int64_t T_total,
    int64_t episode_len
) {
    TORCH_CHECK(ohlc.is_cuda(), "ohlc must be CUDA");
    TORCH_CHECK(ohlc.scalar_type() == at::kFloat, "ohlc must be float32");
    TORCH_CHECK(action.is_cuda() && action.scalar_type() == at::kFloat, "action must be cuda float32");
    TORCH_CHECK(pos_qty.is_cuda(), "state must be CUDA");
    TORCH_CHECK(ohlc.is_contiguous() && action.is_contiguous(), "ohlc/action must be contiguous");

    EnvParams p;
    p.fee_bps = (float)fee_bps;
    p.buffer_bps = (float)buffer_bps;
    p.max_quote_offset_bps = (float)max_quote_offset_bps;
    p.max_leverage = (float)max_leverage;
    p.maint_margin = (float)maint_margin;
    p.liq_penalty = (float)liq_penalty;
    p.init_cash = (float)init_cash;
    p.T_total = (int)T_total;
    p.episode_len = (int)episode_len;

    launch_env_step(ohlc, pos_qty, pos_entry_px, cash, equity, dd_peak,
                    drawdown, t_idx, done_flag, action, reward, cost, p);
}

static void env_step_multisym_py(
    at::Tensor prices,
    at::Tensor pos_sym,
    at::Tensor pos_side,
    at::Tensor pos_qty,
    at::Tensor pos_entry_px,
    at::Tensor cash,
    at::Tensor equity,
    at::Tensor dd_peak,
    at::Tensor drawdown,
    at::Tensor t_idx,
    at::Tensor done_flag,
    at::Tensor action,
    at::Tensor reward,
    at::Tensor cost,
    double fee_bps,
    double buffer_bps,
    double max_leverage,
    double maint_margin,
    double liq_penalty,
    double init_cash,
    int64_t T_total,
    int64_t S,
    int64_t episode_len
) {
    TORCH_CHECK(prices.is_cuda() && prices.scalar_type() == at::kFloat, "prices must be cuda float32");
    TORCH_CHECK(action.is_cuda() && action.scalar_type() == at::kInt, "action must be cuda int32");
    TORCH_CHECK(pos_sym.is_cuda() && pos_sym.scalar_type() == at::kInt, "pos_sym must be cuda int32");
    TORCH_CHECK(pos_side.is_cuda() && pos_side.scalar_type() == at::kInt, "pos_side must be cuda int32");

    MultiEnvParams p;
    p.fee_bps = (float)fee_bps;
    p.buffer_bps = (float)buffer_bps;
    p.max_leverage = (float)max_leverage;
    p.maint_margin = (float)maint_margin;
    p.liq_penalty = (float)liq_penalty;
    p.init_cash = (float)init_cash;
    p.T_total = (int)T_total;
    p.S = (int)S;
    p.episode_len = (int)episode_len;

    launch_env_step_multisym(prices, pos_sym, pos_side, pos_qty, pos_entry_px,
                             cash, equity, dd_peak, drawdown, t_idx, done_flag,
                             action, reward, cost, p);
}

static void portfolio_bracket_step_py(
    at::Tensor cash_in,
    at::Tensor pos_in,
    at::Tensor actions,
    at::Tensor prev_close,
    at::Tensor bar_open,
    at::Tensor bar_high,
    at::Tensor bar_low,
    at::Tensor bar_close,
    at::Tensor tradable,
    at::Tensor cash_out,
    at::Tensor pos_out,
    at::Tensor reward,
    at::Tensor info_eq_prev,
    at::Tensor info_new_eq,
    at::Tensor info_fees,
    at::Tensor info_margin,
    at::Tensor info_borrowed,
    double fee_bps,
    double fill_buffer_bps,
    double max_leverage,
    double annual_margin_rate,
    int64_t trading_days_per_year,
    int64_t S
) {
    TORCH_CHECK(cash_in.is_cuda() && cash_in.scalar_type() == at::kFloat, "cash_in cuda f32");
    TORCH_CHECK(pos_in.is_cuda() && pos_in.scalar_type() == at::kFloat, "pos_in cuda f32");
    TORCH_CHECK(actions.is_cuda() && actions.scalar_type() == at::kFloat, "actions cuda f32");
    TORCH_CHECK(prev_close.is_cuda() && prev_close.scalar_type() == at::kFloat, "prev_close cuda f32");
    TORCH_CHECK(bar_open.is_cuda() && bar_open.scalar_type() == at::kFloat, "bar_open cuda f32");
    TORCH_CHECK(bar_high.is_cuda() && bar_high.scalar_type() == at::kFloat, "bar_high cuda f32");
    TORCH_CHECK(bar_low.is_cuda() && bar_low.scalar_type() == at::kFloat, "bar_low cuda f32");
    TORCH_CHECK(bar_close.is_cuda() && bar_close.scalar_type() == at::kFloat, "bar_close cuda f32");
    TORCH_CHECK(tradable.is_cuda() && tradable.scalar_type() == at::kByte,
                "tradable must be cuda uint8 (cast bool→uint8 in caller)");
    TORCH_CHECK(cash_out.is_cuda() && pos_out.is_cuda() && reward.is_cuda(), "outputs cuda");
    TORCH_CHECK(cash_in.is_contiguous() && pos_in.is_contiguous() && actions.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(prev_close.is_contiguous() && bar_open.is_contiguous() && bar_high.is_contiguous()
                && bar_low.is_contiguous() && bar_close.is_contiguous() && tradable.is_contiguous(),
                "bar tensors must be contiguous");

    PortfolioBracketParams p;
    p.fee_bps               = (float)fee_bps;
    p.fill_buffer_bps       = (float)fill_buffer_bps;
    p.max_leverage          = (float)max_leverage;
    p.annual_margin_rate    = (float)annual_margin_rate;
    p.trading_days_per_year = (int)trading_days_per_year;
    p.S                     = (int)S;

    launch_portfolio_bracket_step(
        cash_in, pos_in, actions, prev_close, bar_open, bar_high, bar_low, bar_close, tradable,
        cash_out, pos_out, reward,
        info_eq_prev, info_new_eq, info_fees, info_margin, info_borrowed,
        p
    );
}

} // namespace gpu_trading_env

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("env_step", &gpu_trading_env::env_step_py, "Fused SoA env_step (CUDA)");
    m.def("env_step_multisym", &gpu_trading_env::env_step_multisym_py,
          "Multi-symbol fused env_step (CUDA)");
    m.def("portfolio_bracket_step", &gpu_trading_env::portfolio_bracket_step_py,
          "Multi-symbol PORTFOLIO bracket env step (CUDA)");
}
