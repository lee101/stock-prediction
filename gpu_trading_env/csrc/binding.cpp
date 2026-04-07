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

} // namespace gpu_trading_env

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("env_step", &gpu_trading_env::env_step_py, "Fused SoA env_step (CUDA)");
}
