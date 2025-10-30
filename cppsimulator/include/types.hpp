#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace msim {

struct FeeLeverageConfig {
  double stock_fee = 0.0005;       // equities trading fee
  double crypto_fee = 0.0015;      // crypto trading fee
  double slip_bps = 1.5;           // linear slippage, basis points
  double annual_leverage = 0.0675; // 6.75% annual financing
  double intraday_max = 4.0;       // <= 4x intraday leverage
  double overnight_max = 2.0;      // auto clamp to 2x at close
};

enum class Mode : int {
  OpenClose = 0,
  Event = 1,
  MaxDiff = 2
};

struct SimConfig {
  int context_len = 128;
  int horizon = 1;
  Mode mode = Mode::OpenClose;
  bool normalize_returns = true;
  int seed = 1337;
  FeeLeverageConfig fees{};
};

struct BatchState {
  torch::Tensor ohlc;      // [B, T, F] float32
  torch::Tensor returns;   // [B, T] float32
  torch::Tensor is_crypto; // [B] bool
  torch::Tensor pos;       // [B] float32, current position
  torch::Tensor equity;    // [B] float32, current equity multiple
  torch::Tensor t;         // scalar int64 step index
  int64_t T = 0;
  int64_t F = 0;
  int64_t B = 0;
};

struct StepResult {
  torch::Tensor obs;    // [B, C, F] context window
  torch::Tensor reward; // [B]
  torch::Tensor done;   // [B] bool
  torch::Tensor gross;  // [B] gross pnl before costs
  torch::Tensor trade_cost;    // [B] entry trading+slippage cost
  torch::Tensor financing_cost; // [B] financing cost at open
  torch::Tensor deleverage_cost; // [B] auto deleverage cost at close
  torch::Tensor deleverage_notional; // [B] absolute exposure trimmed at close
  torch::Tensor position; // [B] end-of-step position after deleverage
  torch::Tensor equity;   // [B] equity after step
};

} // namespace msim
