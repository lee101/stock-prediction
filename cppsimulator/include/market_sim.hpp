#pragma once

#include <utility>

#include <torch/torch.h>

#include "forecast.hpp"
#include "types.hpp"

namespace msim {

class MarketSimulator {
public:
  MarketSimulator(const SimConfig& cfg,
                  const torch::Tensor& ohlc,
                  const torch::Tensor& is_crypto,
                  torch::Device device);

  torch::Tensor reset(int64_t t0);
  StepResult step(const torch::Tensor& actions);

  void attach_forecasts(ForecastBundle fb) { fb_ = std::move(fb); }

  [[nodiscard]] const BatchState& state() const noexcept { return st_; }
  [[nodiscard]] SimConfig cfg() const noexcept { return cfg_; }

private:
  SimConfig cfg_;
  BatchState st_;
  torch::Device device_;
  ForecastBundle fb_{};

  torch::Tensor fees_at(const torch::Tensor& dpos,
                        const torch::Tensor& equity,
                        const torch::Tensor& is_crypto) const;

  torch::Tensor financing_at_open(const torch::Tensor& pos,
                                  const torch::Tensor& equity,
                                  const torch::Tensor& is_crypto) const;

  torch::Tensor make_observation(int64_t t) const;
  torch::Tensor action_to_target(const torch::Tensor& unit_action) const;
  torch::Tensor session_pnl(int64_t t,
                            const torch::Tensor& pos_target,
                            const torch::Tensor& equity) const;

  std::pair<torch::Tensor, torch::Tensor> auto_deleverage_close(
      int64_t t,
      const torch::Tensor& pos_target,
      const torch::Tensor& equity,
      const torch::Tensor& is_crypto) const;
};

} // namespace msim
