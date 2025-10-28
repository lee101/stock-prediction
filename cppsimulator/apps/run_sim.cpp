#include "market_sim.hpp"
#include "forecast.hpp"

#include <iostream>

namespace idx = torch::indexing;

using namespace msim;

torch::Device pick_device() {
  if (!torch::cuda::is_available()) {
    return torch::kCPU;
  }
  try {
    auto probe = torch::rand({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    (void)probe;
    return torch::kCUDA;
  } catch (const c10::Error& err) {
    std::cerr << "[warn] CUDA reported available but probe tensor failed; "
                 "falling back to CPU. "
              << err.what_without_backtrace() << std::endl;
    return torch::kCPU;
  }
}

int main() {
  torch::manual_seed(123);
  auto device = pick_device();

  const int64_t B = 1024;
  const int64_t T = 2048;
  const int64_t F = 6;
  const int64_t C = 128;

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto ohlc = torch::randn({B, T, F}, options);

  // Make OHLC columns coherent
  auto opens = ohlc.index({idx::Slice(), idx::Slice(), 0});
  auto highs = opens + torch::abs(torch::randn({B, T}, options));
  auto lows = opens - torch::abs(torch::randn({B, T}, options));
  auto closes = opens + 0.1 * torch::randn({B, T}, options);
  ohlc.index_put_({idx::Slice(), idx::Slice(), 1}, highs);
  ohlc.index_put_({idx::Slice(), idx::Slice(), 2}, lows);
  ohlc.index_put_({idx::Slice(), idx::Slice(), 3}, closes);

  auto is_crypto =
      (torch::rand({B}, options) > 0.8).to(torch::kBool);

  SimConfig cfg;
  cfg.context_len = C;
  cfg.mode = Mode::OpenClose;

  MarketSimulator sim(cfg, ohlc, is_crypto, device);

  ForecastBundle fb;
  sim.attach_forecasts(std::move(fb));

  auto obs = sim.reset(C);
  for (int step = 0; step < 256; ++step) {
    auto actions = torch::rand({B}, options) * 2.0 - 1.0;
    auto res = sim.step(actions);
    if (step % 32 == 0) {
      auto mean_r = res.reward.mean().item<float>();
      std::cout << "step " << step << " reward " << mean_r << std::endl;
    }
    if (res.done.any().item<bool>()) {
      break;
    }
    obs = res.obs;
  }

  return 0;
}
