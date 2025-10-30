#include "market_sim.hpp"

#include <algorithm>

namespace idx = torch::indexing;

namespace msim {

namespace {

torch::Tensor stable_std(const torch::Tensor& x, int64_t start) {
  TORCH_CHECK(x.dim() >= 2, "stable_std expects at least 2-D tensor");
  TORCH_CHECK(start < x.size(-1), "start index must be less than sequence length");
  auto slice = x.index({idx::Slice(), idx::Slice(start, idx::None)});
  auto s = slice.std(/*dim=*/-1, /*unbiased=*/false, /*keepdim=*/true);
  auto eps = torch::full_like(s, 1e-8);
  return torch::maximum(s, eps);
}

} // namespace

MarketSimulator::MarketSimulator(const SimConfig& cfg,
                                 const torch::Tensor& ohlc,
                                 const torch::Tensor& is_crypto,
                                 torch::Device device)
    : cfg_(cfg), device_(device) {
  TORCH_CHECK(ohlc.dim() == 3, "ohlc must be [B, T, F]");
  TORCH_CHECK(is_crypto.dim() == 1, "is_crypto must be [B]");
  TORCH_CHECK(ohlc.size(0) == is_crypto.size(0),
              "ohlc and is_crypto batch size mismatch");

  st_.ohlc = ohlc.to(device_).contiguous();
  st_.B = st_.ohlc.size(0);
  st_.T = st_.ohlc.size(1);
  st_.F = st_.ohlc.size(2);

  TORCH_CHECK(cfg_.context_len > 0 && cfg_.context_len < st_.T,
              "context_len must be in (0, T)");
  TORCH_CHECK(cfg_.horizon >= 1, "horizon must be >= 1");

  st_.is_crypto = is_crypto.to(device_).to(torch::kBool).contiguous();
  st_.pos = torch::zeros({st_.B}, st_.ohlc.options().dtype(torch::kFloat32));
  st_.equity =
      torch::ones({st_.B}, st_.ohlc.options().dtype(torch::kFloat32));
  st_.t = torch::tensor(int64_t{0},
                        torch::TensorOptions().dtype(torch::kInt64).device(device_));

  auto closes = st_.ohlc.index({idx::Slice(), idx::Slice(), 3});
  st_.returns =
      torch::zeros({st_.B, st_.T}, closes.options().dtype(torch::kFloat32));

  auto prev_close = closes.index({idx::Slice(), idx::Slice(idx::None, -1)});
  auto next_close = closes.index({idx::Slice(), idx::Slice(1, idx::None)});
  auto denom = torch::clamp(prev_close, 1e-6);
  auto simple_ret = (next_close - prev_close) / denom;
  st_.returns.index_put_({idx::Slice(), idx::Slice(1, idx::None)}, simple_ret);

  if (cfg_.normalize_returns) {
    auto s = stable_std(st_.returns, std::max<int>(1, cfg_.context_len));
    st_.returns = st_.returns / s;
  }
}

torch::Tensor MarketSimulator::make_observation(int64_t t) const {
  TORCH_CHECK(t <= st_.T, "observation index out of range");
  auto left = t - cfg_.context_len;
  TORCH_CHECK(left >= 0, "context window before start of series");
  return st_.ohlc.index({idx::Slice(), idx::Slice(left, t), idx::Slice()});
}

torch::Tensor MarketSimulator::action_to_target(
    const torch::Tensor& unit_action) const {
  auto a = torch::tanh(unit_action);
  auto crypto_mask = st_.is_crypto.to(torch::kFloat32);
  auto stock_mask = 1.0 - crypto_mask;

  auto stock_pos = a * cfg_.fees.intraday_max;
  // Crypto instruments are long-only with no leverage.
  auto crypto_pos = torch::clamp(a, 0.0, 1.0);
  return crypto_pos * crypto_mask + stock_pos * stock_mask;
}

torch::Tensor MarketSimulator::fees_at(const torch::Tensor& dpos,
                                       const torch::Tensor& equity,
                                       const torch::Tensor& is_crypto) const {
  auto mag = torch::abs(dpos);
  auto fee_rate = torch::where(
      is_crypto,
      torch::full_like(mag, cfg_.fees.crypto_fee),
      torch::full_like(mag, cfg_.fees.stock_fee));
  auto fee = mag * fee_rate * equity;
  auto slip = mag * (cfg_.fees.slip_bps * 1e-4) * equity;
  return fee + slip;
}

torch::Tensor MarketSimulator::financing_at_open(
    const torch::Tensor& pos,
    const torch::Tensor& equity,
    const torch::Tensor& is_crypto) const {
  auto daily = cfg_.fees.annual_leverage / 252.0;
  auto excess = torch::clamp(torch::abs(pos) - 1.0, 0.0);
  auto finance = excess * daily * equity;
  return torch::where(is_crypto, torch::zeros_like(finance), finance);
}

torch::Tensor MarketSimulator::session_pnl(
    int64_t t,
    const torch::Tensor& pos_target,
    const torch::Tensor& equity) const {
  auto px_open = st_.ohlc.index({idx::Slice(), t, 0});
  auto px_high = st_.ohlc.index({idx::Slice(), t, 1});
  auto px_low = st_.ohlc.index({idx::Slice(), t, 2});
  auto px_close = st_.ohlc.index({idx::Slice(), t, 3});
  auto ret_t = st_.returns.index({idx::Slice(), t});

  switch (cfg_.mode) {
  case Mode::OpenClose: {
    auto session_ret =
        (px_close - px_open) / torch::clamp(px_open, 1e-6);
    return equity * pos_target * session_ret;
  }
  case Mode::Event: {
    auto std_all =
        stable_std(st_.returns, std::max<int>(1, cfg_.context_len)).squeeze(-1);
    auto trigger =
        (torch::abs(ret_t) > 1.5 * std_all).to(torch::kFloat32);
    auto eff_pos = trigger * pos_target + (1.0 - trigger) * st_.pos;
    return equity * eff_pos * ret_t;
  }
  case Mode::MaxDiff:
  default: {
    auto up =
        (px_high - px_open) / torch::clamp(px_open, 1e-6);
    auto down =
        (px_open - px_low) / torch::clamp(px_open, 1e-6);
    auto move = torch::where(pos_target >= 0, 0.5 * up, -0.5 * down);
    return equity * pos_target * move;
  }
  }
}

std::pair<torch::Tensor, torch::Tensor> MarketSimulator::auto_deleverage_close(
    int64_t t,
    const torch::Tensor& pos_target,
    const torch::Tensor& equity,
    const torch::Tensor& is_crypto) const {
  auto cap = torch::full_like(pos_target, cfg_.fees.overnight_max);
  cap = cap.masked_fill(is_crypto, 1.0);
  auto lower = torch::full_like(pos_target, -cfg_.fees.overnight_max);
  lower = lower.masked_fill(is_crypto, 0.0);
  auto capped = torch::minimum(torch::maximum(pos_target, lower), cap);
  auto delta = capped - pos_target;
  auto cost = fees_at(delta, equity, is_crypto);
  return {capped, cost};
}

torch::Tensor MarketSimulator::reset(int64_t t0) {
  TORCH_CHECK(t0 >= cfg_.context_len,
              "t0 must be >= context length");
  TORCH_CHECK(t0 < st_.T - cfg_.horizon - 1,
              "t0 too close to end of series");
  st_.t.fill_(t0);
  st_.pos.zero_();
  st_.equity.fill_(1.0f);
  return make_observation(t0);
}

StepResult MarketSimulator::step(const torch::Tensor& actions) {
  TORCH_CHECK(actions.dim() == 1 && actions.size(0) == st_.B,
              "actions must have shape [B]");
  TORCH_CHECK(actions.device() == device_,
              "actions tensor must be on simulator device");

  const int64_t t = st_.t.item<int64_t>();
  auto is_crypto = st_.is_crypto;

  auto pos_target = action_to_target(actions);
  auto px_open = st_.ohlc.index({idx::Slice(), t, 0});
  auto dpos_open = pos_target - st_.pos;
  auto cost_open = fees_at(dpos_open, st_.equity, is_crypto);
  auto finance = financing_at_open(pos_target, st_.equity, is_crypto);
  auto pnl = session_pnl(t, pos_target, st_.equity);

  auto [end_pos, cost_close] =
      auto_deleverage_close(t, pos_target, st_.equity, is_crypto);

  auto reward = pnl - (cost_open + finance + cost_close);
  auto deleverage_notional = torch::abs(end_pos - pos_target);
  auto equity_next = st_.equity + reward;

  int64_t t_next = t + 1;
  st_.t.fill_(t_next);
  st_.pos = end_pos.detach();
  st_.equity = equity_next.detach();
  bool terminal = (t_next >= (st_.T - cfg_.horizon - 1));
  auto done_tensor = torch::full(
      {st_.B}, terminal,
      torch::TensorOptions().dtype(torch::kBool).device(device_));

  auto obs = make_observation(t_next);

  return {
      obs,
      reward,
      done_tensor,
      pnl,
      cost_open,
      finance,
      cost_close,
      deleverage_notional,
      end_pos,
      st_.equity};
}

} // namespace msim
