#pragma once

#include <string>

#include <torch/script.h>

namespace msim {

class ForecastModel {
public:
  ForecastModel() = default;

  void load(const std::string& path, torch::Device device);
  torch::Tensor forward(const torch::Tensor& context) const;
  [[nodiscard]] bool is_loaded() const noexcept { return loaded_; }

private:
  mutable torch::jit::script::Module module_;
  bool loaded_ = false;
};

struct ForecastBundle {
  ForecastModel chronos_or_kronos;
  ForecastModel toto;
  bool use_chronos = false;
  bool use_toto = false;
};

} // namespace msim
