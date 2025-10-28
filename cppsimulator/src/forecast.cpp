#include "forecast.hpp"

#include <utility>

namespace msim {

void ForecastModel::load(const std::string& path, torch::Device device) {
  module_ = torch::jit::load(path, device);
  module_.eval();
  loaded_ = true;
}

torch::Tensor ForecastModel::forward(const torch::Tensor& context) const {
  TORCH_CHECK(loaded_, "ForecastModel not loaded");
  torch::NoGradGuard ng;
  auto output = module_.forward({context}).toTensor();
  return output;
}

} // namespace msim
