#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include <torch/extension.h>

#include "market_sim.hpp"

namespace py = pybind11;

namespace {

msim::Mode str_to_mode(const std::string& mode) {
  std::string lowered = mode;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (lowered == "open_close" || lowered == "openclose") {
    return msim::Mode::OpenClose;
  }
  if (lowered == "event") {
    return msim::Mode::Event;
  }
  if (lowered == "maxdiff" || lowered == "max_diff") {
    return msim::Mode::MaxDiff;
  }
  throw std::invalid_argument("Unknown simulation mode: " + mode);
}

} // namespace

PYBIND11_MODULE(market_sim_ext, m) {
  m.doc() = "PyTorch bindings for the high-performance market simulator.";

  py::enum_<msim::Mode>(m, "Mode")
      .value("OpenClose", msim::Mode::OpenClose)
      .value("Event", msim::Mode::Event)
      .value("MaxDiff", msim::Mode::MaxDiff)
      .export_values();

  py::class_<msim::FeeLeverageConfig>(m, "FeeLeverageConfig")
      .def(py::init<>())
      .def_readwrite("stock_fee", &msim::FeeLeverageConfig::stock_fee)
      .def_readwrite("crypto_fee", &msim::FeeLeverageConfig::crypto_fee)
      .def_readwrite("slip_bps", &msim::FeeLeverageConfig::slip_bps)
      .def_readwrite("annual_leverage", &msim::FeeLeverageConfig::annual_leverage)
      .def_readwrite("intraday_max", &msim::FeeLeverageConfig::intraday_max)
      .def_readwrite("overnight_max", &msim::FeeLeverageConfig::overnight_max);

  py::class_<msim::SimConfig>(m, "SimConfig")
      .def(py::init<>())
      .def_readwrite("context_len", &msim::SimConfig::context_len)
      .def_readwrite("horizon", &msim::SimConfig::horizon)
      .def_readwrite("mode", &msim::SimConfig::mode)
      .def_readwrite("normalize_returns", &msim::SimConfig::normalize_returns)
      .def_readwrite("seed", &msim::SimConfig::seed)
      .def_readwrite("fees", &msim::SimConfig::fees);

  py::class_<msim::MarketSimulator>(m, "MarketSimulator")
      .def(
          py::init([](msim::SimConfig cfg,
                      const torch::Tensor& ohlc,
                      const torch::Tensor& is_crypto,
                      const std::string& device) {
            return std::make_unique<msim::MarketSimulator>(cfg, ohlc, is_crypto, torch::Device(device));
          }),
          py::arg("cfg"),
          py::arg("ohlc"),
          py::arg("is_crypto"),
          py::arg("device") = std::string("cpu"))
      .def(
          "reset",
          [](msim::MarketSimulator& self, int64_t t0) {
            return self.reset(t0);
          },
          py::arg("t0"))
      .def(
      "step",
      [](msim::MarketSimulator& self, const torch::Tensor& actions) {
        auto result = self.step(actions);
        py::dict out;
        out["obs"] = result.obs;
        out["reward"] = result.reward;
        out["done"] = result.done;
        out["gross"] = result.gross;
        out["trade_cost"] = result.trade_cost;
        out["financing_cost"] = result.financing_cost;
        out["deleverage_cost"] = result.deleverage_cost;
        out["deleverage_notional"] = result.deleverage_notional;
        out["position"] = result.position;
        out["equity"] = result.equity;
        return out;
      },
      py::arg("actions"))
      .def_property_readonly("cfg", &msim::MarketSimulator::cfg);

  m.def(
      "sim_config_from_dict",
      [](const py::dict& cfg_dict) {
        msim::SimConfig cfg;
        if (cfg_dict.contains("context_len")) {
          cfg.context_len = cfg_dict["context_len"].cast<int>();
        }
        if (cfg_dict.contains("horizon")) {
          cfg.horizon = cfg_dict["horizon"].cast<int>();
        }
        if (cfg_dict.contains("mode")) {
          if (py::isinstance<py::str>(cfg_dict["mode"])) {
            cfg.mode = str_to_mode(cfg_dict["mode"].cast<std::string>());
          } else {
            cfg.mode = cfg_dict["mode"].cast<msim::Mode>();
          }
        }
        if (cfg_dict.contains("normalize_returns")) {
          cfg.normalize_returns = cfg_dict["normalize_returns"].cast<bool>();
        }
        if (cfg_dict.contains("seed")) {
          cfg.seed = cfg_dict["seed"].cast<int>();
        }
        if (cfg_dict.contains("fees")) {
          auto fees_obj = cfg_dict["fees"];
          msim::FeeLeverageConfig fees;
          if (py::isinstance<py::dict>(fees_obj)) {
            auto fees_dict = fees_obj.cast<py::dict>();
            if (fees_dict.contains("stock_fee")) {
              fees.stock_fee = fees_dict["stock_fee"].cast<double>();
            }
            if (fees_dict.contains("crypto_fee")) {
              fees.crypto_fee = fees_dict["crypto_fee"].cast<double>();
            }
            if (fees_dict.contains("slip_bps")) {
              fees.slip_bps = fees_dict["slip_bps"].cast<double>();
            }
            if (fees_dict.contains("annual_leverage")) {
              fees.annual_leverage = fees_dict["annual_leverage"].cast<double>();
            }
            if (fees_dict.contains("intraday_max")) {
              fees.intraday_max = fees_dict["intraday_max"].cast<double>();
            }
            if (fees_dict.contains("overnight_max")) {
              fees.overnight_max = fees_dict["overnight_max"].cast<double>();
            }
          } else if (py::isinstance<msim::FeeLeverageConfig>(fees_obj)) {
            fees = fees_obj.cast<msim::FeeLeverageConfig>();
          }
          cfg.fees = fees;
        }
        return cfg;
      },
      py::arg("cfg_dict"));

  m.def(
      "mode_from_string",
      [](const std::string& name) {
        return str_to_mode(name);
      },
      py::arg("name"));
}
