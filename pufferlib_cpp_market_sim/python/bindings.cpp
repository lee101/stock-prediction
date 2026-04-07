// Pybind11 bindings for market_sim::MarketEnvironment.
//
// Exposes a thin Python class returning torch tensors directly via the
// libtorch C++ API. Supports both SCALAR (action_dim=1) and DPS
// (action_dim=3) action modes; the action dimensionality is read at
// runtime from MarketEnvironment::get_action_dim().

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <memory>
#include <string>
#include <vector>

#include "market_config.h"
#include "market_env.h"

namespace py = pybind11;
using market_sim::ActionMode;
using market_sim::EnvOutput;
using market_sim::MarketConfig;
using market_sim::MarketEnvironment;

namespace {

// Convert an EnvOutput struct into a Python dict of torch tensors. The
// tensors live on whichever device the env was constructed with; we do
// not copy through CPU.
py::dict env_output_to_dict(const EnvOutput& out) {
    py::dict d;
    d["observations"]   = out.observations;
    d["rewards"]        = out.rewards;
    d["dones"]          = out.dones;
    d["truncated"]      = out.truncated;
    d["prices"]         = out.prices;
    d["fees_paid"]      = out.fees_paid;
    d["leverage_costs"] = out.leverage_costs;
    d["realized_pnl"]   = out.realized_pnl;
    d["days_held"]      = out.days_held;
    return d;
}

// Build a MarketConfig from kwargs. Only fields the Python user is
// likely to want to override are surfaced; everything else falls back
// to the C++ defaults so existing SCALAR runs stay bit-identical.
MarketConfig make_config(
    const std::string& data_dir,
    const std::string& log_dir,
    const std::string& device,
    const std::string& action_mode,
    float max_leverage_dps,
    float max_limit_offset_bps,
    float fill_buffer_bps
) {
    MarketConfig cfg;
    cfg.data_dir = data_dir;
    cfg.log_dir = log_dir;
    cfg.device = device;
    if (action_mode == "dps" || action_mode == "DPS") {
        cfg.action_mode = ActionMode::DPS;
    } else if (action_mode == "scalar" || action_mode == "SCALAR") {
        cfg.action_mode = ActionMode::SCALAR;
    } else {
        throw std::invalid_argument(
            "action_mode must be 'scalar' or 'dps', got: " + action_mode);
    }
    cfg.max_leverage_dps = max_leverage_dps;
    cfg.max_limit_offset_bps = max_limit_offset_bps;
    cfg.fill_buffer_bps = fill_buffer_bps;
    return cfg;
}

}  // namespace

PYBIND11_MODULE(_market_sim_py_ext, m) {
    m.doc() = "pybind11 bindings for the C++ market simulator (SCALAR + DPS)";

    py::class_<MarketEnvironment>(m, "MarketEnvironment")
        .def(py::init([](const std::string& data_dir,
                         const std::string& log_dir,
                         const std::string& device,
                         const std::string& action_mode,
                         float max_leverage_dps,
                         float max_limit_offset_bps,
                         float fill_buffer_bps) {
                 auto cfg = make_config(data_dir, log_dir, device,
                                        action_mode, max_leverage_dps,
                                        max_limit_offset_bps, fill_buffer_bps);
                 return std::make_unique<MarketEnvironment>(cfg);
             }),
             py::arg("data_dir") = "../../trainingdata",
             py::arg("log_dir") = "../logs",
             py::arg("device") = "cpu",
             py::arg("action_mode") = "scalar",
             py::arg("max_leverage_dps") = 5.0f,
             py::arg("max_limit_offset_bps") = 20.0f,
             py::arg("fill_buffer_bps") = 5.0f)
        .def("load_symbols", &MarketEnvironment::load_symbols,
             py::arg("symbols"),
             "Load OHLCV CSVs for the given list of symbols.")
        .def("reset",
             [](MarketEnvironment& self, py::object env_indices) {
                 torch::Tensor idx;
                 if (env_indices.is_none()) {
                     idx = torch::arange(
                         MarketConfig::NUM_PARALLEL_ENVS,
                         torch::TensorOptions().dtype(torch::kInt64));
                 } else {
                     idx = env_indices.cast<torch::Tensor>();
                 }
                 return env_output_to_dict(self.reset(idx));
             },
             py::arg("env_indices") = py::none(),
             "Reset the given env indices (default: all).")
        .def("step",
             [](MarketEnvironment& self, torch::Tensor action) {
                 return env_output_to_dict(self.step(action));
             },
             py::arg("action"),
             "Step all environments. Action shape is [batch] for SCALAR "
             "mode and [batch, 3] for DPS mode.")
        .def("get_observation_dim", &MarketEnvironment::get_observation_dim)
        .def("get_action_dim", &MarketEnvironment::get_action_dim)
        .def("observation_space",
             [](const MarketEnvironment& self) {
                 return py::make_tuple(MarketConfig::NUM_PARALLEL_ENVS,
                                       self.get_observation_dim());
             },
             "Shape tuple (batch, obs_dim) for the observation tensor.")
        .def("action_space",
             [](const MarketEnvironment& self) {
                 int adim = self.get_action_dim();
                 if (adim == 1) {
                     return py::make_tuple(MarketConfig::NUM_PARALLEL_ENVS);
                 }
                 return py::make_tuple(MarketConfig::NUM_PARALLEL_ENVS, adim);
             },
             "Shape tuple for the action tensor (SCALAR=[batch], DPS=[batch,3]).")
        .def("peek_next_prices", &MarketEnvironment::peek_next_prices)
        .def("set_training_mode", &MarketEnvironment::set_training_mode)
        .def("flush_logs", &MarketEnvironment::flush_logs)
        .def_property_readonly_static("BATCH_SIZE", [](py::object) {
            return MarketConfig::NUM_PARALLEL_ENVS;
        });
}
