// End-to-end GPU XGBoost training with a custom CUDA market-simulator as
// the validation / early-stopping metric.
//
// Pipeline:
//   1. Generate synthetic (features, label) on-device
//   2. Train XGBoost via the C API with device="cuda", tree_method="hist"
//   3. After each boosting round predict on the validation slice from a
//      CUDA array (XGBoosterPredictFromCudaArray) — detect device vs
//      host-resident output and avoid an unnecessary copy either way
//   4. Launch a custom CUDA kernel to compute turnover, fees, total
//      return, and Sharpe over the validation positions
//   5. Track the best-scoring round and slice the booster at the end
//      (XGBoosterSlice) so the saved model is frozen at that round
//
// Build (see Makefile in this directory):
//   export XGBOOST_HOME=$HOME/src/xgboost   # path to -DUSE_CUDA=ON build
//   make
//
// Notes:
// * XGBoosterPredictFromCudaArray's out_result buffer has ambiguous
//   residency in the public docs ("copy before use"), so we use
//   cudaPointerGetAttributes to detect whether it came back on-device
//   or on the host and only memcpy when required.
// * This is a scaffold. Swap the synthetic generator for your real GPU
//   feature tensor (Chronos-2 quantiles + regime cross-sectional
//   features) and you have a full end-to-end GPU research loop.

#include <cuda_runtime.h>
#include <xgboost/c_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                   \
                   cudaGetErrorName(err), __FILE__, __LINE__,                \
                   cudaGetErrorString(err));                                 \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

#define CHECK_XGB(call)                                                      \
  do {                                                                       \
    int err = (call);                                                        \
    if (err != 0) {                                                          \
      std::fprintf(stderr, "XGBoost error at %s:%d: %s\n",                   \
                   __FILE__, __LINE__, XGBGetLastError());                   \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)


// ──────────────────────────────────────────────────────────────────────────
// CUDA kernels
// ──────────────────────────────────────────────────────────────────────────

// Convert predictions into positions: clip(scale * pred, -cap, cap) then
// (optionally) long-only. We keep positions and prev-positions in two
// buffers so turnover is a trivial |pos - prev| afterwards.
__global__ void compute_positions_kernel(
    const float* __restrict__ preds,
    float* __restrict__ positions,
    int n, float scale, float cap, int is_long_only) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float p = scale * preds[i];
  if (p > cap) p = cap;
  if (p < -cap) p = -cap;
  if (is_long_only && p < 0.0f) p = 0.0f;
  positions[i] = p;
}

// Per-step return: pos[t] * y[t] - fee * |pos[t] - pos[t-1]|.
// pos[-1] == 0 (flat start).
__global__ void step_returns_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ y_true,
    float* __restrict__ rets,
    int n, float fee) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float pos = positions[i];
  float prev = (i == 0) ? 0.0f : positions[i - 1];
  float turnover = fabsf(pos - prev);
  rets[i] = pos * y_true[i] - fee * turnover;
}


// Host-side reduction helper (small n — the validation slice is typically
// a few thousand rows; we just pull to host and reduce).
static float reduce_mean(const float* d_ptr, int n) {
  std::vector<float> h(n);
  CHECK_CUDA(cudaMemcpy(h.data(), d_ptr, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += h[i];
  return static_cast<float>(s / n);
}

static float reduce_stddev(const float* d_ptr, int n, float mean) {
  std::vector<float> h(n);
  CHECK_CUDA(cudaMemcpy(h.data(), d_ptr, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    double d = h[i] - mean;
    s += d * d;
  }
  return static_cast<float>(std::sqrt(s / std::max(1, n - 1)));
}

static float reduce_cum_return(const float* d_ptr, int n) {
  std::vector<float> h(n);
  CHECK_CUDA(cudaMemcpy(h.data(), d_ptr, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double prod = 1.0;
  for (int i = 0; i < n; ++i) prod *= (1.0 + h[i]);
  return static_cast<float>(prod - 1.0);
}


// ──────────────────────────────────────────────────────────────────────────
// __cuda_array_interface__ helpers
// ──────────────────────────────────────────────────────────────────────────
// XGBoost ingests GPU data via __cuda_array_interface__ JSON strings. See
// `XGDMatrixCreateFromCudaArrayInterface` in the C API docs.

static std::string cai_json(const void* ptr, int rows, int cols,
                            const char* typestr) {
  std::ostringstream os;
  os << "{";
  os << "\"data\":[" << reinterpret_cast<uintptr_t>(ptr) << ", false],";
  if (cols > 0) {
    os << "\"shape\":[" << rows << "," << cols << "],";
  } else {
    os << "\"shape\":[" << rows << "],";
  }
  os << "\"typestr\":\"" << typestr << "\",";
  os << "\"version\":3,";
  os << "\"strides\":null";
  os << "}";
  return os.str();
}


// ──────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
  const int n_total = 20000;
  const int n_features = 16;
  const int n_train = 14000;
  const int n_valid = n_total - n_train;
  const int n_rounds = 200;
  const int early_stop_patience = 30;

  // ── 1. Generate synthetic data on device ────────────────────────────
  std::vector<float> h_x(n_total * n_features);
  std::vector<float> h_y(n_total);
  for (int i = 0; i < n_total; ++i) {
    float s = 0.0f;
    for (int j = 0; j < n_features; ++j) {
      float v = ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.02f;
      h_x[i * n_features + j] = v;
      // A weak signal from feature 0 to label.
      if (j == 0) s += 0.5f * v;
    }
    h_y[i] = s + ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.01f;
  }
  float *d_x = nullptr, *d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, sizeof(float) * n_total * n_features));
  CHECK_CUDA(cudaMalloc(&d_y, sizeof(float) * n_total));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), sizeof(float) * h_x.size(),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), sizeof(float) * h_y.size(),
                        cudaMemcpyHostToDevice));

  float* d_x_train = d_x;
  float* d_y_train = d_y;
  float* d_x_valid = d_x + n_train * n_features;
  float* d_y_valid = d_y + n_train;

  auto cai_x_train =
      cai_json(d_x_train, n_train, n_features, "<f4");
  auto cai_y_train = cai_json(d_y_train, n_train, 0, "<f4");
  auto cai_x_valid =
      cai_json(d_x_valid, n_valid, n_features, "<f4");

  // ── 2. Build DMatrix + Booster ──────────────────────────────────────
  DMatrixHandle dtrain = nullptr;
  const char* dmat_cfg = "{\"missing\": NaN}";
  CHECK_XGB(XGDMatrixCreateFromCudaArrayInterface(
      cai_x_train.c_str(), dmat_cfg, &dtrain));
  CHECK_XGB(XGDMatrixSetInfoFromInterface(dtrain, "label",
                                          cai_y_train.c_str()));

  BoosterHandle booster = nullptr;
  DMatrixHandle cache_dmats[1] = {dtrain};
  CHECK_XGB(XGBoosterCreate(cache_dmats, 1, &booster));
  CHECK_XGB(XGBoosterSetParam(booster, "device", "cuda"));
  CHECK_XGB(XGBoosterSetParam(booster, "tree_method", "hist"));
  CHECK_XGB(XGBoosterSetParam(booster, "objective", "reg:pseudohubererror"));
  CHECK_XGB(XGBoosterSetParam(booster, "eval_metric", "mae"));
  CHECK_XGB(XGBoosterSetParam(booster, "eta", "0.02"));
  CHECK_XGB(XGBoosterSetParam(booster, "max_depth", "6"));
  CHECK_XGB(XGBoosterSetParam(booster, "subsample", "0.7"));
  CHECK_XGB(XGBoosterSetParam(booster, "colsample_bytree", "0.7"));

  // ── 3. Training loop + GPU market-sim validation ────────────────────
  const char* predict_cfg =
      "{\"type\":0,\"training\":false,\"iteration_begin\":0,"
      "\"iteration_end\":0,\"strict_shape\":false}";

  const float fee = 0.0005f;     // per unit turnover
  const float scale = 1.0f;
  const float cap = 0.3f;
  const int is_long_only = 1;

  float* d_positions = nullptr;
  float* d_rets = nullptr;
  CHECK_CUDA(cudaMalloc(&d_positions, n_valid * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_rets, n_valid * sizeof(float)));

  float best_metric = -1e30f;
  int best_iter = 0;
  int stale = 0;

  const int block = 256;
  const int grid = (n_valid + block - 1) / block;

  for (int iter = 0; iter < n_rounds; ++iter) {
    CHECK_XGB(XGBoosterUpdateOneIter(booster, iter, dtrain));

    // Predict from CUDA array — may return a host or device pointer.
    uint64_t out_dim = 0;
    const uint64_t* out_shape = nullptr;
    const float* out_preds = nullptr;
    CHECK_XGB(XGBoosterPredictFromCudaArray(
        booster, cai_x_valid.c_str(), predict_cfg, nullptr,
        &out_shape, &out_dim,
        reinterpret_cast<const float**>(&out_preds)));

    // Resolve to a device pointer.
    const float* d_preds = nullptr;
    cudaPointerAttributes attr{};
    cudaError_t pa_err = cudaPointerGetAttributes(&attr, out_preds);
    if (pa_err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
      d_preds = out_preds;
    } else {
      cudaGetLastError();  // clear non-device-ptr error state
      float* tmp = nullptr;
      CHECK_CUDA(cudaMalloc(&tmp, n_valid * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(tmp, out_preds, n_valid * sizeof(float),
                            cudaMemcpyHostToDevice));
      d_preds = tmp;
    }

    compute_positions_kernel<<<grid, block>>>(
        d_preds, d_positions, n_valid, scale, cap, is_long_only);
    step_returns_kernel<<<grid, block>>>(
        d_positions, d_y_valid, d_rets, n_valid, fee);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float total_return = reduce_cum_return(d_rets, n_valid);
    float mean = reduce_mean(d_rets, n_valid);
    float sd = reduce_stddev(d_rets, n_valid, mean);
    float sharpe = sd > 1e-9f ? mean / sd * std::sqrt(252.0f) : 0.0f;

    if (d_preds != out_preds) {
      CHECK_CUDA(cudaFree(const_cast<float*>(d_preds)));
    }

    std::printf("[iter %3d] total_return=%+.4f  sharpe=%+.3f\n",
                iter, total_return, sharpe);

    if (total_return > best_metric) {
      best_metric = total_return;
      best_iter = iter;
      stale = 0;
    } else if (++stale >= early_stop_patience) {
      std::printf("[early stop] no improvement for %d rounds; best iter=%d\n",
                  early_stop_patience, best_iter);
      break;
    }
  }

  // ── 4. Slice the booster at the best-scoring round and save ─────────
  BoosterHandle sliced = nullptr;
  CHECK_XGB(XGBoosterSlice(booster, 0, best_iter + 1, 1, &sliced));
  CHECK_XGB(XGBoosterSaveModel(sliced, "xgb_cuda_market_sim.json"));

  std::printf(
      "\n[done] best validation total_return=%+.4f at iter=%d\n"
      "[done] saved xgb_cuda_market_sim.json\n",
      best_metric, best_iter);

  // ── 5. Cleanup ──────────────────────────────────────────────────────
  XGBoosterFree(sliced);
  XGBoosterFree(booster);
  XGDMatrixFree(dtrain);
  cudaFree(d_positions);
  cudaFree(d_rets);
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}
