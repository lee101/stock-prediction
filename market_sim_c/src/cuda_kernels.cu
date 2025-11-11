#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace market_sim {

// Fast parallel portfolio value calculation
__global__ void compute_portfolio_values_kernel(
    const float* __restrict__ cash,
    const float* __restrict__ positions,
    const float* __restrict__ prices,
    float* __restrict__ portfolio_values,
    int num_agents,
    int num_assets
) {
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (agent_idx < num_agents) {
        float value = cash[agent_idx];

        for (int i = 0; i < num_assets; i++) {
            value += positions[agent_idx * num_assets + i] * prices[i];
        }

        portfolio_values[agent_idx] = value;
    }
}

// Vectorized reward calculation with GAE (Generalized Advantage Estimation)
__global__ void compute_gae_kernel(
    const float* __restrict__ rewards,
    const float* __restrict__ values,
    const float* __restrict__ dones,
    float* __restrict__ advantages,
    float* __restrict__ returns,
    float gamma,
    float lambda,
    int num_steps,
    int num_agents
) {
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (agent_idx < num_agents) {
        float gae = 0.0f;

        // Compute GAE backwards through time
        for (int t = num_steps - 1; t >= 0; t--) {
            int idx = t * num_agents + agent_idx;
            int next_idx = (t + 1) * num_agents + agent_idx;

            float delta;
            if (t == num_steps - 1) {
                delta = rewards[idx] - values[idx];
            } else {
                float next_value = dones[idx] ? 0.0f : values[next_idx];
                delta = rewards[idx] + gamma * next_value - values[idx];
            }

            gae = delta + gamma * lambda * (1.0f - dones[idx]) * gae;
            advantages[idx] = gae;
            returns[idx] = gae + values[idx];
        }
    }
}

// Fast technical indicator calculation (RSI)
__global__ void compute_rsi_kernel(
    const float* __restrict__ prices,
    float* __restrict__ rsi,
    int num_assets,
    int history_len,
    int period
) {
    int asset_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (asset_idx < num_assets) {
        float gains = 0.0f;
        float losses = 0.0f;

        for (int i = 1; i < period && i < history_len; i++) {
            float diff = prices[asset_idx * history_len + i] -
                        prices[asset_idx * history_len + i - 1];
            if (diff > 0) {
                gains += diff;
            } else {
                losses -= diff;
            }
        }

        float avg_gain = gains / period;
        float avg_loss = losses / period;

        if (avg_loss < 1e-8f) {
            rsi[asset_idx] = 100.0f;
        } else {
            float rs = avg_gain / avg_loss;
            rsi[asset_idx] = 100.0f - (100.0f / (1.0f + rs));
        }
    }
}

// Parallel Sharpe ratio calculation
__global__ void compute_sharpe_ratios_kernel(
    const float* __restrict__ returns,
    float* __restrict__ sharpe_ratios,
    int num_agents,
    int num_returns
) {
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (agent_idx < num_agents) {
        // Calculate mean
        float mean = 0.0f;
        for (int i = 0; i < num_returns; i++) {
            mean += returns[agent_idx * num_returns + i];
        }
        mean /= num_returns;

        // Calculate std dev
        float variance = 0.0f;
        for (int i = 0; i < num_returns; i++) {
            float diff = returns[agent_idx * num_returns + i] - mean;
            variance += diff * diff;
        }
        variance /= num_returns;
        float std_dev = sqrtf(variance);

        // Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratios[agent_idx] = (std_dev > 1e-8f) ?
            (mean * sqrtf((float)num_returns) / std_dev) : 0.0f;
    }
}

// Wrappers for PyTorch
torch::Tensor compute_portfolio_values_cuda(
    torch::Tensor cash,
    torch::Tensor positions,
    torch::Tensor prices
) {
    int num_agents = cash.size(0);
    int num_assets = positions.size(1);

    auto portfolio_values = torch::zeros({num_agents}, cash.options());

    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;

    compute_portfolio_values_kernel<<<blocks, threads>>>(
        cash.data_ptr<float>(),
        positions.data_ptr<float>(),
        prices.data_ptr<float>(),
        portfolio_values.data_ptr<float>(),
        num_agents,
        num_assets
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return portfolio_values;
}

std::tuple<torch::Tensor, torch::Tensor> compute_gae_cuda(
    torch::Tensor rewards,
    torch::Tensor values,
    torch::Tensor dones,
    float gamma,
    float lambda
) {
    int num_steps = rewards.size(0);
    int num_agents = rewards.size(1);

    auto advantages = torch::zeros_like(rewards);
    auto returns = torch::zeros_like(rewards);

    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;

    compute_gae_kernel<<<blocks, threads>>>(
        rewards.data_ptr<float>(),
        values.data_ptr<float>(),
        dones.data_ptr<float>(),
        advantages.data_ptr<float>(),
        returns.data_ptr<float>(),
        gamma,
        lambda,
        num_steps,
        num_agents
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return std::make_tuple(advantages, returns);
}

torch::Tensor compute_rsi_cuda(
    torch::Tensor prices,
    int period
) {
    int num_assets = prices.size(0);
    int history_len = prices.size(1);

    auto rsi = torch::zeros({num_assets}, prices.options());

    int threads = 256;
    int blocks = (num_assets + threads - 1) / threads;

    compute_rsi_kernel<<<blocks, threads>>>(
        prices.data_ptr<float>(),
        rsi.data_ptr<float>(),
        num_assets,
        history_len,
        period
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return rsi;
}

torch::Tensor compute_sharpe_ratios_cuda(
    torch::Tensor returns
) {
    int num_agents = returns.size(0);
    int num_returns = returns.size(1);

    auto sharpe_ratios = torch::zeros({num_agents}, returns.options());

    int threads = 256;
    int blocks = (num_agents + threads - 1) / threads;

    compute_sharpe_ratios_kernel<<<blocks, threads>>>(
        returns.data_ptr<float>(),
        sharpe_ratios.data_ptr<float>(),
        num_agents,
        num_returns
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return sharpe_ratios;
}

} // namespace market_sim

// PyTorch bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_portfolio_values", &market_sim::compute_portfolio_values_cuda,
          "Compute portfolio values (CUDA)");
    m.def("compute_gae", &market_sim::compute_gae_cuda,
          "Compute Generalized Advantage Estimation (CUDA)");
    m.def("compute_rsi", &market_sim::compute_rsi_cuda,
          "Compute RSI indicators (CUDA)");
    m.def("compute_sharpe_ratios", &market_sim::compute_sharpe_ratios_cuda,
          "Compute Sharpe ratios (CUDA)");
}
