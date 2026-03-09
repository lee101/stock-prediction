#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void abs_diff_kernel(
    const scalar_t* __restrict__ pred,
    const scalar_t* __restrict__ tgt,
    float* __restrict__ out,
    int64_t n
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p = static_cast<float>(pred[idx]);
        float t = static_cast<float>(tgt[idx]);
        out[idx] = fabsf(p - t);
    }
}

template <typename scalar_t>
__global__ void weighted_abs_diff_kernel(
    const scalar_t* __restrict__ pred,
    const scalar_t* __restrict__ tgt,
    const scalar_t* __restrict__ weights,
    float* __restrict__ out,
    int64_t n,
    int64_t horizon
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p = static_cast<float>(pred[idx]);
        float t = static_cast<float>(tgt[idx]);
        float w = static_cast<float>(weights[idx % horizon]);
        out[idx] = fabsf(p - t) * w;
    }
}

}  // namespace

torch::Tensor fast_mae_cuda(torch::Tensor prediction, torch::Tensor target) {
    auto pred = prediction.contiguous();
    auto tgt = target.contiguous();
    TORCH_CHECK(pred.is_cuda(), "prediction must be CUDA tensor");
    TORCH_CHECK(tgt.is_cuda(), "target must be CUDA tensor");

    int64_t n = pred.numel();
    auto out = torch::empty({n}, pred.options().dtype(torch::kFloat));

    constexpr int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, pred.scalar_type(), "fast_mae_cuda", [&] {
        abs_diff_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            pred.data_ptr<scalar_t>(),
            tgt.data_ptr<scalar_t>(),
            out.data_ptr<float>(),
            n
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out.mean();
}

torch::Tensor fast_weighted_mae_cuda(torch::Tensor prediction, torch::Tensor target, torch::Tensor weights) {
    auto pred = prediction.contiguous();
    auto tgt = target.contiguous();
    auto w = weights.contiguous().to(pred.device()).to(pred.scalar_type());

    TORCH_CHECK(pred.is_cuda(), "prediction must be CUDA tensor");
    TORCH_CHECK(tgt.is_cuda(), "target must be CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(w.dim() == 1, "weights must be a 1D tensor");
    TORCH_CHECK(pred.dim() >= 1, "prediction must have at least 1 dimension");
    TORCH_CHECK(pred.size(-1) == w.size(0), "weights length must match prediction.size(-1)");

    int64_t n = pred.numel();
    int64_t horizon = w.size(0);
    auto out = torch::empty({n}, pred.options().dtype(torch::kFloat));

    constexpr int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        pred.scalar_type(),
        "fast_weighted_mae_cuda",
        [&] {
            weighted_abs_diff_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
                pred.data_ptr<scalar_t>(),
                tgt.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                out.data_ptr<float>(),
                n,
                horizon
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out.mean();
}
