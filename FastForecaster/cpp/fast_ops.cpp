#include <torch/extension.h>

#include <vector>

namespace {

torch::Tensor fast_mae_cpu(torch::Tensor prediction, torch::Tensor target) {
    auto pred = prediction.contiguous();
    auto tgt = target.contiguous();
    return torch::mean(torch::abs(pred - tgt));
}

torch::Tensor fast_mae_cuda(torch::Tensor prediction, torch::Tensor target);
torch::Tensor fast_weighted_mae_cuda(torch::Tensor prediction, torch::Tensor target, torch::Tensor weights);

torch::Tensor fast_weighted_mae_cpu(torch::Tensor prediction, torch::Tensor target, torch::Tensor weights) {
    auto pred = prediction.contiguous();
    auto tgt = target.contiguous();
    TORCH_CHECK(pred.dim() >= 1, "prediction must have at least 1 dimension");
    TORCH_CHECK(pred.sizes() == tgt.sizes(), "prediction and target must have identical shapes");

    auto w = weights.contiguous().to(pred.device()).to(pred.scalar_type());
    TORCH_CHECK(w.dim() == 1, "weights must be a 1D tensor");
    TORCH_CHECK(pred.size(-1) == w.size(0), "weights length must match prediction.size(-1)");

    std::vector<int64_t> view_shape(pred.dim(), 1);
    view_shape.back() = w.size(0);
    auto w_view = w.view(view_shape);
    return torch::mean(torch::abs(pred - tgt) * w_view);
}

torch::Tensor fast_mae(torch::Tensor prediction, torch::Tensor target) {
    TORCH_CHECK(prediction.sizes() == target.sizes(), "prediction and target must have identical shapes");
    if (prediction.is_cuda()) {
#ifdef WITH_CUDA
        return fast_mae_cuda(prediction, target);
#else
        TORCH_CHECK(false, "fast_mae CUDA tensor provided but extension was built without CUDA support.");
#endif
    }
    return fast_mae_cpu(prediction, target);
}

torch::Tensor fast_weighted_mae(torch::Tensor prediction, torch::Tensor target, torch::Tensor weights) {
    TORCH_CHECK(prediction.sizes() == target.sizes(), "prediction and target must have identical shapes");
    TORCH_CHECK(prediction.dim() >= 1, "prediction must have at least 1 dimension");
    TORCH_CHECK(weights.dim() == 1, "weights must be a 1D tensor");
    TORCH_CHECK(prediction.size(-1) == weights.size(0), "weights length must match prediction.size(-1)");

    if (prediction.is_cuda()) {
#ifdef WITH_CUDA
        return fast_weighted_mae_cuda(prediction, target, weights);
#else
        TORCH_CHECK(false, "fast_weighted_mae CUDA tensor provided but extension was built without CUDA support.");
#endif
    }
    return fast_weighted_mae_cpu(prediction, target, weights);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_mae", &fast_mae, "Fast MAE loss (CPU/CUDA)");
    m.def("fast_weighted_mae", &fast_weighted_mae, "Fast weighted MAE loss (CPU/CUDA)");
}
