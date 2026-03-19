#include <torch/extension.h>
#include <vector>
#include <cmath>

namespace {

// CPU fallback: pure PyTorch ops, matching the Chronos2 preprocessing pipeline.
std::vector<torch::Tensor> fused_preprocess_cpu(
    torch::Tensor context,
    int patch_size,
    int context_length,
    bool use_arcsinh
) {
    auto ctx = context.contiguous().to(torch::kFloat32);
    int64_t B = ctx.size(0);
    int64_t L = ctx.size(1);

    // Truncate if longer than model's context_length
    if (L > context_length) {
        ctx = ctx.slice(1, L - context_length, L).contiguous();
        L = context_length;
    }

    // 1. Build NaN mask
    auto nan_mask = torch::isnan(ctx);
    auto valid_mask = nan_mask.logical_not().to(torch::kFloat32);  // (B, L)

    // 2. Instance normalization: nanmean and nanstd
    auto ctx_zero = torch::where(nan_mask, torch::zeros_like(ctx), ctx);
    auto count = valid_mask.sum(/*dim=*/ -1, /*keepdim=*/ true);  // (B, 1)
    auto count_safe = torch::where(count > 0, count, torch::ones_like(count));

    auto loc = ctx_zero.sum(/*dim=*/ -1, /*keepdim=*/ true) / count_safe;  // (B, 1)
    // Where count==0, mean should be 0
    loc = torch::where(count > 0, loc, torch::zeros_like(loc));

    auto diff = torch::where(nan_mask, torch::zeros_like(ctx), ctx - loc);
    auto variance = (diff * diff).sum(/*dim=*/ -1, /*keepdim=*/ true) / count_safe;
    auto scale = torch::sqrt(variance);  // (B, 1)
    // Where count==0, std should be 1
    scale = torch::where(count > 0, scale, torch::ones_like(scale));
    // Clamp: if std is 0, use eps
    auto eps = torch::full_like(scale, 1e-5);
    scale = torch::where(scale == 0.0, eps, scale);

    // Normalize
    auto normalized = (ctx - loc) / scale;
    if (use_arcsinh) {
        normalized = torch::arcsinh(normalized);
    }
    // Fill NaN with 0
    normalized = torch::where(nan_mask, torch::zeros_like(normalized), normalized);
    auto context_mask = valid_mask;

    // 3. Patching: left-pad to multiple of patch_size
    int64_t L_padded = L;
    if (L % patch_size != 0) {
        L_padded = L + (patch_size - (L % patch_size));
        int64_t pad_len = L_padded - L;
        auto pad_zeros = torch::zeros({B, pad_len}, normalized.options());
        normalized = torch::cat({pad_zeros, normalized}, /*dim=*/ 1);
        auto pad_mask = torch::zeros({B, pad_len}, context_mask.options());
        context_mask = torch::cat({pad_mask, context_mask}, /*dim=*/ 1);
    }

    int num_patches = (int)(L_padded / patch_size);

    // Unfold: (B, L_padded) -> (B, num_patches, patch_size)
    auto patched_context = normalized.unfold(/*dimension=*/ 1, /*size=*/ patch_size, /*step=*/ patch_size);
    auto patched_mask = context_mask.unfold(/*dimension=*/ 1, /*size=*/ patch_size, /*step=*/ patch_size);
    patched_context = torch::where(patched_mask > 0.0, patched_context, torch::zeros_like(patched_context));

    // attention_mask: 1 if any item in patch is valid
    auto attn_mask = (patched_mask.sum(/*dim=*/ -1) > 0).to(torch::kFloat32);  // (B, P)

    // 4. Time encoding
    int64_t final_ctx_len = (int64_t)num_patches * patch_size;
    auto time_enc = torch::arange(-final_ctx_len, 0, torch::TensorOptions().dtype(torch::kFloat32).device(ctx.device()));
    time_enc = time_enc.reshape({1, num_patches, patch_size}).expand({B, num_patches, patch_size}).clone();
    time_enc = time_enc / (float)context_length;

    // 5. Concatenate: [time_enc, patched_context, patched_mask] along last dim
    auto patched_out = torch::cat({time_enc, patched_context, patched_mask}, /*dim=*/ 2);

    return {patched_out, attn_mask, loc, scale};
}

}  // namespace

#ifdef WITH_CUDA
std::vector<torch::Tensor> fused_preprocess_cuda(
    torch::Tensor context,
    int patch_size,
    int context_length,
    bool use_arcsinh
);
#endif

std::vector<torch::Tensor> fused_preprocess(
    torch::Tensor context,
    int patch_size,
    int context_length,
    bool use_arcsinh
) {
    TORCH_CHECK(context.dim() == 2, "context must be 2D (B, L), got ", context.dim(), "D");

    if (context.is_cuda()) {
#ifdef WITH_CUDA
        return fused_preprocess_cuda(context, patch_size, context_length, use_arcsinh);
#else
        TORCH_CHECK(false, "fused_preprocess: CUDA tensor provided but extension was built without CUDA support.");
#endif
    }
    return fused_preprocess_cpu(context, patch_size, context_length, use_arcsinh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_preprocess", &fused_preprocess,
          "Fused Chronos2 preprocessing: normalize + patch + time_enc (CPU/CUDA)");
}
