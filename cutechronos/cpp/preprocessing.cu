#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>
#include <cmath>

namespace {

// Phase 1: Compute per-series nanmean and nanstd using shared memory reduction.
// One block per series.
template <typename scalar_t>
__global__ void compute_loc_scale_kernel(
    const scalar_t* __restrict__ context,  // (B, L)
    float* __restrict__ loc,               // (B,)
    float* __restrict__ scale,             // (B,)
    int64_t L
) {
    extern __shared__ float smem[];
    float* s_sum = smem;                // blockDim.x floats
    float* s_count = smem + blockDim.x; // blockDim.x floats
    float* s_sq = smem + 2 * blockDim.x; // blockDim.x floats

    int b = blockIdx.x;
    int tid = threadIdx.x;
    const scalar_t* series = context + b * L;

    // First pass: compute sum and count for nanmean
    float local_sum = 0.0f;
    float local_count = 0.0f;
    for (int64_t i = tid; i < L; i += blockDim.x) {
        float val = static_cast<float>(series[i]);
        if (!isnan(val)) {
            local_sum += val;
            local_count += 1.0f;
        }
    }
    s_sum[tid] = local_sum;
    s_count[tid] = local_count;
    __syncthreads();

    // Parallel reduction for sum and count
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }

    float mean_val = 0.0f;
    float total_count = s_count[0];
    if (total_count > 0.0f) {
        mean_val = s_sum[0] / total_count;
    }

    // Broadcast mean to shared memory for second pass
    if (tid == 0) {
        s_sum[0] = mean_val;  // reuse s_sum[0] to store the mean
    }
    __syncthreads();
    mean_val = s_sum[0];

    // Second pass: compute sum of squared deviations for nanstd
    float local_sq = 0.0f;
    for (int64_t i = tid; i < L; i += blockDim.x) {
        float val = static_cast<float>(series[i]);
        if (!isnan(val)) {
            float diff = val - mean_val;
            local_sq += diff * diff;
        }
    }
    s_sq[tid] = local_sq;
    __syncthreads();

    // Parallel reduction for squared deviations
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        loc[b] = mean_val;
        float std_val = 1.0f;
        if (total_count > 0.0f) {
            std_val = sqrtf(s_sq[0] / total_count);
        }
        // Clamp: if std is 0, use eps (matching InstanceNorm which uses 1e-5)
        if (std_val == 0.0f) {
            std_val = 1e-5f;
        }
        scale[b] = std_val;
    }
}

// Phase 2: Fused normalize + patch + time_encoding + concat
// Grid: (num_patches, B), each thread handles one element within a patch.
// Output: patched_context (B, P, 3*patch_size) = [time_enc, normalized, mask]
//         attention_mask (B, P) = 1 if any non-NaN in patch
template <typename scalar_t>
__global__ void fused_patch_kernel(
    const scalar_t* __restrict__ context,   // (B, L_orig)
    const float* __restrict__ loc,          // (B,)
    const float* __restrict__ scale,        // (B,)
    float* __restrict__ patched_out,        // (B, P, 3*patch_size)
    float* __restrict__ attn_mask,          // (B, P)
    int64_t L_orig,          // original context length
    int64_t L_padded,        // padded context length (multiple of patch_size)
    int patch_size,
    int num_patches,
    int time_encoding_scale, // divisor for time encoding
    bool use_arcsinh
) {
    int p = blockIdx.x;  // patch index
    int b = blockIdx.y;  // batch index
    int t = threadIdx.x; // position within patch [0, patch_size)

    if (t >= patch_size) return;

    // Global position in the padded context
    int64_t padded_pos = (int64_t)p * patch_size + t;
    // Offset into original context (padded_pos may be in the left-padding region)
    int64_t orig_pos = padded_pos - (L_padded - L_orig);

    float mean_val = loc[b];
    float std_val = scale[b];

    // Determine if this position is NaN or in padding region
    bool is_valid = false;
    float normalized = 0.0f;
    float mask_val = 0.0f;

    if (orig_pos >= 0 && orig_pos < L_orig) {
        float raw = static_cast<float>(context[b * L_orig + orig_pos]);
        if (!isnan(raw)) {
            is_valid = true;
            normalized = (raw - mean_val) / std_val;
            if (use_arcsinh) {
                normalized = asinhf(normalized);
            }
            mask_val = 1.0f;
        }
    }
    // If not valid (padding or NaN), normalized=0, mask=0

    // Output layout: patched_out[b, p, :] = [time_enc(patch_size), normalized(patch_size), mask(patch_size)]
    int64_t out_base = ((int64_t)b * num_patches + p) * (3 * patch_size);

    // Time encoding: position in [-final_context_length, 0) / time_encoding_scale
    int64_t final_context_length = (int64_t)num_patches * patch_size;
    float time_val = (float)(padded_pos - final_context_length) / (float)time_encoding_scale;
    patched_out[out_base + t] = time_val;
    patched_out[out_base + patch_size + t] = normalized;
    patched_out[out_base + 2 * patch_size + t] = mask_val;

    // Compute attention_mask: 1 if any element in this patch is valid
    // Use shared memory for a simple OR reduction within the patch
    extern __shared__ int s_any_valid[];
    // We use one int per patch in the block. Since blockIdx covers one patch,
    // we just need one shared int.
    if (t == 0) {
        s_any_valid[0] = 0;
    }
    __syncthreads();

    if (is_valid) {
        atomicOr(&s_any_valid[0], 1);
    }
    __syncthreads();

    if (t == 0) {
        attn_mask[(int64_t)b * num_patches + p] = (s_any_valid[0] > 0) ? 1.0f : 0.0f;
    }
}

}  // namespace

std::vector<torch::Tensor> fused_preprocess_cuda(
    torch::Tensor context,
    int patch_size,
    int context_length,
    bool use_arcsinh
) {
    TORCH_CHECK(context.is_cuda(), "context must be a CUDA tensor");
    TORCH_CHECK(context.dim() == 2, "context must be 2D (B, L)");

    auto ctx = context.contiguous();
    int64_t B = ctx.size(0);
    int64_t L = ctx.size(1);

    // Truncate if longer than model's context_length
    if (L > context_length) {
        ctx = ctx.slice(1, L - context_length, L).contiguous();
        L = context_length;
    }

    // Compute padded length (multiple of patch_size)
    int64_t L_padded = L;
    if (L % patch_size != 0) {
        L_padded = L + (patch_size - (L % patch_size));
    }
    int num_patches = (int)(L_padded / patch_size);

    // Use context_length as time_encoding_scale (matching Chronos2 default)
    int time_encoding_scale = context_length;

    // Allocate outputs
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(ctx.device());
    auto loc = torch::empty({B}, opts_f32);
    auto scale_out = torch::empty({B}, opts_f32);
    auto patched = torch::empty({B, num_patches, 3 * patch_size}, opts_f32);
    auto attn_mask = torch::empty({B, num_patches}, opts_f32);

    auto stream = at::cuda::getDefaultCUDAStream();

    // Phase 1: Compute loc and scale
    {
        int threads = 256;
        if (L < 256) threads = 128;
        if (L < 128) threads = 64;
        // Ensure threads is power of 2 for reduction
        int smem = 3 * threads * sizeof(float);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            ctx.scalar_type(),
            "compute_loc_scale",
            [&] {
                compute_loc_scale_kernel<scalar_t><<<B, threads, smem, stream>>>(
                    ctx.data_ptr<scalar_t>(),
                    loc.data_ptr<float>(),
                    scale_out.data_ptr<float>(),
                    L
                );
            }
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Phase 2: Fused patch + normalize + time_enc + concat
    {
        int threads = patch_size;
        // Round up to nearest warp for efficiency
        if (threads < 32) threads = 32;
        dim3 grid(num_patches, B);
        int smem = sizeof(int);  // for the atomicOr reduction
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            ctx.scalar_type(),
            "fused_patch",
            [&] {
                fused_patch_kernel<scalar_t><<<grid, threads, smem, stream>>>(
                    ctx.data_ptr<scalar_t>(),
                    loc.data_ptr<float>(),
                    scale_out.data_ptr<float>(),
                    patched.data_ptr<float>(),
                    attn_mask.data_ptr<float>(),
                    L,
                    L_padded,
                    patch_size,
                    num_patches,
                    time_encoding_scale,
                    use_arcsinh
                );
            }
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Reshape loc and scale to (B, 1) to match InstanceNorm output
    loc = loc.unsqueeze(1);
    scale_out = scale_out.unsqueeze(1);

    return {patched, attn_mask, loc, scale_out};
}
