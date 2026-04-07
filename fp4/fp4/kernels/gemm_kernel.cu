// CUTLASS Blackwell SM120 NVFP4 x NVFP4 -> BF16 GEMM, wrapped as a torch
// extension. Adapted from
//   external/flash-attention/csrc/cutlass/examples/79_blackwell_geforce_gemm/
//     79a_blackwell_geforce_nvfp4_bf16_gemm.cu
//
// We expose two entry points to torch:
//
//   nvfp4_matmul_bf16(a_bf16, b_bf16) -> bf16
//       Quantizes A and B internally to NVFP4 (E2M1) with E4M3 per-16-element
//       block scales, lays them out in the CUTLASS-native interleaved scale
//       factor format, runs the GEMM, and returns BF16 (M, N).  This is the
//       path used by the python `nvfp4_matmul` wrapper and the bench harness.
//
//   nvfp4_gemm(a_codes, sfa, b_codes, sfb, m, n, k) -> bf16
//       Advanced entry: caller has already produced packed uint8 FP4 codes
//       (two values per byte, K-major) and the SFA / SFB tensors in the
//       layout reported by `nvfp4_sf_shape(m, n, k)`.  Useful for fused
//       activation/quant pipelines.
//
// Both paths return a contiguous (M, N) BF16 tensor.  Build is lazy via
// torch.utils.cpp_extension.load() so importing fp4 on a non-Blackwell box
// never triggers a CUDA compile.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

// ---------------------------------------------------------------------------
// CUTLASS GEMM type (NVFP4 x NVFP4 -> BF16, SM120 GeForce Blackwell)
// ---------------------------------------------------------------------------
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#define FP4_HAVE_BLACKWELL 1
#else
#define FP4_HAVE_BLACKWELL 0
#endif

#if FP4_HAVE_BLACKWELL

using namespace cute;

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128,_128,_128>;
using ClusterShape     = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ---------------------------------------------------------------------------
// Quantization kernels (BF16 -> NVFP4 + interleaved E4M3 scales)
//
// Both A (row-major M x K) and B (col-major N x K, i.e. logical (K x N) but
// stored as N rows of K) are quantized along K in 16-element blocks.
// ---------------------------------------------------------------------------

// E2M1 absolute representable values (positive half), in order of code 0..7.
//   sign bit | exponent (2) | mantissa (1)
//   0 000 0  -> 0.0       0 100 0 -> 1.0
//   0 001 0  -> 0.5       0 101 0 -> 1.5
//   0 010 0  -> 1.0??     -- actual NVFP4 (E2M1, bias=1) values:
//   The canonical NVFP4 magnitudes are: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
__device__ __forceinline__ uint8_t encode_e2m1_abs(float v) {
  // v >= 0
  // pick nearest of {0, 0.5, 1, 1.5, 2, 3, 4, 6}
  if (v < 0.25f) return 0;
  if (v < 0.75f) return 1;        // 0.5
  if (v < 1.25f) return 2;        // 1.0
  if (v < 1.75f) return 3;        // 1.5
  if (v < 2.5f)  return 4;        // 2.0
  if (v < 3.5f)  return 5;        // 3.0
  if (v < 5.0f)  return 6;        // 4.0
  return 7;                       // 6.0
}

__device__ __forceinline__ uint8_t encode_e2m1(float v) {
  uint8_t mag = encode_e2m1_abs(fabsf(v));
  uint8_t sign = (v < 0.0f) ? 0x8 : 0x0;
  return sign | mag;
}

// Quantize A: shape (M, K) row-major bf16. Output:
//   a_codes: (M, K/2) uint8, two FP4 nibbles per byte (K-major, low nibble = even k)
//   a_sf_raw: (M, K/16) uint8 (E4M3 raw bytes, packed in M-major order)
__global__ void quantize_a_kernel(
    const __nv_bfloat16* __restrict__ A,
    uint8_t* __restrict__ A_codes,
    uint8_t* __restrict__ A_sf_raw,
    int M, int K)
{
  int row   = blockIdx.y * blockDim.y + threadIdx.y;
  int block = blockIdx.x * blockDim.x + threadIdx.x;   // block index along K
  int K_blocks = K / 16;
  if (row >= M || block >= K_blocks) return;

  const __nv_bfloat16* row_ptr = A + row * K + block * 16;
  float vals[16];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float v = __bfloat162float(row_ptr[i]);
    vals[i] = v;
    float a = fabsf(v);
    if (a > amax) amax = a;
  }
  // E2M1 max magnitude is 6.0; scale = amax / 6
  float scale = amax / 6.0f;
  if (!(scale > 0.f)) scale = 1.f;
  // E4M3 round-to-nearest of `scale` via __nv_fp8_e4m3 cast (CUDA 12+)
  __nv_fp8_storage_t sf_byte = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
  // Reconstruct the actual value the kernel will use, so codes are quantized
  // against the same scale CUTLASS will dequantize with.
  float scale_used = static_cast<float>(__half2float(__nv_cvt_fp8_to_halfraw(sf_byte, __NV_E4M3)));
  if (!(scale_used > 0.f)) scale_used = scale;
  float inv = 1.0f / scale_used;

  uint8_t packed[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t lo = encode_e2m1(vals[2*i]   * inv);
    uint8_t hi = encode_e2m1(vals[2*i+1] * inv);
    packed[i] = (uint8_t)((hi << 4) | (lo & 0xF));
  }
  uint8_t* code_ptr = A_codes + row * (K / 2) + block * 8;
  #pragma unroll
  for (int i = 0; i < 8; ++i) code_ptr[i] = packed[i];

  // Raw scale: write linearly (M, K/16). The proper interleaved layout is
  // computed by host code; this kernel only emits the dense raw form, then
  // a separate scatter step rewrites into CUTLASS layout.
  A_sf_raw[row * K_blocks + block] = sf_byte;
}

// Same for B: logical (K, N), stored col-major as (N, K) — so we treat B as a
// row-major (N, K) tensor for the purposes of quantization.
__global__ void quantize_b_kernel(
    const __nv_bfloat16* __restrict__ B,    // (N, K) row-major view
    uint8_t* __restrict__ B_codes,
    uint8_t* __restrict__ B_sf_raw,
    int N, int K)
{
  int col   = blockIdx.y * blockDim.y + threadIdx.y;   // n
  int block = blockIdx.x * blockDim.x + threadIdx.x;
  int K_blocks = K / 16;
  if (col >= N || block >= K_blocks) return;

  const __nv_bfloat16* row_ptr = B + col * K + block * 16;
  float vals[16];
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float v = __bfloat162float(row_ptr[i]);
    vals[i] = v;
    float a = fabsf(v);
    if (a > amax) amax = a;
  }
  float scale = amax / 6.0f;
  if (!(scale > 0.f)) scale = 1.f;
  __nv_fp8_storage_t sf_byte = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
  float scale_used = static_cast<float>(__half2float(__nv_cvt_fp8_to_halfraw(sf_byte, __NV_E4M3)));
  if (!(scale_used > 0.f)) scale_used = scale;
  float inv = 1.0f / scale_used;

  uint8_t* code_ptr = B_codes + col * (K / 2) + block * 8;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t lo = encode_e2m1(vals[2*i]   * inv);
    uint8_t hi = encode_e2m1(vals[2*i+1] * inv);
    code_ptr[i] = (uint8_t)((hi << 4) | (lo & 0xF));
  }
  B_sf_raw[col * K_blocks + block] = sf_byte;
}

// Scatter the dense (rows x K_blocks) raw scale buffer into the
// CUTLASS-native interleaved layout via the host-resolved cute layout.
// We materialize the layout on host (it's tiny) and copy a coordinate map
// into a device buffer once per call.
__global__ void scatter_sf_kernel(
    const uint8_t* __restrict__ raw,
    const int32_t* __restrict__ map,    // map[row * K_blocks + b] = dst_offset
    uint8_t* __restrict__ dst,
    int n_elems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_elems) return;
  dst[map[i]] = raw[i];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#include <unordered_map>
#include <mutex>

struct SfMapCache {
  std::mutex mu;
  std::unordered_map<uint64_t, torch::Tensor> a_maps;
  std::unordered_map<uint64_t, torch::Tensor> b_maps;
};
static SfMapCache& sf_cache() { static SfMapCache c; return c; }
static inline uint64_t shape_key(int x, int y, int z) {
  return (uint64_t)x * 1000003ull * 1000003ull
       + (uint64_t)y * 1000003ull
       + (uint64_t)z;
}

template <class LayoutSF>
static torch::Tensor build_sf_index_map(LayoutSF const& layout, int rows, int K_blocks) {
  // returns int32 (rows*K_blocks,) with linearized destination offsets in
  // the (filter_zeros) CUTLASS-shape SF buffer.
  auto h = torch::empty({rows * K_blocks}, torch::dtype(torch::kInt32));
  int32_t* p = h.data_ptr<int32_t>();
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < K_blocks; ++b) {
      // SFA/SFB layout is over (M/N, K, L) with the inner 16-element K
      // chunk mapped via stride-0 to a single scale slot.  Pick k = b*16
      // so the coord lands in block b.
      auto coord = make_coord(r, b * 16, 0);
      int32_t off = static_cast<int32_t>(layout(coord));
      p[r * K_blocks + b] = off;
    }
  }
  return h.to(torch::kCUDA);
}

template <class L>
static int64_t sf_buffer_size(L const& layout) {
  return static_cast<int64_t>(size(filter_zeros(layout)));
}

#endif // FP4_HAVE_BLACKWELL

// ---------------------------------------------------------------------------
// Public torch entry points
// ---------------------------------------------------------------------------

torch::Tensor nvfp4_matmul_bf16(torch::Tensor A, torch::Tensor B) {
#if !FP4_HAVE_BLACKWELL
  TORCH_CHECK(false, "fp4: CUTLASS NVFP4 GEMM was compiled without SM120/SM121 support");
#else
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A,B must be CUDA tensors");
  TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16,
              "A,B must be bf16");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A,B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "A.size(1) must equal B.size(0)");

  int M = (int)A.size(0);
  int K = (int)A.size(1);
  int N = (int)B.size(1);
  TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");
  TORCH_CHECK(M % 8  == 0 && N % 8 == 0, "M,N must be multiples of 8");

  A = A.contiguous();
  // We want B as (N, K) row-major (= col-major (K, N)) for the kernel.
  torch::Tensor B_nk = B.transpose(0, 1).contiguous();   // (N, K)

  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  torch::Tensor A_codes = torch::empty({M, K / 2}, opts_u8);
  torch::Tensor B_codes = torch::empty({N, K / 2}, opts_u8);

  int K_blocks = K / 16;
  torch::Tensor A_sf_raw = torch::empty({M * K_blocks}, opts_u8);
  torch::Tensor B_sf_raw = torch::empty({N * K_blocks}, opts_u8);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  {
    dim3 block(8, 8);
    dim3 grid((K_blocks + 7) / 8, (M + 7) / 8);
    quantize_a_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        A_codes.data_ptr<uint8_t>(),
        A_sf_raw.data_ptr<uint8_t>(),
        M, K);
  }
  {
    dim3 block(8, 8);
    dim3 grid((K_blocks + 7) / 8, (N + 7) / 8);
    quantize_b_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(B_nk.data_ptr()),
        B_codes.data_ptr<uint8_t>(),
        B_sf_raw.data_ptr<uint8_t>(),
        N, K);
  }

  // Build CUTLASS layouts and scatter SF.
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  int64_t sfa_n = sf_buffer_size(layout_SFA);
  int64_t sfb_n = sf_buffer_size(layout_SFB);
  torch::Tensor SFA = torch::zeros({sfa_n}, opts_u8);
  torch::Tensor SFB = torch::zeros({sfb_n}, opts_u8);

  torch::Tensor map_a, map_b;
  {
    auto& cache = sf_cache();
    std::lock_guard<std::mutex> lk(cache.mu);
    uint64_t ka = shape_key(M, N, K);
    auto it_a = cache.a_maps.find(ka);
    if (it_a == cache.a_maps.end()) {
      map_a = build_sf_index_map(layout_SFA, M, K_blocks);
      cache.a_maps.emplace(ka, map_a);
    } else {
      map_a = it_a->second;
    }
    auto it_b = cache.b_maps.find(ka);
    if (it_b == cache.b_maps.end()) {
      map_b = build_sf_index_map(layout_SFB, N, K_blocks);
      cache.b_maps.emplace(ka, map_b);
    } else {
      map_b = it_b->second;
    }
  }

  {
    int n = M * K_blocks;
    int t = 256;
    scatter_sf_kernel<<<(n + t - 1) / t, t, 0, stream>>>(
        A_sf_raw.data_ptr<uint8_t>(), map_a.data_ptr<int32_t>(),
        SFA.data_ptr<uint8_t>(), n);
  }
  {
    int n = N * K_blocks;
    int t = 256;
    scatter_sf_kernel<<<(n + t - 1) / t, t, 0, stream>>>(
        B_sf_raw.data_ptr<uint8_t>(), map_b.data_ptr<int32_t>(),
        SFB.data_ptr<uint8_t>(), n);
  }

  // Output
  auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(A.device());
  torch::Tensor D = torch::empty({M, N}, opts_bf16);
  torch::Tensor C = torch::zeros({M, N}, opts_bf16);

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {
      reinterpret_cast<typename ElementA::DataType const*>(A_codes.data_ptr()), stride_A,
      reinterpret_cast<typename ElementB::DataType const*>(B_codes.data_ptr()), stride_B,
      reinterpret_cast<typename ElementA::ScaleFactorType const*>(SFA.data_ptr()), layout_SFA,
      reinterpret_cast<typename ElementB::ScaleFactorType const*>(SFB.data_ptr()), layout_SFB
    },
    {
      {1.0f, 0.0f},
      reinterpret_cast<ElementC const*>(C.data_ptr()), stride_C,
      reinterpret_cast<ElementD*>(D.data_ptr()),       stride_D
    }
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(arguments);
  auto opts_ws = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  torch::Tensor workspace = torch::empty({(int64_t)ws_size}, opts_ws);

  cutlass::Status st = gemm.can_implement(arguments);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass can_implement failed");
  st = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass initialize failed");
  st = gemm.run(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess, "cutlass run failed");

  return D;
#endif
}

// pybind module is registered in gemm_ext.cpp.
bool fp4_have_blackwell() {
#if FP4_HAVE_BLACKWELL
  return true;
#else
  return false;
#endif
}
