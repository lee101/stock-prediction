// TODO: thin pybind/torch extension wrapping the CUTLASS Blackwell NVFP4 GEMM
// example located at:
//   external/flash-attention/csrc/cutlass/examples/79_blackwell_geforce_gemm/
//     79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu
//
// For now this file is intentionally empty so the python fallback in gemm.py
// is used. The CUDA build will be wired up by a follow-up task once the
// reference path is validated end to end.
