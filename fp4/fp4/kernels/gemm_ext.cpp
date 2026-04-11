// Pybind11 entry point for the fp4 NVFP4 GEMM torch extension.
// The actual CUDA + CUTLASS implementation lives in gemm_kernel.cu.

#include <torch/extension.h>

torch::Tensor nvfp4_matmul_bf16(torch::Tensor A, torch::Tensor B);
bool fp4_have_blackwell();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_matmul_bf16", &nvfp4_matmul_bf16,
        "NVFP4xNVFP4 -> BF16 GEMM (Blackwell SM120). Inputs are bf16, "
        "quantized internally to NVFP4 with E4M3 block scales.",
        py::arg("a"), py::arg("b"));
  m.attr("HAVE_BLACKWELL") = fp4_have_blackwell();
}
