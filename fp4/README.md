# fp4

Minimal reference NVFP4 (E2M1) training library. See
`/home/administrator/.claude/plans/cozy-coalescing-cerf.md` for the full plan.

This package contains a pure-PyTorch reference implementation that runs on CPU
or any CUDA GPU. Real Blackwell NVFP4 GEMM kernels are stubbed in
`fp4/kernels/` and will wrap the CUTLASS example 79b when compiled.
