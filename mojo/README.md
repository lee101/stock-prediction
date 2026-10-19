# Timestral-209M in Mojo

GPU forward pass for the univariate, one-step-ahead case (context of L patches x 32 log returns -> 21 quantiles of the
next return), written in Mojo 1.2 with custom kernels. Verified against the official `timestral` package.

## What is custom
- Tensor-core GEMMs: `max`/`linalg` multistage kernel with a tuned tile config (128x128x32, warp 64x64, 3 stages;
  ~76-80 TFLOPS fp16 on the 3090 Ti, the fp32-accumulate ceiling). cuBLAS gave the same times.
- Fused kernels: causal Welford stats + asinh patch normalisation, embed + RMSNorm, residual-add + RMSNorm,
  RoPE + causal attention + sigmoid gate (16-byte vector loads), variate-gate (variate attention over a single key is the
  identity on v, so only the v and gate projections are computed), SwiGLU, quantile head + denormalisation + symmetric averaging.
- Pruned last layer: only the forecast-query token feeds the head, so after the K/V projection the last layer runs on R rows, not 17R.
- Context length is a compile-time parameter (`LPATCH` = 8/16/24/32 -> context 256/512/768/1024).

## Build / run
    uv venv mojo/.venv && source mojo/.venv/bin/activate
    uv pip install mojo && uv pip install --prerelease allow --extra-index-url https://whl.modular.com/nightly/simple/ --index-strategy unsafe-best-match max
    python mojo/export_weights.py            # weights + test batches + fp32 oracle into mojo/data (git-ignored)
    for l in 8 16 24 32; do LPATCH=$l mojo/build.sh; done
    mojo/build/tsfm16_L16 512 0 10           # B=512, no symmetric averaging, 10 iters: correctness vs oracle + us/window + stage profile
    python mojo/eval_e2e.py 768              # official vs Mojo on real held-out windows, same metrics

`mojo/mojo_timestral.py: forecast(Z[n, ctx], sa)` calls the binary in file mode (reloads weights per call).
Contexts must be full (no NaN left padding); `tsfm` = bf16, `tsfm16` = fp16 (use fp16: ~10x lower error, same speed).

## Results (RTX 3090 Ti, us per window, B=512, one-step forecast)
| ctx | official fp32 | Mojo fp16 | speedup | official sym-avg | Mojo sym-avg |
|---|---|---|---|---|---|
| 256 | 346 | 67 | 5.2x | 700 | 126 |
| 512 | 560 | 116 | 4.8x | 1138 | 225 |
| 768 | 797 | 173 | 4.6x | 1589 | 338 |
| 1024 | 1025 | 232 | 4.4x | 2055 | 454 |

Reference points at ctx 512: torch reimplementation fp16 eager 280, `torch.compile` (max-autotune) fp16 140-153.
Mojo at B=1024 reaches ~111 us/window. Accuracy vs the fp32 official model: median-quantile error ~3e-6 (signal std ~1.1e-3);
on 5600 held-out windows (ctx 768, log-return input) MAE-vs-zero 1.00124 / dir-acc 0.5041 / IC -0.0199 are identical for both.

## Findings
- GEMMs are ~73% of the time and already at the fp16 tensor-core ceiling; bf16/fp16 and cuBLAS/Mojo kernels tie.
- W8A8 int8 (per-token / per-channel) is too lossy here: corr with the fp32 median drops to 0.94, so it was not pursued.
- Price-level input in fp16 loses information (corr 0.993 with fp32); log-return input is lossless (0.999996).
- Fusing SwiGLU into the GEMM epilogue is not possible with the shipped `_matmul_gpu` (it only exposes a value-transforming epilogue).
