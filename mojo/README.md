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


# Fine-tuning (training) in Mojo

`mojo/src/train.mojo` is a full training step for Timestral-209M: forward with saved activations, a hand-written backward
for every op, a pinball loss on the 32-step x 21-quantile head, and a fused AdamW with fp32 master weights and global-norm
clipping. Compute dtype is a compile-time choice (fp32 for validation, bf16 or fp16 for speed).

## Design
- Loss: pinball in denormalised log-return space; step 0 has weight 1, steps 1..31 share weight `aux`=0.5; normalised by 0.02.
  `mojo/train_ref.py` is the PyTorch autograd oracle of exactly this loss.
- Forward saves per layer: residual stream (fp32), norm outputs, qkv+gate, attention output before/after the gate, variate
  branch, MLP pre-activations and the SwiGLU output. Backward recomputes only the 17x17 attention probabilities.
- Backward kernels: attention + gate + RoPE (one block per row/head, fp32 softmax backward), RMSNorm (fused with the bf16
  cast of the gradient and a partial-sum weight gradient), SwiGLU, variate gate, embedding, quantile head (monotone cumsum,
  softplus, sinh denormalisation).
- GEMMs: forward (NT) and input-gradient (NN) use the Modular multistage tensor-core kernel with per-shape tile configs
  (128x128x32/3 stages for wide N, 128x64x32/4 stages otherwise; 10-20% over the default dispatch); weight gradients (TN, fp32
  output, no zeroing needed) use cuBLAS. fp32 validation builds use cuBLAS everywhere.
- Optimizer: all parameters live in one flat buffer (matrices first, then norms/biases), so the clip norm is one kernel,
  AdamW is one 4-wide vectorised kernel (~88% of DRAM bandwidth) that also refreshes the bf16 compute copy.
- Not done: gradient accumulation across micro-batches (matrix gradients are overwritten), last-layer pruning in training,
  contexts above 768 (the attention-backward kernel needs shared memory for all S x S blocks).

## Correctness
- Gradients: fp32 build vs PyTorch fp32 autograd on 64 real windows, all 173 parameter tensors: relative error 1e-5 to 5e-4,
  cosine similarity 1.000000 (`mojo/compare_grads.py`). bf16 itself is noisy: bf16-vs-fp32 autograd differ by ~20% on interior
  weights, so only the fp32 build can validate the maths.
- Training curves: Mojo bf16 and PyTorch bf16 with identical batches, schedule and clipping give train loss 0.26895 vs 0.26882
  at step 50 and validation loss 0.32336 vs 0.32338 at step 300 (fp16 with loss scale 1024: 0.32338).

## Build / run
    python mojo/export_train.py 512                  # fp32 master weights, oracle gradients, train/val windows
    python -c "import numpy as np; np.random.default_rng(123).integers(0,32768,size=(4000,512),dtype=np.int32).tofile('mojo/data/train/train_idx.i32')"
    mojo/build_train.sh                              # train32_L16 (fp32), train16_L16 (fp16), trainbf_L16 (bf16)
    mojo/build/train32_L16 gradcheck 64 && python mojo/compare_grads.py 512
    mojo/build/trainbf_L16 train 64 300 2e-5 50      # B steps lr eval_every [loss_scale]
    mojo/build/trainbf_L16 bench 256 15 2e-5 100 1   # step time (min/median); append `prof` for phase timers
    python mojo/bench_train_torch.py eager|compile 256 15

## Training throughput (RTX 3090 Ti, bf16 compute, fp32 master/Adam, ctx 512, minimum step time on a quiet GPU)
| batch | Mojo ms/step (us/sample) | torch.compile (max-autotune) | PyTorch eager | vs compile | vs eager |
|---|---|---|---|---|---|
| 128 | 63.5 (496) | 75.5 (590) | 209 (1634) | 1.19x | 3.3x |
| 256 | 108.2 (423) | 126.7 (495) | 273 (1066) | 1.17x | 2.5x |
| 512 | 190.5 (372) | 233.3 (456) | 501 (978) | 1.22x | 2.6x |

A model-FLOP count of 20.1 GFLOP per sample (3x the forward) gives 54 TFLOPS at B=512, about 67% of the fp16 tensor-core peak.
Kernel profile at B=256 (nsys): GEMMs 72% of the step at 64 TFLOPS effective, AdamW 6.3%, attention backward 5.9%,
other elementwise ~16%. The GPU is shared with other jobs, so numbers are minimum-of-N step times; interleaved A/B runs under
contention kept the same ordering.
