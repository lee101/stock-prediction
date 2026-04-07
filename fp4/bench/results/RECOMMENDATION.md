# Production stack for RL trading on RTX 5090

_Date: 2026-04-07. Source: Phase 1-5 of the fp4 NVFP4 trading library._

## Recommended stack

| Layer        | Choice                                                                 |
|--------------|------------------------------------------------------------------------|
| Environment  | `gpu_trading_env` (C++/CUDA SoA fused step, SM120, commit `15cdd8b9`)  |
| Fills/fees   | 5bps buffered limit fill, fee=10bps, max_lev=5x, maint margin, lag>=2  |
| Policy       | `TwoLayerPolicy` (`policy_two_layer.py`) — Layer A slow risk, Layer B fast quote |
| Precision    | **BF16 default** for hidden Linears; NVFP4 ready for transformer encoder variants |
| Trainer      | **PPO** (constrained variant disabled by default) — `fp4/trainer.py`   |
| Constraints  | Lagrangian + CVaR + smooth-PnL (`losses.py`, `lagrangian.py`) — opt-in |
| Graph capture| Full rollout+update CUDA graph (`cuda_graph.py`) when shapes are static|
| Eval         | binary-fill marketsim @ slippage {0,5,10,20} bps, lag>=2               |

### Why this combination

- Smoke leaderboard (50k steps, n=2 seeds, sweep `37add767`) ranks
  `pufferlib_bf16 + unconstrained` first on `(p10@5bps, sortino@5bps, max_dd)`.
  SAC and QR-PPO completed but did not pass the |max_dd| <= 0.20 feasibility
  filter at this horizon.
- NVFP4 GEMM is 2.24x BF16 at 4096^3 (`f9806fe0`) and the graph-safe
  `NVFP4Linear` adds +62% trainer SPS (`d8eb1a5b`); this matters once the
  policy grows past the current 2-layer MLP. For the current small policy
  the gain is < env-step time, so BF16 is the simpler default.
- `gpu_trading_env` keeps fills/fees/leverage/liquidation in **one** kernel
  shared with future training and eval, eliminating sim drift.

## Caveats

- **Training horizon**: every Phase 5 number is a 50k-step smoke. Real
  ranking needs the standard ladder: 200k -> 2M -> 100M env steps with 3+
  seeds before any deployment claim.
- **Lookahead bias**: per project CLAUDE.md, soft-fill sortino is **not**
  trustable on its own. Validate **only** with the binary-fill marketsim
  at lag >= 2, slippage 0/5/10/20 bps, before deploying any neural model.
- **Production bar**: the live champion is the 32-model softmax_avg
  ensemble at **p10@5bps = 0.662**, 0/111 negative windows. The current
  fp4 leaderboard winner (`pufferlib_bf16`) sits at p10@5bps = -0.16,
  far below the bar at this horizon. **Do not deploy fp4 cells yet.**

## Decision on `alpacaprod.md`

No fp4 sweep cell exceeds the 32-model ensemble bar `p10@5bps = 0.662`, so
**`alpacaprod.md` is left unchanged** this phase.

## Next research directions

1. **Long-horizon rerun**: take the top-2 cells (`pufferlib_bf16`,
   `hf_trainer`) to 2M and then 100M env steps with 3 seeds each.
2. **Transformer encoder + NVFP4**: NVFP4 wins grow with hidden size; swap
   the MLP encoder for a 4-layer transformer block where NVFP4Linear
   replaces every QKV/MLP projection (BF16 stays on embed/head/LN).
3. **Constrained-MDP retest**: re-run `--constrained on` at 2M steps once
   the multipliers have time to converge; smoke runs are too short for
   Lagrangian dual ascent.
4. **Multi-instrument batching**: extend `gpu_trading_env` to step a
   universe in parallel inside the SoA kernel — should keep the 36M sps
   synthetic figure on real data.
5. **LOB env**: add a level-2 order book mode to `gpu_trading_env` so
   Layer B can quote against real depth instead of mid +/- offset.
