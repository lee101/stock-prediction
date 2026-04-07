# Algorithm Leaderboard — 2026-04-07

- Source sweep summary: `fp4/bench/results/sweep_demo/summary.csv`
- Generated: 2026-04-07
- Production bar: 32-model ensemble **p10@5bps = 0.662**, max_dd budget = 0.20

## Upstream commits
- `f9806fe0` bench_gemm uses real CUTLASS NVFP4 GEMM (2.24x BF16 @ 4k^3)
- `d8eb1a5b` graph-safe NVFP4Linear (+62% SPS over Unit A baseline)
- `3a1a29c2` market_sim_py pybind11 bindings
- `57a0879b` generic eval + first PnL/Sortino comparison run

## Ranked leaderboard
Sorted by (p10@5bps DESC, sortino@5bps DESC, |max_dd| ASC).

| rank | algo | constrained | n_seeds | p10@5bps | median@5bps | sortino@5bps | max_dd | sps | gpu_mb |
|------|------|-------------|---------|----------|-------------|--------------|--------|-----|--------|
| 1 | pufferlib_bf16 | off | 2 | -0.1600 ± 0.0179 | -0.0635 ± 0.0003 | -0.5825 ± 0.0399 | 0.1208 ± 0.0121 | 4693 | 0 |
| 2 | hf_trainer | off | 2 | -0.2100 ± 0.1382 | -0.0451 ± 0.0556 | -0.1715 ± 0.2911 | 0.1753 ± 0.0047 | 6773 | 33 |

## Risk-adjusted ranking
Feasible set: cells with `|max_dd| ≤ 0.20` and finite p10.

| rank | algo | constrained | p10@5bps | sortino@5bps | max_dd |
|------|------|-------------|----------|--------------|--------|
| 1 | pufferlib_bf16 | off | -0.1600 ± 0.0179 | -0.5825 ± 0.0399 | 0.1208 ± 0.0121 |
| 2 | hf_trainer | off | -0.2100 ± 0.1382 | -0.1715 ± 0.2911 | 0.1753 ± 0.0047 |

## Recommendation
**Recommendation: pufferlib_bf16 + unconstrained + NVFP4 (graph-safe linear, commit d8eb1a5b).** This cell posted p10@5bps = -0.1600 ± 0.0179 with sortino@5bps = -0.5825 ± 0.0399 and max_dd = 0.1208 ± 0.0121 over n=2 seeds, the best feasible risk-adjusted cell under the |max_dd| ≤ 0.20 budget.


_No cell beat the 32-model ensemble bar (p10@5bps = 0.662); `alpacaprod.md` left unchanged._
