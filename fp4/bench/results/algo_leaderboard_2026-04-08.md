# Algorithm Leaderboard — 2026-04-08

- Source sweep summary: `fp4/bench/results/sweep_20260407_235756/summary.csv`
- Generated: 2026-04-08
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
| 1 | ppo | off | 3 | -4.0110 ± 0.1769 | -1.2644 ± 0.0453 | -0.4616 ± 0.0225 | n/a | n/a | n/a |

## Risk-adjusted ranking
Feasible set: cells with `|max_dd| ≤ 0.20` and finite p10.

_No cell satisfies the drawdown budget._

## Recommendation
**Recommendation: ppo + unconstrained + NVFP4 (graph-safe linear, commit d8eb1a5b).** This cell posted p10@5bps = -4.0110 ± 0.1769 with sortino@5bps = -0.4616 ± 0.0225 and max_dd = n/a over n=3 seeds, the best feasible risk-adjusted cell under the |max_dd| ≤ 0.20 budget.


_No cell beat the 32-model ensemble bar (p10@5bps = 0.662); `alpacaprod.md` left unchanged._
