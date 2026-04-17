# Failed Alpaca Progress 7 — Frontier Experiments

Dead-end experiments from the 2026-04-17 frontier-efficiency push. Each
block records: what we tried, why it looked plausible, what happened, and
the 1-line "don't repeat this" takeaway so future sessions don't burn
the same GPU hours.

## How to read

- **Hypothesis**: one sentence.
- **Config**: exact CLI or commit hash.
- **Result**: metric row vs 13-model v5 baseline (med=+19.57%, p10=+7.68%, neg=8/263 on 263w OOS).
- **Postmortem**: why this didn't work / what it tells us about the next move.

---

## Prior dead-ends (summarized from MEMORY.md + push doc)

- **D sweep seeds 73-120+** (96 seeds, 5.2% pass rate on neg≤17, none beat ensemble)
- **E/F variants** (0/8 success rate)
- **Cross-features CF s1** (holdout med=-4.28%, 31/50 neg despite good in-training val)
- **per-sym-norm (F/G variants)** — holdout diverges 20+ neg/50 even with perfect in-training val
- **BF16 + CUDA graph** — degenerate (see `feedback_stocks12_bf16_cudagraph.md`)
- **newcfg seeds** — hurt ensemble (see `feedback_stocks12_seedsweep_preset.md`)
- **s42 v5_rsi standalone** — overfit to Jul-Nov 2025 val slice; on val_full 0/50 positive
- **D_s67 as 14th member** (2026-04-16): mean delta −2.19%, +2 neg → REJECT
- **AB s1 as 14th member** (2026-04-16): mean delta −0.81%, +5 neg → REJECT
- **AA s2 as 14th member** (2026-04-16): mean delta −0.74%, +1 neg → REJECT
- **AB s2 as 14th member** (2026-04-16): mean delta −0.68%, +2 neg → REJECT
- **AA s1 ≡ AC s1, AA s2 ≡ AC s2** (both md5-identical — seed converged to best-val
  before `--group-relative-mix 0.3` could diverge the trajectory from cosine+anneal)
- **AD s9 as 14th member** (2026-04-17): mean delta −1.07% med, −2.28% p10, +2.25 neg → REJECT
  - Standalone best-in-class (263w med=+14.10%, neg=11), yet adding it to v5 hurts all 4 cells.
  - Matches `feedback_ensemble_diversity_over_strength.md`: raw standalone strength ≠ ensemble additivity.
- **C lev1p5x s1 as 14th member** (2026-04-17): mean delta −4.04% med, −5.56% p10, +8.00 neg → REJECT
  - Leverage-boost variant from 2026-04-12 sweep; worst single eval in this batch.
  - Takeaway: 1.5× leverage on C (C_s7 anchor lineage) destabilises ensemble diversity wholesale.

---

## 2026-04-17 frontier-session dead-ends (this push)

- **D_s67, AB s1, AA s2, AB s2, AD s1, AD s9, C_lev1p5x_s1, E4 lev2x_ds03 s1** — eight distinct 14th-member
  ensemble-add evals, 0 wins, mean delta median monthly return ∈ [−4.04%, −0.56%].
  13-model v5 ensemble confirmed locally optimal under the 14th-member one-at-a-time test.
- **E4 lev2x_ds03 s1 as 14th member** (2026-04-17): mean delta −0.56% med, −0.33% p10, **+0.00 neg** → REJECT
  - Softest reject on the neg axis (zero extra losing windows!) but median drops 0.56%.
  - Standalone val was weak (med=-1.4%, neg=21/30, best_neg=14) yet ensemble additivity
    was nearly break-even on risk. Leverage-boosted D may have real complementary signal
    but needs more seeds to find one that wins the median too.
  - **Next move**: run the leverage sweep to seeds 2-5 instead of killing E4 outright.
