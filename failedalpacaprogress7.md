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

- **E4 lev2x_ds03 seeds 2-5 as 14th members** (2026-04-17, follow-through on E4 lineage):
  - s2: mean Δ −0.64% med, +0.25 neg, worst −1.01% @ full/100d/5bps → REJECT
  - s3: mean Δ **−0.23%** med, +0.25 neg, worst **−0.44%** @ full/100d/5bps → REJECT (softest in this group)
  - s4: standalone dud (train best_score=−53, neg=20) — skipped
  - s5: mean Δ −0.45% med, +0.25 neg, worst −0.87% @ full/30d/5bps → REJECT
  - All three val_best.pt checkpoints hit best_neg=0 during training (s2=1, s3/5=0) — strong individual risk
    profile that did NOT translate into ensemble additivity. The full lineage (s1-5) now stands 0/4 evaluated wins.
  - **Takeaway**: E4's lev2x_ds03 recipe produces checkpoints that are individually risk-controlled but
    consistently pull the 13m v5 median down by 0.2-0.6%/mo when added as a 14th member. The
    neg-window axis is nearly neutral (+0.25 ≈ a single 263w negative tip); the median axis is not.
    Leverage-boosted D does not get us to a 14th member. Stop expanding this branch.

- **AD_s9 as swap-in** (2026-04-17, `scripts/screened32_swap_in.py` via `docs/swap_in/ad_s9_swap.json`):
  - 0 / 13 wins. Every swap adds +2 to +23 neg windows and drops median by 0.0072 to 0.0222.
  - Worst swap: replace D_s16 → neg=34 (+23), med=+5.30% (−2.22%), sortino=4.42 (−2.13).
  - Best swap: replace D_s28 → neg=13 (+2), med=+5.77% (−1.75%) — still worse.
  - **Takeaway**: AD_s9's contribution is anti-correlated with EVERY current member's residual error.
    Both the add-test (14th-member) and the swap-test (replace member i) confirm the same thing:
    the 13m v5 ensemble is tighter than any single swap / addition with AD_s9 can produce. Drop from
    candidate pool.

- **EO-PPO seed=1 beta=0.03 warmup_frac=0.3** (2026-04-17, first EO-PPO pilot):
  - **Early-stopped at update 100** on `val_neg > 25` for 2 consecutive evals.
  - First val: med=-15.2%, neg=50/50, best_score=-54.0, best_neg=26 → catastrophic.
  - Training return never crossed zero (hovered -0.15 to -0.20 across all 100 updates).
  - **Diagnosis — design-doc risk #1 confirmed**: beta=0.03 with warmup_frac=0.3 ramps to
    ~0.022 by update 100, overwhelming the PPO policy gradient before the policy has found
    any profitable trading signal on its own. KL pressure in probability space is much
    stronger than early-policy PG.
  - **Takeaway**: beta range 0.01-0.05 from design doc was too aggressive. Try
    beta=0.005 + warmup_frac=0.7 (KL stays near-zero for first 70% of training). Seed=2
    launched 2026-04-17 09:15 UTC.
  - Checkpoint: `pufferlib_market/checkpoints/eo_ppo_pilot/b0.03_s1/` (kept for postmortem
    analysis of how the KL loss drove the policy away from profitability; this is useful
    data for tuning the next run).

- **EO-PPO seed=2 beta=0.005 warmup_frac=0.7** (2026-04-17, second EO-PPO pilot):
  - Training: peaked val med=+8.7%, neg=0/50 at update 50 (captured in `val_best.pt`),
    degraded by update 100 (val med=-2.6%, neg=34/50), early-stopped at update 200.
  - Standalone val_best looked promising (+8.7% med / 0 neg) but...
  - **14th-member realism gate REJECT**: adding val_best.pt to v7 (12-model) ensemble
    @ fb=5 1× gave med +7.19% (Δ −0.28%), p10 +2.65% (Δ −0.53%), neg 16/263 (Δ +6).
    `docs/realism_gate_v7_plus_eo_ppo_b005_s2/`
  - **Diagnosis — design-doc risk #2 revealed**: EO-PPO pushes specialist AWAY from
    ensemble MEAN distribution. But v7's residual errors are NOT distributed around the
    mean — they cluster on specific 2026-tariff-crash windows (start_idx 248-259).
    "Away in probability space" ≠ "helpful on failed windows". We're adding disagreement
    that's NOT aligned with the ensemble's actual weaknesses.
  - **Takeaway**: KL-based orthogonality is the wrong diversity axis for this ensemble.
    Options to try next:
      (a) Error-targeted training — hard-mine 248-259-like windows at train time.
      (b) Finetune-from-existing with small beta (script staged at
         `scripts/launch_eo_ppo_finetune.sh`), warm-starting AD_s9/D_s64 so PG doesn't
         collapse early.
      (c) Architecture diversity — different hidden size / activation rather than loss-based.
  - Checkpoints kept: `pufferlib_market/checkpoints/eo_ppo_pilot/b0.005_s2/{best,val_best}.pt`
    for post-mortem.

- **D_s97 swap-in sweep** (2026-04-17):
  - Script started but output `docs/swap_in/d_s97_swap.json` never materialized; must have
    errored silently. Process ran 2h58m accumulating 44 CPU-hours before exiting.
  - Earlier simpler 14th-member test at `/tmp/ensemble14_tests/D_s97.json` (April 14) already
    showed median_monthly −17.37% (wait, median_total_return 17.37% but 15 neg windows) —
    inconclusive but not a clear win vs v7's 10 neg baseline.
  - **Takeaway**: not pursuing D_s97 further unless we can reproduce the swap sweep cleanly.
