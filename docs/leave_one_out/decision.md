---
date: 2026-04-17
tool: scripts/screened32_leave_one_out.py
val: pufferlib_market/data/screened32_single_offset_val_full.bin (263 windows)
gate: fb=5, lev=1.0, lag=2, fee=10bps, slip=5bps, deterministic argmax
baseline: 13-model v5 — med +6.89%/mo, p10 +2.34%, sortino 6.10, neg 11/263
---

# Leave-one-out analysis of 13-model v5 ensemble

For each of the 13 members, build (N-1)-model ensemble (softmax_avg) and
re-evaluate at the deploy-gate cell. Δ = (without member) − baseline.
Positive Δmed_monthly = the member is hurting the ensemble.

## Results (sorted by Δmed_monthly, most droppable first)

| idx | member  | Δmed_mo  | Δp10_mo  | Δneg | Δsortino | verdict       |
|---:|:--------|---------:|---------:|----:|---------:|:--------------|
|  3 | D_s3    | **+0.137%** | **+0.236%** |   0 | **+0.165** | **WEAK LINK** |
|  8 | D_s81   |  +0.044% |  −0.072% |  +3 |  −0.07   | borderline    |
| 12 | I_s32   |  −0.137% |  +0.071% |  +3 |  −0.25   | load-bearing  |
| 11 | D_s64   |  −0.355% |  −0.246% |   0 |  −0.59   | load-bearing  |
|  9 | D_s57   |  −0.526% |  +0.181% |  +4 |  −0.83   | load-bearing  |
|  4 | I_s3    |  −0.652% |  −0.920% |  +6 |  −0.87   | load-bearing  |
| 10 | I_s3 (2x) | −0.652% |  −0.920% |  +6 |  −0.87  | load-bearing  |
|  5 | D_s2    |  −0.731% |  −0.382% |  +7 |   0.03   | load-bearing  |
|  2 | D_s42   |  −1.023% |  −2.001% | +13 |  −0.93   | load-bearing  |
|  7 | D_s28   |  −1.260% |  −1.099% |  +7 |  −1.40   | load-bearing  |
|  1 | D_s16   |  −1.264% |  −3.572% | +23 |  −1.50   | load-bearing  |
|  6 | D_s14   |  −1.327% |  −1.629% |  +8 |  −1.93   | load-bearing  |
|  0 | C_s7    |  −2.210% |  −1.406% |  +6 |  −1.48   | most critical |

## Key finding

**D_s3 is a free-drop candidate.** Removing it gives:
- median_monthly  +6.89% → +7.03% (+0.14%)
- p10_monthly     +2.34% → +2.58% (+0.24%)
- median_sortino    6.10 → 6.27   (+0.17)
- median_max_dd    6.28% → 5.60% (−0.68%, BETTER)
- n_neg              11  →   11   (no change)

**Every metric improves on the 12-model variant** (drop D_s3). This is the
first member found to be a net negative since the v5 ensemble was
assembled. Most likely cause: D_s3 was added in earlier ensemble version
(pre-v3) when fewer members existed; since then I_s3 (added v3) and
I_s32 (added v5) cover its directional signal better, and its
errors now correlate with the ensemble majority on the same windows
where the majority is wrong.

## Implications

1. **Slot opens for AD_s4** — the only 14m candidate from the A* batch
   that scored Δmed > 0 (+0.09%). Tested as 13m swap (drop D_s3, add
   AD_s4) at deploy gate to see if it strictly beats both baseline and
   12-model.

2. **D_s81 is also borderline** — Δmed +0.044% but adds 3 neg windows.
   Net negative on the joint metric. Worth watching but not droppable
   alone (the +3 negs is the bigger risk).

3. **C_s7 is critical** — dropping it costs −2.21%/mo (worst by 2x).
   Confirms the conservative AdamW anchor pattern from prior research
   (`feedback_stocks12_ensemble_expansion.md`). Don't touch C_s7.

4. **D_s16 has the largest neg-spread** — drop costs +23 neg windows
   despite only −1.26% median. It's hedging tail risk. Keep.

5. **Both copies of I_s3 (2× weight) score identically** — confirms the
   weighting is correct (each copy has full influence in the average).

## Action

If 12-model deploy gate confirms LOO findings (med ≥ +7.03%, neg ≤ 11,
1.5× knee preserved), update `src/daily_stock_defaults.py`:
- Remove D_s3 from `DEFAULT_EXTRA_CHECKPOINTS`
- 12-model becomes new baseline (or 13m with AD_s4 if that strictly wins)
