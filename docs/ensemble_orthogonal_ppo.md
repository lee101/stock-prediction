# Ensemble-Orthogonal PPO (EO-PPO)

*Design doc, 2026-04-17. Targets the diagnosed blind-spot in the 13-model
v6 screened32 ensemble.*

## TL;DR

Train a **14th specialist** whose gradient is explicitly pushed to
**disagree with the v6 ensemble at timesteps where v6 is uncertain or
recently losing**. We add one auxiliary loss term to PPO:

```
L_total = L_ppo  -  beta * w(s) * KL( pi_specialist(·|s) || pi_v6(·|s) )
```

Note the **minus** sign: we *maximise* KL divergence from the ensemble.
The `w(s)` gate keeps divergence pressure only on states where the
ensemble is already near-indifferent or has been underperforming, so
the specialist still learns the profitable argmax on regime where v6 is
correct.

## Problem diagnosis (from `docs/diversity_screen/best_diversity.json`)

The 13-model v6 baseline loses **11/263 windows** on the full screened32
val set. All 11 negatives cluster at start_idx **248-259** (one
contiguous run of 10 windows plus one at 259). This is a single-regime
failure — almost certainly the 2026 tariff-crash band already flagged
in `feedback_val_zero_neg_unreliable.md`.

Every attempted 14th member so far has been one of two modes:

1. **Correlated-strong**: AD_s9, AD_s1, D_s67 — they beat baseline on
   normal windows but *share* v6's losses on the crash cluster (high
   Pearson corr with baseline returns, overlapping neg-window set). They
   dilute v6's decisive consensus on normal windows without fixing the
   crash, so median drops.
2. **Anti-correlated-weak**: AD_s4 at corr=0.29 has good diversity but
   is a standalone dud (51/263 neg). Joining the ensemble, its 1/14
   weight is not enough to flip votes, and its noise on normal windows
   costs more than its signal on crash windows.

We have exhausted the **blind 14th-member search**. The gap is not
"find a stronger seed"; it is **"force a specialist that disagrees on
the crash regime"**. EO-PPO does this at training time.

## Related work (web-searched 2026-04-17)

- **MED-RL** (Sheikh et al., ICLR 2022): uses economic inequality
  metrics (Gini, Atkinson, Theil) across ensemble-member activations as
  a diversity regulariser. Up to 300% gains on Mujoco/Atari. Key
  insight: *encourage* inequality across members.
- **Dynamic NCL** (JSAI 2025): dynamically anneals the diversity vs
  accuracy tradeoff during training. Matches our need for a schedule:
  beta starts small, grows once policy is competent.
- **Negatively Correlated Ensemble RL** for level generation (ICLR 2024
  workshop): multiple sub-policies + a stochastic selector. Parallel
  to the 14-member argmax-average setup, but they use explicit negative
  correlation in the shared loss.
- **Ensemble Policy Gradient with KL constraint** (Shitanda 2026
  preprint): theoretical result that *excessive* inter-policy diversity
  harms stability. Hence we gate beta by the ensemble-advantage signal.
- **BootDQN-EVOI** (IJCNN 2025): expected-value-of-information across
  ensemble heads. Relevant for a *future* bandit-style member selector,
  not for this training loop.

EO-PPO is a minimalist fusion: MED-RL's "push members apart" in
*policy-distribution* space, gated by the regime-uncertainty signal
borrowed from Dynamic NCL.

## Algorithm

### Inputs at training time

1. Standard PPO rollouts from the screened32 daily env (same as D/AD
   variants).
2. A frozen cached v6 ensemble `pi_v6(·|s)`: 13 MLPs in eval mode,
   ~1.3M total params on the same GPU as the specialist. One forward
   pass per minibatch, no backward.

### Per-minibatch computation

```python
with torch.no_grad():
    logits_v6 = mean(pi_v6_i(obs) for i in range(13))   # [B, A]
    probs_v6  = softmax(logits_v6)
    # Gate: entropy of v6 on this state.  High entropy = v6 is unsure,
    # more pressure on specialist to diverge.
    H_v6 = -(probs_v6 * log_softmax(logits_v6)).sum(-1)  # [B]
    w    = (H_v6 / H_v6.max()).clamp(min=W_MIN)          # [B]

logits_s = policy(obs)               # specialist, with grad
log_probs_s = log_softmax(logits_s)
probs_s     = softmax(logits_s)

# KL(specialist || v6) per state
kl = (probs_s * (log_probs_s - log_softmax(logits_v6))).sum(-1)  # [B]

L_orth = -(w * kl).mean()            # negative => maximises KL
L      = L_ppo + beta * L_orth
```

### Hyperparameters

- `beta`: 0.01 → 0.05 (linear ramp over first 30% of training, then
  constant). Rationale: avoid corrupting early policy learning with
  orthogonality pressure before the specialist has found *any* good
  action on its own.
- `W_MIN`: 0.1. Prevents zeroing out the bonus on purely-confident v6
  states where we might still want some diversity.
- `H_v6` normalised by its running per-batch max, not the theoretical
  `log(A)`, so the gate is scale-free.

### Safety cushions

- Clamp KL to `[0, 5]` per sample to avoid a single outlier state
  dominating the gradient.
- Share weights/optimizer state with the *same* D/AD training recipe
  (Muon, tp=0.05 target-pos head, horizon=252). We only add one loss
  term; everything else matches a known-good single-member config.
- Deterministic eval at deploy gate: `scripts/eval_multihorizon_candidate.py`
  with `--baseline-extra-checkpoints` = the full 12 other v6 members
  (AD_s4 already in v6 — do NOT double-add, see
  `feedback_prod_ensemble_already_has_ADs4.md`).

### Why this beats raw NCL on activations

Activation-space NCL (classic Brown/Liu 1996) penalises covariance of
hidden units. On our 13 *frozen* members it would need 13×13 inner
products every minibatch *and* cannot target a specific deployment
geometry (argmax over averaged probs). Probability-space KL is
(a) cheaper — one softmax comparison, (b) directly aligned with how
the ensemble votes at inference, and (c) gate-able by the v6 entropy
signal, which is exactly the "where is v6 unsure" readout we want.

## Gate criteria

A trained EO-PPO specialist enters the 14-model v6 candidate pool iff,
via `scripts/eval_multihorizon_candidate.py` on the worst-slip cell
(fb=5, slip=5, lag=2, lev=1.0, shorts off, 263 OOS windows):

1. `neg_windows <= 11` (do not regress on risk).
2. `median monthly >= baseline_median` (do not hurt median on normal
   regime).
3. p10 monthly `>= baseline_p10 - 0.003` (small risk-axis tolerance if
   (1)+(2) hold strongly).

AND the specialist *must* show **at least 1 of the 11 baseline-negative
windows flipped to positive** in the candidate-alone run. If it can't
fix any of v6's blind-spot windows on its own, the KL maximisation
failed — it's just noise orthogonal to the decision boundary we care
about.

## Why this is the right next step

- Our 14th-member search has hit diminishing returns (8 REJECTs this
  session, 0 promotes). Continuing to scan seeds is stuck at a local
  optimum.
- LOO analysis shows v6 is internally tight: no free-drop slot for a
  swap (the closest, D_s81, has Δmed ≈ 0). There is no "obvious weak
  member to replace" — the *composition* is already Pareto-optimal
  under the current training recipe.
- The only degree of freedom left is the **training recipe itself**.
  EO-PPO is the smallest principled change to that recipe that
  directly targets the diagnosed failure mode.

## Implementation plan

1. `pufferlib_market/train.py`: add three flags —
   `--ensemble-kl-beta`, `--ensemble-kl-checkpoints` (comma list),
   `--ensemble-kl-gate` (`entropy` | `off`). Default off (no behaviour
   change).
2. At init, load the 13-member v6 into eval-mode on the same device,
   share obs normalisation with the specialist.
3. In `_ppo_loss_fn`, compute the orthogonality term only if `beta > 0`.
4. Test stubs: `tests/test_ensemble_orthogonal_ppo.py` verifying
   (a) term is zero when beta=0, (b) gradient flows only through
   specialist logits (not v6), (c) per-sample KL clamped to [0, 5],
   (d) beta ramp applies correctly.
5. Pilot run: D variant, seed=1, `--ensemble-kl-beta 0.03`, 60M steps
   on screened32 single-offset. Compare to D_s1 baseline at the 14th-
   member gate.

## Risks

- **Over-regularisation**: beta too large → specialist diverges from
  profitable argmax everywhere, becomes a standalone dud worse than
  AD_s4 (already weak at 51 neg). Mitigation: tight beta range + gate.
- **GPU memory**: 13 v6 models × ~100K params = 1.3M. At fp16 this is
  ~2.6MB; at fp32 ~5.2MB. Trivial. The forward pass adds one 13-way
  matmul; should fit well under the current memory budget.
- **Correlation with stochastic eval**: if val window returns change
  under different random seeds for binary-fill simulation, the gate
  might accept lucky runs. Mitigation: same determinism guarantees as
  all other 14th-member evals (they run with fixed seeds).
- **Identifiability of the crash regime**: the gate uses entropy of
  v6, not the crash-regime label directly. It's plausible that v6 is
  *confidently wrong* on crash windows (low entropy) and the gate
  under-weights them. If so, pivot to a **volatility-feature gate**
  instead of entropy.

## Success criteria

- EO-PPO pilot passes the 14th-member gate on first or second seed.
- Beats every 14th-member seed in the current REJECT list
  (mean Δmed >= 0, Δneg <= 0).
- Pattern replicates across ≥2 seeds (not a single lucky run).
