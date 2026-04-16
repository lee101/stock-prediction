# Push Toward 30%/month: findings, validated promising paths, and experiment plan

Date: 2026-04-16
Author: Claude code session
Scope: Audit of RL training stack, market simulator, ensemble members, and sim-to-real path in preparation for raising the monthly PnL gate from ~20% to ~30% median.

Ground truth: `scripts/eval_100d.py` at `decision_lag=2` binary fills, worst-slip cell, median monthly. Current stock champion: **13-model screened32 v5 ensemble**, 263-window OOS: median **+19.57%/mo**, p10 **+7.68%/mo**, **8/263 neg**, Sortino **34.07** (per `alpacaprod.md`, reconfirmed in `deployment.md`). Single-member performance is much worse — e.g. `C_s7_eval100d.md` shows -2.04%/mo at slip=0 for one member, so the ensemble is doing the work.

---

## 1. What I validated in this pass

- `tests/test_compiled_sim_loss.py` — **4/4 pass** (parity between soft sim and `torch.compile`'d path holds within 1e-6).
- `pufferlib_market/prod_ensemble_screened32/C_s7_eval100d.md` — individual members fail the 27% gate; ensemble aggregation is what clears it.
- `env_real.py` IS in `.gitignore` (not tracked). The plaintext PROD keys (`env_real.py:38-39`) are only a local-disk concern, not a GitHub leak. Still, rotate them before any public repo activity.
- `pufferlib_market/src/binding.c:134` — `decision_lag` silently defaults to 1 despite the header comment declaring "Production needs 2". `train.py:2075` sets `val_decision_lag=2` for the eval path, so production validation is correct today, but the binding default is a silent footgun.
- `pufferlib_market/src/trading_env.c:174` — observation at `t-lag`, execution at bar `t` open. Design is correct; fragile and not unit-tested at the C level.
- `trainingefficiency/compiled_sim_loss.py` is real: it's a `torch.compile(mode="reduce-overhead")` fused sim+sortino for the **binanceneural** (soft sim) training path. It does **not** make the pufferlib C env differentiable; it replaces the hand-rolled autograd.Function with a compiled equivalent. Still useful — 2–3× training-step speedup claimed — but does not solve the C-sim sample efficiency problem.
- Crypto12 v8 "2,658×/30d" and crypto70 results live entirely on bull-market data (2023–2025). Crypto70 fails lag=2 exhaustive eval (sweepresults/crypto70_s292_exhaustive.json, median return -7.2%). Assume crypto12 has the same overfit-to-regime risk until validated on an out-of-bull holdout.

## 2. Bugs and gaps found (by severity)

### HIGH — Silent `decision_lag=1` in C binding
- `pufferlib_market/src/binding.c:134`: `env->decision_lag = val ? (int)PyLong_AsLong(val) : 1;`
- Header comment `include/trading_env.h:128` says "Production needs 2" but default is 1.
- Mitigation today: `train.py:2075` explicitly passes `val_decision_lag=2` for validation. But any new script that doesn't pass `decision_lag=2` silently gets lookahead.
- Fix: default to 2 and require explicit opt-in (`decision_lag=1`) for legacy research; add a warning at init if `decision_lag < 2`.

### HIGH — No unit tests on the C simulator directly
- All correctness confidence is indirect: `test_compiled_sim_loss.py` checks parity against the soft sim, not against the C env.
- No test asserts:
  - zero-action policy earns exactly `-fee × turnover - margin_cost × time`
  - `fee` scales linearly with turnover
  - order at t fills at t+lag (lag enforcement)
  - `max_hold_hours=1` force-closes after exactly one bar
  - `fill_slippage_bps=20` rejects fills that slip outside the day range
- Fix: add `tests/test_trading_env_c_invariants.py` covering these five invariants. Each is ~20 LOC.

### MEDIUM — Death-spiral guard has no "short-after-buy" test
- `src/alpaca_singleton.py:328-396` blocks any sell >50 bps below last recorded buy.
- `tests/test_alpaca_singleton.py` covers sell cases but not the case of buying AAPL and then immediately *shorting* it. The guard's logic treats short == sell, so it should block, but there's no explicit test.
- Fix: add `test_death_spiral_blocks_short_after_buy` with buy at 200, short at 198 → RuntimeError.

### MEDIUM — `env_real.py` plaintext PROD keys
- `env_real.py:38-39` contains live Alpaca keys as hardcoded fallback defaults.
- File is in `.gitignore` so not pushed to GitHub, but any `git add -f env_real.py` would leak them.
- Memory says these keys return 401 — if so, they're dead anyway. Rotate + remove fallbacks in the same commit.

### LOW — `unified_orchestrator/service_config.json` drift
- Already flagged in `errors-fixed.md`; advertised an outdated 12-symbol universe vs live screened32. Fixed.

## 3. The realistic path from ~20% to 30% median/month

The gap is large. Expect to combine multiple independent sources of alpha/efficiency, not one big win.

### A. Pick up the already-implemented, not-default training knobs
All of these exist in `pufferlib_market/train.py` and are gated behind flags. Run a single screened32 seed sweep with each to measure median-monthly delta.

| Knob | Flag | Status | Expected delta | Effort |
|---|---|---|---|---|
| Muon optimizer | `--optimizer muon --muon-momentum 0.95` | available, not default | +0.5–2% | 1 training run per seed |
| Cosine LR + entropy anneal | `--lr-schedule cosine --anneal-ent --anneal-clip` | available, not default | +0.5–1% | seed sweep |
| Group-relative advantage | `--group-relative-mix 0.3` | implemented, disabled (mix=0.0) | +0.5–1% | seed sweep |
| Per-symbol obs norm | `--obs-norm --per-sym-norm` | risky (memory says F/G breaks val) | unknown, test small | A/B 4 seeds |

**Experiment E1**: run 4 screened32 seeds under each combination, promote only if 263-window median monthly ≥ current champion at worst-slip cell. No seed-cherry-picking: always use the same 4 seeds.

### B. Compound ensemble member discovery (highest-leverage known path)
The memory shows adding `I_s32` to swap out `D_s27` moved neg from 10 → 8 and median +0.55%. This is the bread-and-butter path and it already runs as `scripts/sweep_screened32.sh`, `sweep_screened32_cross.sh`, `sweep_screened32_weekly.sh`.

**Experiment E2**: continue sweeps but change acceptance rule — a candidate must *cause* p10 to rise, not just avoid lowering it. The current 13-model ensemble p10 is 7.68%. Require 14-model p10 ≥ 8.0%. Use `scripts/evaluate_screened32_candidates.py` (new — already added).

### C. Differentiable sim for sample-efficient warmup
`compiled_sim_loss.py` works for binanceneural. Two concrete uses:

1. **Short differentiable warmup, then PPO fine-tune**: train the policy for 1–3M steps in the soft compiled sim (2–3× faster steps), then swap to PPO on the C env. Measure whether warmup carries to OOS. If yes, expect 30–50% reduction in total wall-clock per seed, which lets us run more seeds and widen the ensemble search.

2. **Fit-for-purpose hourly experiment**: the soft sim uses decision_lag-aware indexing correctly (`compiled_sim_loss.py:291-299`). Use it to train a hourly-stock variant on synthetic resampled data before committing to the full C-env port.

**Experiment E3**: train one screened32 seed under soft sim → C-env fine-tune and compare 263-window median to a pure C-env seed with the same total compute.

### D. Leverage path (2×)
Not safe to flip today. Concrete pre-work:

1. Train a new ensemble with `max_leverage=2.0` in sim — current sim supports it (`trading_env.c:130`). Do *not* run 2× policies inference-only against a 1×-trained checkpoint; the policy learned the wrong sizing.
2. Add hard Reg-T 2:1 cap in `trade_daily_stock_prod.py:2360` — currently `BUYING_POWER_USAGE_CAP=0.95` is scalar. Change to `min(allocation_pct/100 * equity, 2.0 * equity, buying_power * 0.95)`.
3. Validate Alpaca account type is Reg-T margin (not cash) in `alpaca_deploy_preflight.py`.
4. Paper test for ≥ 5 sessions before any live.

**Experiment E4**: retrain 3 screened32 seeds at `max_leverage=2.0`, compare 263-window median to 1× baseline. If >2× the single-side return ×0.9 (to pay for margin), promote.

### E. Crypto and hourly — valid but not the shortest path
- Crypto12/crypto70 are bull-regime-overfit. Before trusting any crypto number, build a val set that includes Q2 2022 (crypto winter) and Q1 2018 (ICO crash) and re-eval top checkpoints. If they still pass, crypto is viable; if not, the record is noise.
- Hourly-stock RL has no training data in-tree (stocks are daily-close only per the subagent pass). Building a dataset is 3–7 days of pipeline work plus retraining from scratch. Park until A–D are exhausted.

## 4. Rough ranking of expected monthly-return upside

Totals compound, they do not add. This is a best-guess breakdown, not a guarantee:

| Source | Lift estimate | Confidence |
|---|---|---|
| Ensemble expansion (E2) | +1–3% | HIGH (proven path) |
| Training-efficiency knobs (E1) | +1–3% | MEDIUM |
| 2× leverage done safely (E4) | +5–10% (on current median) | MEDIUM (policy must be retrained) |
| Differentiable sim warmup (E3) | 0% return; −30–50% wall-clock | MEDIUM |
| Crypto / hourly | unknown | LOW until val set includes bear |

Hitting 30%/mo median probably requires 2× leverage plus one other win. Everything else is a multiplier on the baseline.

## 5. Immediate next actions (ranked)

1. **Fix the decision_lag default**: change `binding.c:134` default to 2, add warning if caller passes <2. Keep `include/trading_env.h:128` comment accurate.
2. **Add five C-env invariant tests** (see section 2 HIGH #2). These protect the ground truth we rely on.
3. **Run E1** (training-knob A/B) — 4 screened32 seeds × 4 knob combos, compare 263-window median at lag=2.
4. **Run E4 paper leg** in parallel — retrain 3 seeds at `max_leverage=2.0`, run paper deploy for 5 sessions, measure live-vs-sim slippage.
5. **Rotate PROD Alpaca keys** (separate from model work; prerequisite for any live deploy).

Anti-goal: do *not* chase crypto12 v8's 2,658× number. It's almost certainly a bull-regime artifact and a distraction from the stock ensemble that actually clears the gate.

---

## 6. Iteration log (2026-04-16 session)

**Sim-realism fixes shipped (both with tests):**
- `intrabar_replay._stop_loss_triggered` now fills at worse-of-stop-vs-bar_open on gap-through events — previously always filled at the stop level. Matches real exchange stop-market behaviour on gap-down (long) / gap-up (short) bars. Commit `cd00dc37`.
- `replay_intrabar` now calls `_apply_short_borrow_cost` each hour a short is open (parity with `simulate_daily_policy_intrabar`). Commit `d8a38e69`.
- `pufferlib_market/environment.py` TradingEnvConfig plumbs `decision_lag=2` default to the C binding so new scripts can't silently default to lag=1. (Fixing the C default itself is blocked on rebuilding + retesting all existing configs.)
- Added `tests/test_trading_env_c_invariants.py` — 6 unit tests on the C env directly (zero-action, decision_lag shift, max_hold, fill_slippage, fee linearity, lag=2 obs). All pass.

**Candidate evaluations vs 13-model v5 baseline (50-window, 30d+100d, 5bps):**
| Candidate | 14m med (100d/full) | 14m p10 (100d/full) | 14m neg (30d/full) | Decision |
|---|---|---|---|---|
| Baseline (13m) | +45.34% | +24.48% | 9/50 | reference |
| + D_s67 | +32.53% | +19.41% | 11/50 | REJECT (−13pp median, +2 neg) |
| + AA val_best (cosine+anneal, seed=1) | pending | pending | pending | (identical to AC — seed-1 parity before divergence) |
| + AB val_best (group-rel 0.3, seed=1) | running | running | running | pending |
| + AC val_best (full E1 stack, seed=1) | identical to AA | identical to AA | identical to AA | (AA≡AC md5-equal; see `feedback_E1_sweep_results.md`) |

**E1 standalone OOS results (263-window val_best_oos_eval.json):**
- AA: neg=77, med=−9.4%, p10=−95.6% (overfit val)
- AB: neg=53, med=−0.7%, p10=−95.4% (softest overfit; worth ensemble test)
- AC: identical to AA (byte-equal checkpoint; seed=1 parity)

**Takeaway:** neither D_s67 nor AA/AC improves the 13-model ensemble. AB ensemble test pending. If AB also fails, the 13-model v5 ensemble appears locally optimal on the current data — next step is E2 (new seeds) or E4 (2× leverage retrain) per section 4.
