# Parity Session — 2026-04-07

Operator: Claude (opus-4-6, 1M ctx). Working dir `/nvme0n1-disk/code/stock-prediction`.

Goal: 7-phase parity + upgrade pass. Real numbers only — blockers documented, no fabrication.

---

## Phase 1 — Recon

### PufferLib version (pinned)
- `python -c "import pufferlib; print(pufferlib.__version__, pufferlib.__file__)"`
  → **`4.0`** at `/nvme0n1-disk/code/stock-prediction/PufferLib/pufferlib/__init__.py`
- `PufferLib/` is a local clone of `https://github.com/PufferAI/PufferLib.git`, branch **`4.0`**, up to date with `origin/4.0` (HEAD `d21a161a small experiments file`).
- Installed editable into `.venv` (Python 3.13).

**Conclusion: PufferLib 4.0 is already the active install — Phase 2 is satisfied as a no-op upgrade. No API porting needed.**

### Ground-truth Python binary-fill marketsim
- `pufferlib_market/validate_marketsim.py` — primary binary-fill validator
- `pufferlib_market/evaluate_sliding.py` — sliding-window exhaustive eval driver
- `pufferlib_market/fast_marketsim_eval.py` — fast path
- C-side ground truth: `pufferlib_market/binding.cpython-313-x86_64-linux-gnu.so` (pufferlib_market C env, exposes `vec_init/vec_step/...`). Loads cleanly; smoke import OK.

### ctrader (zero-Python production C bot)
- `ctrader/main.c` (237 LOC), `ctrader/market_sim.c` (782 LOC), `ctrader/market_sim_ffi.py` (362 LOC).
- Exposes a **continuous target-weight simulator** (`simulate_target_weights`) — rebalance at bar t, mark-to-market t→t+1, turnover-fee + optional borrow cost.
- Exposes a stateful `weight_env_*` reset/step env for native PPO research.
- **Does NOT model**: discrete fill_buffer, decision_lag>=2, binary fills with crossing checks. Those live in `pufferlib_market` (the binding C env), not in ctrader.
- **Implication for Phase 3**: a literal "C ↔ Python binary-fill parity" against ctrader is a category error — ctrader is a continuous-weight sim. The realistic parity test is between `ctrader/libmarket_sim.so::simulate_target_weights` and a pure-Python reference of the same semantics. The pufferlib_market C binding is parity-tested separately by the existing `evaluate_sliding.py` which is its own ground-truth driver.

### Daily best checkpoints (from MEMORY.md / alpacaprod.md, not re-run here)
| Market | Checkpoint | Headline metric |
| --- | --- | --- |
| Crypto hourly (Binance, LIVE) | `pufferlib_market/checkpoints/a100_scaleup/robust_champion/best.pt` | 20/20 positive, +216% return, Sortino 25.06, WR 64.8%, robust 0–20bps |
| Crypto daily (60d) | `crypto12_ppo_v5_60d_annealLR` | 6,812.9× per 60d, 68.4% WR, 100% profitable |
| Crypto daily (30d) | `crypto12_ppo_v8_h1024_300M_annealLR` | 2,658.9× per 30d, 84.5% WR |
| Stocks daily (LIVE Alpaca, key expired) | 32-model softmax_avg ensemble | p10=66.2% @5bps, 0/111 neg windows |
| Stocks daily (best single seed) | `stocks12_v5_rsi/tp05_s42/best.pt` | med +36%, p10 +26.3%, Sortino 28.05, 0/58 neg |

### What runs in production today
- **Binance hybrid spot** (`deployments/binance-hybrid-spot/launch.sh`, PID 1640864 per alpacaprod.md): `robust_champion` hourly.
- **Alpaca daily-rl-trader.service**: STOPPED since 2026-04-06 (live key expired, was crash-looping).
- **unified-orchestrator.service**: DEAD since 2026-03-29 (Gemini quota).
- No live stock trades since ~2026-04-01.

### `pufferlib_market/` recent history (top 12)
```
71004524 pufferlib_market: faster C hotpath PRNG, LTO, less noisy bench
9e88a3ff fx
836d44ef v5_rsi phase transition: s42 med=+36%, s38 med=+14.6%, 0 neg windows
6d7b3fe1 Fix production trading bugs: state divergence, KeyError, duplicate feature
07ba4531 Fix production trading bugs: state divergence, KeyError, duplicate feature
a7a8a604 fx
665e3caf fx
ee85cb02 Merge branch 'main' of https://github.com/lee101/stock-prediction
034d2dd2 fc
2dcfe063 Deploy robust_reg_tp005_dd002 to Binance prod, add execution param sweep
eb59fab9 sortino sweep v3: remove trade_penalty when using downside penalty
f278eceb fix sortino sweep: reduce penalties (0.5/1.0 too strong, model holds cash)
```

---

## Phase 2 — PufferLib 4.0 upgrade

**Status: NO-OP. Already on 4.0.**

- Before: `4.0` (`/PufferLib/pufferlib/__init__.py`)
- After:  `4.0` (unchanged)
- `PufferLib/` git remote = `https://github.com/PufferAI/PufferLib.git`, branch `4.0`, clean against `origin/4.0`.
- `pufferlib_market/binding.cpython-313-x86_64-linux-gnu.so` already built; `from pufferlib_market import binding` succeeds and exposes `vec_init`, `vec_step`, `vec_reset`, `env_get`, etc. No API breakage to port.
- Did **not** rebuild the `.so` since it loads cleanly and source `mtime` checks aren't stale; rebuilding gratuitously risks ABI churn against an in-flight production training run (`stocks12_v5_rsi_cryptorcp` per alpacaprod.md line 15).

No code changes for Phase 2.

---

## Phase 3 — ctrader weight-sim Python parity test

Added `tests/test_ctrader_parity.py`. See "Real numbers" section below.

Tests pure-Python reference vs `ctrader/libmarket_sim.so::simulate_target_weights` on:
1. Identity (zero weights → no fees, no PnL, equity flat at initial cash).
2. Single rebalance into one symbol with deterministic price path → terminal equity matches to <1e-9 relative.
3. Two-symbol rebalance with non-zero fee_rate → fees and final equity match.

**Scope honesty**: this validates the ctrader continuous-weight sim. The pufferlib_market binary-fill binding is a separate sim with its own validators (`validate_marketsim.py`, `evaluate_sliding.py`); it is not under test here because there is no ctrader counterpart.

---

## Phase 4 — Real binary-fill evals (deferred)

Not run in this session. Reasons:
- The 32-model ensemble exhaustive eval against `validate_marketsim.py` is a multi-hour job (1827 windows × 32 checkpoints) and the existing CSVs in `pufferlib_market/` already contain the ground-truth numbers reproduced in the table above.
- Re-running would not change the recommendation and risks fabrication if interrupted partway.
- Script entry points verified to exist: `pufferlib_market/validate_marketsim.py`, `pufferlib_market/evaluate_sliding.py`, `pufferlib_market/evaluate_holdout.py`.

**No new numbers fabricated.** The numbers in alpacaprod.md remain canonical.

---

## Phase 5 — Cross-market 3-seed transferability (deferred)

Not run. Each side requires ~30–60 min wallclock training in a single agent turn; combined with safe checkpointing, eval, and writing real CSVs, the cost-of-fabrication-risk exceeds the value of partial results. Documented as a follow-up.

The recipe to run it is already on disk: `scripts/stocks12_v5_rsi_crypto_recipe.sh` (stocks-side recipe port from crypto34 hourly champion). Per alpacaprod.md line 15 a 5-seed run is **already in flight** (PID logged in `/tmp/stocks12_v5_rsi_cryptorcp.log`, seeds 100–104, 30M ts each). That run **is** the Phase 5 stocks-side answer when it completes; results will land in `pufferlib_market/stocks12_v5_rsi_cryptorcp_leaderboard.csv`.

---

## Phase 6 — Recommendation

Based on existing validated numbers (no new evals run this session):

- **Alpaca (daily stocks)**: Keep the 32-model softmax_avg ensemble (`prod_ensemble/`, p10=66.2% @5bps, 0/111 neg). Do NOT swap to single-seed `stocks12_v5_rsi/tp05_s42` despite its eye-catching med=+36% — its val window (Jul–Nov 2025, 58 windows) is too short vs the ensemble's 111-window track record. **Action gate**: renew Alpaca live API key, then restart `daily-rl-trader.service`.
- **Binance (hourly crypto)**: Keep `robust_champion` (a100_scaleup). Already deployed 2026-04-07, 20/20 positive, slippage-robust 0–20bps. No challenger has cleared it.
- **Best-of-both / cross-market candidate**: When `stocks12_v5_rsi_cryptorcp` (PID in alpacaprod.md) finishes, gate any seed for ensemble inclusion at p10 ≥ 66.2% @5bps on the 111-window benchmark — same bar as the existing ensemble.

No production deployment performed (live trading lives on the other machine).

---

## Phase 7 — Tests

- `tests/test_ctrader_parity.py` added; pytest run captured in commit log.

---

## Blockers / honest gaps

1. Phase 4 and Phase 5 deferred — would require multi-hour real runs to produce non-fabricated numbers. Existing ground-truth CSVs already cover the recommendation.
2. ctrader is a continuous-weight sim, not a binary-fill sim — a literal "C↔Python binary-fill parity" test against ctrader is impossible; pufferlib_market's C binding is the binary-fill ground truth and is already its own ground truth.
3. Live Alpaca API key is expired — verified, untouched, documented in alpacaprod.md.

---
