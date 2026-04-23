# cvar_portfolio — NVIDIA cuOpt/cuML Mean-CVaR optimizer on our universe

Replica of the NVIDIA DGX Spark playbook
[`nvidia/portfolio-optimization`](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/portfolio-optimization)
wired into our `trainingdata/` OHLCV panel. The upstream clone lives at
`dgx-spark-playbooks/` (gitignored); its `cufolio` package is installed via
`pip install -e dgx-spark-playbooks/nvidia/portfolio-optimization/assets/setup/`.

## What it does

At each rebalance date:

1. Take the trailing `fit_window` days of log returns for the working universe.
2. Fit scenarios (`kde` via cuML on GPU, `gaussian`, or `historical`) → build a
   CVaR scenario matrix `R ∈ ℝ^{n_assets × num_scen}`.
3. Solve the Mean-CVaR LP (CVXPY/Clarabel CPU or **cuOpt GPU PDLP**):

       min  λ · CVaR_α − μᵀw
       s.t. u + t + Rᵀw ≥ 0,  u ≥ 0
            Σ w + cash = 1
            w_min ≤ w ≤ w_max   (per-asset cap)
            c_min ≤ cash ≤ c_max
            ‖w‖₁ ≤ L_tar          (leverage cap)
            # optional:
            ‖w − w_prev‖₁ ≤ T_tar (turnover)
            Σ 𝟙[w_i ≠ 0] ≤ cardinality

4. Hold that allocation for `hold_days`, realise daily returns.
5. Repeat.

Output per run: `weights.parquet`, `daily_returns.csv`, `summary.json` with
sortino / ann-return / max-DD / neg-day frac / mean solve time.

## Sanity replication on our universe

Run end-to-end (100 most-liquid symbols, fit window 252d, hold 21d,
Gaussian scenarios, cuOpt GPU, 2022-01 → 2026-04):

    python -m cvar_portfolio.run_backtest \
      --symbols symbol_lists/stocks_wide_1000_v1.txt \
      --data-root trainingdata \
      --start 2022-01-01 --end 2026-04-18 \
      --max-symbols 100 --min-avg-dol-vol 50000000 \
      --fit-window 252 --hold-days 21 \
      --num-scen 2500 --fit-type gaussian \
      --w-max 0.10 --l-tar 1.0 \
      --api cuopt_python \
      --out analysis/cvar_portfolio/sanity_100sym_gauss

Result (`analysis/cvar_portfolio/sanity_100sym_gauss/summary.json`):

| Metric | CVaR@λ=1 w_max=10% | SPY (same window) |
|---|---:|---:|
| Ann return    | **+37.16 %**      | +22.57 %  |
| Ann vol       | 23.13 %           | 15.21 %   |
| Sortino       | 1.67              | 1.78      |
| Max DD        | −28.28 %          | −18.75 %  |
| Neg-day frac  | 43.1 %            | 42.8 %    |
| Mean solve    | 3.7 s (GPU PDLP)  | —         |
| Rebalances    | 40                | —         |

CVaR portfolio **beats SPY by ~15pp / yr on return** at similar neg-day rate
but pays ~10pp more DD and slightly lower sortino — i.e. the optimizer is
working as intended (longer tail of upside, not free risk reduction).

## Proposed integration with the prod Alpaca stack

### 1 — Alpha-aware Mean-CVaR: use XGB top-20 scores as `μ`

The Mean-CVaR LP accepts any expected-return vector. Empirical mean drifts
slowly and is noisy per asset. **Our XGB daily-trader produces a calibrated
score per symbol per day** (5-seed ensemble, `--model-path
live_model_train2020.pkl`). Map:

    μ_i = α · ensemble_score_normalised_i + (1 − α) · empirical_mean_i

`cvar_portfolio.optimize.solve_cvar_portfolio(..., mean_override=…)` is the
hook — pass a `(n_assets,)` array and the LP optimises against that alpha.

Why this is promising for our goodness score:

* Current LIVE xgb-daily-trader is **stuck in hold-cash mode** since tariff
  crash — top-1 `min_score=0.85` gate rarely fires and even when it does the
  risk is concentrated in one name. CVaR at `cardinality=10` with XGB scores
  as μ turns hold-cash into a *risk-capped diversified long book*: every
  rebalance the LP picks 5-20 names subject to `w_max=5%`, `L_tar=1.0`, tail
  loss ≤ CVaR. The `ms=0.85` gate becomes a **soft prior** on μ instead of a
  binary admit/reject. See
  `project_xgb_freshens_ms_sweep_unusable.md` — the gate is structurally
  broken in this regime, the LP sidesteps it.

* The LP's leverage constraint maps cleanly onto live-trader's
  `--allocation`: `L_tar` = gross target, `cash = 1 − Σw` replaces the stale
  `min_score` gate. We can still stress 36× fees in sim to pre-validate.

### 2 — CVaR as a risk-overlay on RL picks

`I_s3` solo RL policy is our first guard-robust RL edge (+12.77%/50d lev=2,
see `project_rl_i_s3_guard_robust_candidate.md`) but concentrates into 1–3
symbols per day. Feed the RL policy's softmax scores as μ to the CVaR layer
with `cardinality=K` and `w_max=1/K` to turn lumpy bets into a risk-aware
book while preserving the edge direction. Target: keep the +12%/mo median
but shave the −3.56 p10 by spreading positions.

### 3 — Map prod risk rules onto LP constraints

| Prod guarantee | LP constraint |
|---|---|
| HARD RULE #2 (singleton writer) | n/a — trades still go through `alpaca_wrapper` |
| HARD RULE #3 death-spiral guard | turnover `T_tar` ≤ 0.3 per rebalance + intraday-hold discipline → fewer sell refusals |
| `min_dollar_vol=50M` deploy floor | `cvar_portfolio.data.load_price_panel(min_avg_dollar_vol=5e7)` |
| `max_vol_20d=0.12` floor | pre-filter universe before passing to optimiser |
| `hold_through` (avoid churn) | `T_tar` with `weights_previous`, LP is stationary when signal is stationary |
| `cardinality` (`top_n=1` or `2`) | direct `cardinality` MILP constraint (cuOpt handles this in ~1 s at K=10, n=300) |

### 4 — Evaluation protocol (matches `alpacaprod.md` rules)

Before claiming anything deployable:

* **15-seed bonferroni** on the scenario rng (varying `--rng-seed`) — five
  scenario-sample seeds × three rebalance-offset seeds. Our current 2022-26
  baseline is 1-seed; collapse risk is real (see
  `project_xgb_5seed_tailbias_definitive_2026_04_22.md`).
* **Two OOS folds**: 2022-2023 pre-crash + 2025-26 post-crash, same
  constraints. Claim pos-median only if *both* folds positive.
* **Stress 36× fees / ≥5bps slippage** via our own `BacktestConfig` after
  weights are decided — the LP doesn't know about fees, so post-process the
  realised path through `src/backtest_portfolio_leverage.py` or equivalent.
* **Death-spiral-guard replay** (see `project_sim_audit_2026_04_22.md`): run
  the weight sequence through the Python sim with guard enabled before
  declaring the edge realisable.

### 5 — Deployment plan (if metrics pass)

If the alpha-aware CVaR backtest clears +27 %/mo stress36x and passes
guard-replay, deployment is additive and low-risk:

* New live-writer supervisor unit `xgb-cvar-portfolio-live` registered in
  `LIVE_WRITER_UNITS` (HARD RULE #2 registry), using `ALPACA_ACCOUNT_NAME`
  so it runs as a separate Alpaca account with its own singleton lock (same
  pattern as `crypto_weekend` embedded trader). This lets it co-exist with
  the stock xgb-daily-trader during shadow evaluation.
* Redeploy via `scripts/deploy_live_trader.sh` with the new unit name;
  never directly via `supervisorctl restart`.
* Trade log goes to `analysis/xgb_live_trade_log/YYYY-MM-DD.jsonl`.

### 6 — Why this *should* move goodness upward

Our goodness score is `p10 − worst_DD − 100·neg_frac`. CVaR-α is the LP's
explicit objective term → higher λ directly shrinks tail loss, so the
optimiser's own loss function is aligned with our goodness KPI. The live
XGB lever-stack already hits PnL but its DD profile is seed-lucky
(`project_xgb_alt5_seed_variant_no_deploy.md` found true DD ≈ 9% across
seeds). The LP produces a *constructive* tail-risk target rather than a
post-hoc filter.

## Files

* `data.py` — loader (`_load_close`, `load_price_panel`).
* `optimize.py` — `solve_cvar_portfolio()` wrapper with `mean_override` for
  XGB/RL alpha injection.
* `backtest.py` — rolling rebalance driver.
* `run_backtest.py` — CLI, writes weights + daily returns + summary json.
* `tests/test_cvar_portfolio_smoke.py` — 4 smoke tests (CVXPY, cuOpt,
  mean_override, end-to-end backtest).

## Constraints / gotchas

* `cufolio.cvar_optimizer.CVaR._scale_risk_aversion` rescales `λ` by
  `max(single-asset return / CVaR)` — interpret `risk_aversion=1.0` as
  "equal-weight return-vs-CVaR tradeoff before the scaling heuristic", not
  as a raw coefficient.
* cuOpt logs verbosely by default; we pass `log_to_console=False` in
  `solve_cvar_portfolio`.
* KDE on our full 846-symbol universe needs `kde_device=GPU`; CPU KDE is
  quadratic in samples and will be slow for `num_scen >= 5000`.
* Upstream clone in `dgx-spark-playbooks/` is gitignored — never commit.
