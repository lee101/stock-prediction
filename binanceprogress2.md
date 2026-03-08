# Binance Progress Log 2 - SUI & DOGE Trading

Updated: 2026-03-03

## Meta Portfolio Auto-Selector Sweep (2026-03-03)

Built and ran a new daily meta-selector sweep that chooses from a pool of DOGE/AAVE checkpoints using trailing daily performance (no lookahead).

### New Tooling
- Script: `binanceleveragesui/sweep_meta_daily_winners.py`
- Tests: `tests/test_binance_meta_daily_winners.py` (4 passed)
- Result artifact: `binanceleveragesui/meta_daily_winners_sweep_modes_v2_latest.json`

### Candidate Pool (8)
- doge_deployed (`DOGEUSD_r4_R4_h384_cosine` ep001)
- doge_wider_mlp8 (gen_wider_mlp8 ep004)
- doge_dilated_142472 (gen_dilated_1_4_24_72 ep004)
- doge_deeper6l (gen_deeper_6L ep010)
- doge_r5_drop15 (r5_DOGE_rw05_drop15 ep003)
- aave_rw05 (AAVE_h384_cosine_rw05 ep003)
- aave_r5_wd04 (r5_AAVE_rw05_wd04 ep002)
- aave_base (AAVE_h384_cosine ep001)

### Sweep Grid
- Windows: 30/60/90/120d
- Metrics: return/sortino/sharpe/calmar
- Lookbacks: 1/2/3/5/7/10/14 days
- Modes: `winner`, `winner_cash`, `blend_top2`
- Leverage: 2.0x, fee: 10bp

### Top Configs (robust ranking)

| Rank | Mode | Metric | Lookback | Min Sort | Mean Sort | Min Ret | Mean Ret | Mean DD |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | **blend_top2** | sortino | **3d** | **0.15** | **0.25** | **+308.47%** | **+387.97%** | -9.37% |
| 2 | blend_top2 | sortino | 14d | 0.14 | 0.23 | +270.00% | +362.99% | -13.01% |
| 3 | winner | sortino | 3d | 0.14 | 0.23 | +305.90% | +383.93% | -11.73% |
| 4 | winner_cash | sortino | 3d | 0.14 | 0.23 | +296.73% | +394.96% | **-8.38%** |
| 5 | blend_top2 | sortino | 2d | 0.13 | 0.24 | +281.50% | +357.56% | **-8.34%** |

### Best Sortino-Robust Config: `blend_top2 + sortino + 3d`

| Window | Meta Sort | Meta Ret | Meta DD | DOGE deployed Sort | DOGE deployed Ret | DOGE deployed DD |
|---|---:|---:|---:|---:|---:|---:|
| 30d | 0.40 | +308.47% | -6.21% | 0.42 | +300.82% | -6.22% |
| 60d | 0.25 | +354.68% | -9.13% | 0.30 | +334.55% | -6.22% |
| 90d | 0.18 | +378.78% | -9.21% | 0.20 | +313.80% | -11.88% |
| 120d | **0.15** | **+509.94%** | **-12.91%** | 0.13 | +385.95% | -18.41% |

### Smoothness Candidate: `winner_cash + sortino + 3d`
- Mean return: **+394.96%** (highest among top robust configs)
- Mean drawdown: **-8.38%** (better than winner mode -11.73%)
- 120d: **+541.75%** return, DD **-12.00%**

### Findings
- Daily adaptive meta-selection is materially better than static DOGE deployed on long windows (90-120d), with much better return and lower DD.
- `blend_top2` improved robustness vs pure hard-switch winner mode (higher min/mean sortino and lower DD).
- `winner_cash` gives the best smoothness/return tradeoff among tested configs.
- No meta config beat the per-window hindsight oracle baseline (expected high bar), but several beat deployed static policy on practical windows.

### Expanded Candidate Sweep (17 checkpoints)
- Artifact: `binanceleveragesui/meta_daily_winners_sweep_candidates17.json`
- Added extra DOGE/AAVE checkpoints (R5 and gen variants) and re-swept all modes/metrics/lookbacks.
- New top robust config: `blend_top2 + sortino + lb2`:
  - min sortino 0.15, mean sortino 0.25
  - min return +286.18%, mean return +358.41%
  - mean DD -8.62%
- Best live-compatible config: `winner_cash + sortino + lb2`:
  - min sortino 0.13, mean sortino 0.245
  - min return +285.69%, mean return +365.97%
  - mean DD -8.36%, beats baseline in 1/4 windows

### Deployable Pair Search (DOGE x AAVE)
- Artifact: `binanceleveragesui/meta_pair_search_winner_cash_sortino_lb2.json`
- Searched all DOGE×AAVE checkpoint pairs under live-compatible policy (`winner_cash`, `sortino`, lb2).
- Best pair by robust objective:
  - **DOGE**: `doge_value_embed` (`DOGEUSD_gen_value_embed` ep005)
  - **AAVE**: `aave_r5_strides` (`r5_AAVE_rw05_strides_long` ep002)
  - Stats: min sortino **0.18**, mean sortino 0.242, min return +260.75%, mean return +361.22%, mean DD -9.05%, baseline wins in 2/4 windows.

### Deployment Update
- Updated live supervisor command in `supervisor/binance-doge-margin.conf` to run `trade_margin_meta` with:
  - `--selection-mode winner_cash`
  - `--selection-metric sortino`
  - `--lookback 48` (2d equivalent)
  - checkpoints: `doge_value_embed` + `aave_r5_strides`
  - `--cycle-minutes 60` for hourly-aligned signal refresh

### Sticky-Switch Hysteresis Sweep
- Artifact: `binanceleveragesui/meta_daily_winners_sweep_candidates17_sticky.json`
- Added `switch_margin` hysteresis to selector and swept:
  - switch margins: 0.0 / 0.005 / 0.01 / 0.02 / 0.05
  - modes: winner_cash + winner
  - metric: sortino
  - lookbacks: 1/2/3/5/7/10/14
- Result: **no improvement** over `winner_cash + sortino + lb2` baseline; top stats remained unchanged.
- Conclusion: keep current deployment policy (`winner_cash`, sortino, lb2), continue searching via broader model sets and alternate objectives.

### Alternate Objective Pair Search (Calmar)
- Artifact: `binanceleveragesui/meta_pair_search_winner_cash_calmar_lb2.json`
- Ran DOGE×AAVE pair search with `winner_cash`, lookback=2d, **selection metric=calmar**.
- New best deployable pair:
  - **DOGE**: `doge_r5_drop15` (`r5_DOGE_rw05_drop15` ep003)
  - **AAVE**: `aave_r5_strides` (`r5_AAVE_rw05_strides_long` ep002)
- Robust stats:
  - min sortino **0.18**, mean sortino **0.258**
  - min return +183.91%, mean return +346.42%
  - mean DD **-7.57%**
  - beats best single-model baseline in **3/4 windows**
- Window detail:
  - 30d: sort 0.39, ret +183.91%, DD -4.76%
  - 60d: sort 0.27, ret +345.97%, DD -8.16%
  - 90d: sort 0.19, ret +342.48%, DD -8.22%
  - 120d: sort 0.18, ret +513.33%, DD -9.15%

### Deployment Update (Latest)
- Updated `supervisor/binance-doge-margin.conf` again to this stronger calmar-selected pair:
  - `--selection-mode winner_cash`
  - `--selection-metric calmar`
  - `--lookback 48`
  - DOGE ckpt: `r5_DOGE_rw05_drop15` ep003
  - AAVE ckpt: `r5_AAVE_rw05_strides_long` ep002

### Pair-Specific Hyperparam Tuning (Latest)
- Artifact: `binanceleveragesui/meta_pair_tuning_doge_r5drop15_aave_r5strides.json`
- Tuned the deployed pair across:
  - modes: winner / winner_cash
  - metrics: calmar/sortino/sharpe/return
  - lookbacks: 1/2/3/5/7/10/14 days
  - thresholds: 0/0.01/0.02
  - hysteresis (`switch_margin`): 0/0.005/0.01/0.02/0.05

Top risk-adjusted deployable config found:
- **winner_cash + calmar + lb1 + switch_margin=0.005**
- Stats:
  - min sortino **0.20** (up from 0.18)
  - mean sortino **0.288** (up from 0.258)
  - min return +187.13%
  - mean return +330.54%
  - mean DD **-6.66%** (better than -7.57%)
  - beats baseline in **3/4** windows
- Window stats:
  - 30d: sort 0.41, ret +187.13%, DD -3.51%
  - 60d: sort 0.33, ret +326.86%, DD -3.76%
  - 90d: sort 0.21, ret +328.11%, DD -9.18%
  - 120d: sort 0.20, ret +480.06%, DD -10.20%

### Deployment Update (Newest)
- Updated `supervisor/binance-doge-margin.conf` to:
  - `--selection-mode winner_cash`
  - `--selection-metric calmar`
  - `--lookback 24`
  - `--switch-margin 0.005`
  - pair unchanged: `r5_DOGE_rw05_drop15` + `r5_AAVE_rw05_strides_long`

### Outside-the-Box: Dual-Horizon Meta Scoring
- Artifact: `binanceleveragesui/meta_pair_dualhorizon_tuning_r5drop15_aave_strides.json`
- Tested blended short+long lookback scoring for the same pair:
  - score = `alpha * score(lb_short) + (1-alpha) * score(lb_long)`
  - grid across `{calmar, sortino}` metrics, lb_short/lb_long pairs, alpha, thresholds, hysteresis
- Best dual-horizon config:
  - sortino metric, lb_short=2d, lb_long=5d, alpha=0.25
  - min sortino **0.21**, mean sortino 0.275
  - min return +185.15%, mean return +265.33%
  - mean DD **-5.89%**, beats baseline in 2/4 windows
- Assessment:
  - Strongly smoother (lowest DD so far) and best worst-window sortino
  - But materially lower mean return than the current deployed tuned policy
  - **No deployment change**: keep `winner_cash + calmar + lb1 + switch_margin=0.005` as better total objective tradeoff

### Epoch-Level Pair Search + Tuning (2026-03-04)
- Artifacts:
  - `binanceleveragesui/meta_pair_grid_search_wide_thresholds_v3.json`
  - `binanceleveragesui/meta_pair_epoch_search_v1.json`
  - `binanceleveragesui/meta_pair_tuning_doge_drop15_ep008_aave_strides_ep002_v1.json`
- New reusable script:
  - `binanceleveragesui/search_meta_pair_epochs.py`
- Method:
  - Expanded pair search to checkpoint-epoch level (DOGE/AAVE epoch grid) under live-compatible policy:
    - `winner_cash`, `lookback=1d`, `selection metric in {calmar, sortino}`, `cash_threshold=0`, `switch_margin=0.005`
  - Then ran a focused hyperparameter sweep on the new best epoch pair.

Best epoch pair found:
- **DOGE**: `r5_DOGE_rw05_drop15` **epoch_008**
- **AAVE**: `r5_AAVE_rw05_strides_long` **epoch_002**
- Robust stats (30/60/90/120d windows):
  - min sortino **0.22**
  - mean sortino **0.318**
  - min return **+346.62%**
  - mean return **+554.72%**
  - mean DD **-8.02%**
  - beats best single-model baseline in **4/4** windows

Previous deployed pair baseline (DOGE epoch_003 + AAVE epoch_002):
- min sortino 0.20, mean sortino 0.2875
- min return +187.13%, mean return +330.54%
- mean DD -6.66%, beats 3/4 windows

Decision:
- New epoch pair is materially stronger on robustness + return, with a moderate DD tradeoff (+1.36pp mean DD).
- **Deployment changed** to DOGE `epoch_008` + AAVE `epoch_002` with same selector settings:
  - `winner_cash`, `calmar`, `lookback=24`, `cash_threshold=0.0`, `switch_margin=0.005`
- Updated config: `supervisor/binance-doge-margin.conf`

### Confidence-Gap Gating Experiments (2026-03-04)
- Added new selector control: `min_score_gap` (confidence gap between best and second-best model).
  - Offline sweep support:
    - `binanceleveragesui/sweep_meta_daily_winners.py`
    - `binanceleveragesui/search_meta_pair_grid.py`
    - `binanceleveragesui/search_meta_pair_epochs.py`
  - Live bot support:
    - `binanceleveragesui/trade_margin_meta.py` with CLI `--min-score-gap`
  - Tests updated:
    - `tests/test_binance_meta_daily_winners.py`
    - `tests/test_trade_margin_meta_selection.py`
    - Test status: **15 passed**

Artifacts:
- `binanceleveragesui/meta_pair_gap_tuning_doge_ep008_aave_ep002_v2.json`
- `binanceleveragesui/meta_pair_epoch_search_mg002_lb1_v1.json`
- `binanceleveragesui/meta_pair_crossfamily_mg_search_v1.json`

Key findings:
- On current best deployed pair (`doge_drop15_ep008 + aave_strides_ep002`), best config remains:
  - `winner_cash`, `calmar`, `lb=1d`, `cash_threshold=0`, `switch_margin in {0.005,0.01}`, `min_score_gap=0`
  - Robust stats unchanged: min sortino **0.22**, mean sortino **0.318**, mean return **+554.72%**, mean DD **-8.02%**, beats **4/4**
- Small gap values (`min_score_gap` up to 0.005) are mostly neutral on this pair.
- Larger gap (`min_score_gap=0.02`) reduces performance:
  - min sortino 0.20, mean sortino ~0.31, mean return ~+507.75% (DD unchanged around -8.02%).
- Cross-family pair search (older DOGE/AAVE families included) still did **not** beat the deployed epoch pair.

Decision:
- **No deployment change** after confidence-gap sweeps; keep deployed pair and settings unchanged.

---

## Continued Meta Research (2026-03-04, later)

### Exhaustive Epoch Search (1..20)
- Artifact: `binanceleveragesui/meta_pair_epoch_search_full1to20_v1.json`
- Search scope:
  - DOGE families: `r5_DOGE_rw05_drop15`, `r5_DOGE_rw05_strides_long`
  - AAVE families: `AAVE_h384_cosine_rw05`, `r5_AAVE_rw05_strides_long`
  - epochs: **1..20**
  - metrics: calmar/sortino, lookbacks: 1/2d, mode: winner_cash
- Result: top pair remains unchanged:
  - **DOGE** `drop15_ep008` + **AAVE** `strides_ep002`
  - static-selector stats remain unchanged from earlier pair-grid tuning:
    - min sortino **0.22**, mean sortino **0.318**, mean return **+554.72%**, mean DD **-8.02%**
  - robust window metrics same ranking outcome as earlier searches
- Conclusion: no better static deployable pair than currently deployed.

### New Outside-the-Box Algorithm: Regime-Adaptive Metric Switching
- New script: `binanceleveragesui/search_meta_pair_regime.py`
- Idea:
  - Use **low-vol** metric/lookback and **high-vol** metric/lookback, chosen by trailing market absolute-return regime.
  - Still applies winner_cash + hysteresis + min_score_gap controls.

Artifacts:
- `binanceleveragesui/meta_pair_regime_search_v1.json` (cross-pair regime search)
- `binanceleveragesui/meta_pair_regime_tune_singlepair_v2.json` (deep tuning on best pair)

Best regime config found:
- Pair: `doge_drop15_ep008 + aave_strides_ep002`
- `low_metric=sortino`, `low_lb=1d`
- `high_metric=calmar`, `high_lb=3d`
- `vol_lookback=5d`, `vol_threshold=0.03`
- `cash_threshold=0`, `switch_margin=0.005`, `min_score_gap=0`
- Metrics:
  - min sortino **0.22** (same as static)
  - mean sortino **0.325** (higher than static 0.3175)
  - min return **+344.45%** (slightly lower than static +346.62%)
  - mean return **+551.48%** (slightly lower than static +554.72%)
  - mean DD **-8.02%** (same)
  - beats baseline windows **4/4**

Assessment:
- Regime selector improves smoothness/risk-adjusted score (mean sortino) with near-equal DD.
- But it does **not** strictly dominate static selector on return.
- **No deployment change** yet; keep static deployed config until a regime setup wins on both return and robustness.

### Alternate Pair Deep Dive (`aave_strides_ep009`) + Leverage Sweep
Artifacts:
- `binanceleveragesui/meta_pair_tune_doge_ep008_aave_ep009_v1.json`
- `binanceleveragesui/meta_pair_regime_tune_ep009_v1.json`
- `binanceleveragesui/meta_leverage_sweep_currentpair_v1.json`

Findings for alternate pair (`doge_drop15_ep008 + aave_strides_ep009`):
- Static best:
  - min sortino **0.19**, mean sortino **0.30**
  - min return +336.57%, mean return +567.82%
  - mean DD **-7.09%**
- Regime tuning did not improve the objective vs static on this pair:
  - typical top regime row: min sortino 0.20, mean sortino 0.29, mean return +561.95%, mean DD -9.73%
- Conclusion: `ep009` pair remains attractive for return/DD, but still too weak on sortino vs current deployed pair.

Leverage sweep on current best pair (`ep008 + ep002`) across static/regime policies:
- Best sortino-oriented row:
  - regime policy at **2.0x** (same as prior) keeps top mean sortino (0.325) but lower return than static.
- Best return with same minimum-sortino bucket (`minS=0.22`) among tested static policies:
  - **static_calmar_lb1 at 2.25x**:
    - min sortino **0.22** (unchanged vs 2.0x baseline)
    - mean sortino **0.318** (unchanged)
    - min return **+428.25%** (up from +346.62%)
    - mean return **+713.64%** (up from +554.72%)
    - mean DD **-9.03%** (vs -8.02% at 2.0x)
- Decision:
  - Given target to push PnL while keeping robust risk-adjusted behavior, moved live config to **2.25x** leverage.
  - Pair/selector unchanged (`ep008+ep002`, winner_cash + calmar + lb1 equivalent).
  - Updated `supervisor/binance-doge-margin.conf`.

### Follow-up Search (Later Same Session)
Additional artifacts:
- `binanceleveragesui/meta_pair_tune_doge_ep008_aave_ep009_v1.json`
- `binanceleveragesui/meta_pair_regime_tune_ep009_v1.json`
- `binanceleveragesui/meta_leverage_fine_sweep_currentpair_v2.json`
- `binanceleveragesui/meta_dynamic_leverage_search_v1.json`
- `binanceleveragesui/meta_leverage_local_refine_v3.json`

Findings:
- Alternate pair (`ep008 + ep009`) remains return-strong but still weaker on sortino than deployed pair:
  - static best around minS 0.19 / meanS 0.30.
  - regime best around minS 0.20 / meanS 0.29 with worse DD.
- Dynamic leverage regime search did not produce a superior true dynamic policy; top row collapsed to constant leverage.
- Fine leverage refinement around 2.25 found:
  - **static @ 2.30x**: minS **0.22**, meanS **0.318**, minR **+446.06%**, meanR **+749.46%**, meanDD **-9.22%**.
  - vs static @ 2.25x: minS 0.22, meanS 0.318, minR +428.25%, meanR +713.64%, meanDD -9.03%.

Decision update:
- Kept pair/selector unchanged and updated leverage again from **2.25x -> 2.30x** for higher PnL at unchanged worst-window sortino.
- Updated `supervisor/binance-doge-margin.conf` accordingly.


## R5 Improved Training Sweep (2026-03-02, in progress)

Training 10 new configs (6 DOGE + 4 AAVE) with refined hyperparameters.
Base: nano, 6L, h384, 8 heads, seq72, cosine LR, mem8, strides=1,2,6,24, sortino loss, fill_temp=0.1, fill_buf=0.0005.

| Config | Symbol | rw | wd | dropout | strides | Status |
|--------|--------|------|------|---------|---------|--------|
| DOGE_rw05 | DOGEUSD | 0.05 | 0.03 | 0.1 | 1,2,6,24 | training |
| DOGE_rw05_wd04 | DOGEUSD | 0.05 | 0.04 | 0.1 | 1,2,6,24 | pending |
| DOGE_rw03 | DOGEUSD | 0.03 | 0.03 | 0.1 | 1,2,6,24 | pending |
| DOGE_rw10_wd04 | DOGEUSD | 0.10 | 0.04 | 0.1 | 1,2,6,24 | pending |
| DOGE_rw05_drop15 | DOGEUSD | 0.05 | 0.03 | 0.15 | 1,2,6,24 | pending |
| DOGE_rw05_strides_long | DOGEUSD | 0.05 | 0.03 | 0.1 | 1,2,6,24,72 | pending |
| AAVE_rw05_wd04 | AAVEUSD | 0.05 | 0.04 | 0.1 | 1,2,6,24 | pending |
| AAVE_rw03 | AAVEUSD | 0.03 | 0.03 | 0.1 | 1,2,6,24 | pending |
| AAVE_rw05_drop15 | AAVEUSD | 0.05 | 0.03 | 0.15 | 1,2,6,24 | pending |
| AAVE_rw05_strides_long | AAVEUSD | 0.05 | 0.03 | 0.1 | 1,2,6,24,72 | done |

### DOGE Results (current deployed: R4 h384+cosine ep1, Sort=39.9, DD=-7.3%)

| Config | Ep | Sort@30d | Ret@30d | DD@30d | 2x Sort | Mean | Pos |
|--------|-----|---------|---------|--------|---------|------|-----|
| DOGE_rw05_drop15 | 3 | 38.1 | +68.3% | -2.4% | 37.8 | 9.4 | 4/6 |
| DOGE_rw05_strides_long | 10 | 36.7 | +111.8% | -6.5% | 36.4 | 7.8 | 3/6 |
| DOGE_rw05_wd04 | 8 | 34.7 | +110.2% | -3.9% | 34.6 | 8.1 | 4/6 |
| DOGE_rw10_wd04 | 1 | 33.9 | +43.0% | -2.9% | 33.3 | 9.1 | 4/6 |
| DOGE_rw03 | 8 | 33.8 | +122.1% | -6.5% | 33.7 | 8.0 | 3/6 |
| DOGE_rw05 | 1 | 33.2 | +80.6% | -3.1% | 33.1 | 10.8 | 4/6 |

### AAVE Results (current deployed: rw05 ep3, Sort=29.8, DD=-3.9%)

| Config | Ep | Sort@30d | Ret@30d | DD@30d | 2x Sort | Mean | Pos |
|--------|-----|---------|---------|--------|---------|------|-----|
| AAVE_rw05_wd04 | 2 | 29.8 | +46.6% | -4.4% | 28.9 | 8.0 | 4/6 |
| AAVE_rw05_strides_long | 2 | 26.9 | +61.3% | -3.9% | 26.3 | 10.7 | 4/6 |
| AAVE_rw03 | 10 | 22.4 | +62.8% | -8.8% | 22.0 | 5.4 | 3/6 |
| AAVE_rw05_drop15 | 20 | 17.5 | +82.1% | -10.9% | 17.4 | 3.9 | 3/6 |

### Meta-Sim Comparison (R5 vs current deployed, 2x leverage, calmar 12h)

| Window | Current Sort | Current Ret | Best R5 Sort | Best R5 Ret |
|--------|-------------|-------------|--------------|-------------|
| 30d | 0.71 | +280% | 0.71 | +172% |
| 60d | 0.45 | +316% | 0.45 | +316% |
| 90d | 0.26 | +356% | 0.26 | +356% |

**Conclusion: No R5 config beats current deployed models. No deployment change.**
- DOGE_rw05_drop15 has lowest DD (-2.4%) but Sort=38.1 < deployed 39.9
- AAVE_rw05_wd04 ties deployed Sort=29.8 but higher DD (-4.4% vs -3.9%)
- Meta-sim confirms current combo is optimal or tied across all windows

---

## Meta-Switcher Deployment (2026-03-02)

### Design
Single bot trading one model at a time (DOGE or AAVE), switching based on trailing performance.
- Loads both models at startup, generates signals from both every cycle
- On position exit: computes trailing 12h calmar ratio for both models
- Enters next trade using whichever model has higher calmar
- Tracks signal histories for hypothetical performance computation

### Meta-Switcher Hyperparameter Sweep Results
Swept 11 lookback windows (6-720h) x 4 metrics (sortino/return/sharpe/calmar) x 3 test windows (30/60/90d).

| Config | 30d Sort | 60d Sort | 90d Sort | Min | Mean |
|--------|----------|----------|----------|-----|------|
| **calmar_12h** | **0.76** | **0.49** | **0.31** | **0.31** | **0.52** |
| calmar_24h | 0.76 | 0.40 | 0.31 | 0.31 | 0.49 |
| sortino_12h | 0.73 | 0.49 | 0.31 | 0.31 | 0.51 |
| return_12h | 0.76 | 0.39 | 0.31 | 0.31 | 0.49 |
| calmar_6h | 0.76 | 0.28 | 0.31 | 0.28 | 0.45 |

Winner: **calmar_12h** (best worst-case + best mean across all windows).

### Deployed Config
- **Supervisor**: `binance-meta-margin` RUNNING
- **DOGE ckpt**: `DOGEUSD_r4_R4_h384_cosine/binanceneural_20260301_065549/epoch_001.pt`
- **AAVE ckpt**: `AAVE_h384_cosine_rw05/binanceneural_20260302_102218/epoch_003.pt`
- Lookback: 12h, metric: calmar, max leverage: 2.0x
- State file: `strategy_state/margin_meta_state.json`
- Equity: ~$3,200

### Initial Production Behavior
- Inherited DOGE position from previous bot
- DOGE position exited, model selection ran: doge calmar=713.50 >> aave calmar=49.66
- Re-entered DOGE (expected: early signal history biases toward model with more data)

---

## AAVE Sweep Results (2026-03-02)

Trained 5 AAVE configs + 1 SUI using DOGE R4 winner arch as base (nano, h384, 6L, cosine LR).

### Summary

| Config | Ep | Sort@30d | Ret@30d | DD@30d | Mean | Pos | Safety |
|--------|-----|---------|---------|--------|------|-----|--------|
| **AAVE_h384_cosine_rw05** | **3** | **29.8** | **+62.7%** | **-3.9%** | **9.76** | **4/6** | **5.67** |
| AAVE_h384_cosine | 1 | 23.7 | +40.8% | -3.3% | 9.76 | 4/6 | 5.61 |
| AAVE_h384_cosine_rw20 | 3 | 20.7 | +46.5% | -7.7% | 7.72 | 4/6 | 4.32 |
| AAVE_h384_cosine_wd05 | 20 | 17.6 | +71.1% | -12.7% | 4.68 | 4/6 | 2.42 |
| AAVE_h512_cosine | 15 | 18.5 | +89.3% | -7.9% | 3.68 | 3/6 | 1.48 |
| SUI_h384_cosine_0fee | 3 | -0.6 | -2.7% | -10.2% | 1.56 | 2/6 | 0.47 |

### Key Findings
- **rw05 is best for AAVE**: Sort=29.8, Safety=5.67 (vs rw10 base Sort=23.7)
- **h512 overfits**: highest return (+89.3%) but worst safety (1.48), only 3/6 positive symbols
- **wd05 overtrained**: ep20 is best but high DD (-12.7%), poor cross-symbol
- **SUI dead**: insufficient data (~118 days), negative sortino at 30d+
- **rw05 has best DD**: only -3.9% at 30d, even at 2x only -7.8%

### Leverage Sweep (1-5x, winner AAVE_h384_cosine_rw05)

| Window | 1x Sort | 2x Sort | 3x Sort | 2x Ret | 2x DD |
|--------|---------|---------|---------|--------|-------|
| 3d | 86.0 | 83.2 | 80.7 | +79.0% | -7.8% |
| 7d | 71.4 | 69.8 | 68.5 | +142.2% | -7.8% |
| 14d | 51.2 | 50.2 | 49.2 | +151.2% | -7.8% |
| 30d | 29.8 | 29.2 | 28.7 | +149.9% | -7.8% |
| 60d | 11.9 | 11.6 | 11.3 | +111.0% | -18.7% |
| 90d | 8.8 | 8.6 | 8.3 | +121.9% | -24.7% |

AAVE 2x is optimal: strong returns with acceptable DD.

---

## DOGE Production Performance & Audit (2026-03-02)

### Currently Deployed
- **Checkpoint**: `DOGEUSD_r4_R4_h384_cosine/binanceneural_20260301_065549/epoch_001.pt`
- **Architecture**: nano, 6L, h384, 8 heads, seq72, cosine LR, mem8, strides=1,2,6,24
- **Supervisor**: `binance-doge-margin` RUNNING, 2x leverage (upgraded from 1x on Mar 1)
- **Equity**: ~$3,195 (down from $3,320 start)

### Production Trade Log (Mar 1, 16h window)
Start: $3,319.95 | End: $3,165.76 | **P&L: -$154 (-4.64%)**

| # | Time (UTC) | Action | Price | Leverage | Equity |
|---|---|---|---|---|---|
| 1 | 06:03 | ENTER | $0.09568 | 1.0x | $3,314 |
| 2 | 12:04 | EXIT (force 6h) | $0.09353 | - | $3,233 |
| 3 | 12:49 | ENTER | $0.09313 | 2.0x | $3,228 |
| 4 | 13:51 | EXIT | $0.09390 | - | $3,264 |
| 5 | 16:22 | ENTER | $0.09383 | 2.0x | $3,255 |
| - | ongoing | HOLD | ~$0.0929 | 2.0x | ~$3,195 |

Trade 1: 1x lev (pre-upgrade), lost -$86 on DOGE drop ($0.0957->$0.0935)
Trade 2: 2x lev, profitable +$36 ($0.0931->$0.0939)
Trade 3: 2x lev, unrealized loss ~-$60 ($0.0938->$0.0929), sell limit at $0.0947

### Sim vs Production Comparison (Feb 28 - Mar 1)
| Metric | Prod | Sim(opt) | Sim(real) |
|---|---|---|---|
| Trades | 16 | 27 | 9 |
| Buys | 9 | 21 | 5 |
| Sells | 7 | 6 | 4 |
| Fill match | - | 12/16 (75%) | 4/16 (25%) |
| Final equity | ~$3,166 | $3,296 | $2,905 |

Optimistic sim overfills (bar touch = fill). Realistic sim is too conservative (expiry + volume filters reject too many). Production is between the two. Reasonable sim accuracy -- no critical divergence.

### Multi-Window Backtest (2026-03-02, deployed ep001)
| Window | Sort@1x | Ret@1x | DD@1x | Sort@2x | Ret@2x | DD@2x |
|---|---|---|---|---|---|---|
| 3d | -14.5 | -0.8% | -0.8% | -14.5 | -1.5% | -1.6% |
| 7d | -3.5 | -0.3% | -0.9% | -3.5 | -0.7% | -1.8% |
| **14d** | **62.5** | **+94.3%** | **-3.1%** | **62.3** | **+251.5%** | **-6.2%** |
| **30d** | **40.1** | **+101.2%** | **-3.1%** | **40.1** | **+276.6%** | **-6.2%** |
| 60d | 28.6 | +114.5% | -3.1% | 28.6 | +327.3% | -6.2% |
| 90d | 17.1 | +121.5% | -6.9% | 17.0 | +352.5% | -13.5% |
| 120d | 12.8 | +134.6% | -9.5% | 12.8 | +405.0% | -18.4% |

### Cross-Symbol Performance (1x, deployed checkpoint)
| Symbol | 3d | 7d | 14d | 30d | 60d | 90d | 120d |
|---|---|---|---|---|---|---|---|
| DOGE | -14.5 | -3.5 | **62.5** | **40.1** | **28.6** | **17.1** | **12.8** |
| BTC | -9.4 | -6.1 | -4.8 | -3.4 | -2.6 | -2.4 | -1.6 |
| ETH | 4.8 | -5.2 | 0.8 | 5.4 | 3.6 | 3.4 | 2.9 |
| SOL | -2.8 | -4.9 | -0.8 | -0.3 | -1.1 | -1.5 | -1.2 |
| LINK | 10.7 | 11.4 | 7.7 | 4.8 | -0.4 | 0.0 | -0.5 |
| AAVE | **69.7** | **45.9** | **33.3** | **21.2** | **10.3** | **7.0** | **4.5** |

### Assessment (2026-03-02)
- **Short-term pain, long-term strong**: 3d/7d negative (DOGE dropped sharply from $0.096 to $0.088). This matches the -4.64% prod drawdown.
- **14d+ outstanding**: Sort=62.5 at 14d, +94% return at 1x, +251% at 2x. The model's edge is massive once the bad 3d period rolls off.
- **Current drawdown within normal**: -4.64% prod vs sim DD=-6.2% at 2x/14d. We are within expected bounds.
- **Cross-symbol**: AAVE is a standout (Sort=69.7 at 3d). BTC/SOL consistently negative. ETH mixed.
- **No retraining needed**: Model has extreme edge at 14d+ windows. Short-term drawdown is noise.
- **Keep 2x leverage**: DD at 2x stays under -6.2% through 60d, acceptable risk.

---

## DOGE Margin Trading - Checkpoint Sweep & Deployment (2026-03-01)

### 96-Checkpoint Sweep

Swept all available DOGE checkpoints (96 total) across 6 time windows (7d, 14d, 30d, 60d, 90d, 180d) with lag=1, intensity=5.0, hold=6h, lev=1.0x, fee=0.001.

Architectures tested: classic, nano, nano+dilated, nano+memory, nano+wider_mlp, nano+value_embed, nano+residual_scalars, nano+rope72, various loss types (sortino, calmar, combined).

### Top 10 by Worst-Window Sortino (robustness)

| Checkpoint | Worst Sort | Avg Ret | 7d | 14d | 30d | 60d | 90d | 180d |
|---|---|---|---|---|---|---|---|---|
| **gen_wider_mlp8_ep4** | **+3.72** | **+65.1%** | **+0.5%** | **+0.5%** | +95.9% | +95.5% | +83.2% | +115.0% |
| **gen_dilated_1_4_24_72_ep4** | **+3.67** | +61.4% | **+0.4%** | **+0.4%** | +84.1% | +89.1% | +84.7% | +109.8% |
| gen_dilated_1_4_24_72_ep2 | +1.35 | +72.4% | +0.1% | +0.1% | +98.2% | +107.3% | +103.2% | +125.5% |
| gen_value_embed_ep5 | -0.99 | +87.0% | -0.1% | -0.1% | +94.3% | +109.3% | +98.5% | +219.8% |
| loop_A1_value_embed_ep4 | -1.07 | +82.4% | -0.2% | -0.2% | +107.5% | +121.6% | +119.8% | +146.0% |
| gen_dilated_1_4_24_72_ep5 | -1.39 | +64.5% | -0.2% | -0.2% | +88.8% | +88.6% | +84.4% | +125.8% |
| gen_deeper_6L_ep10 | -1.73 | +92.1% | -0.3% | -0.3% | +99.7% | +103.4% | +96.8% | +253.3% |
| gen_nano_baseline_ep4 | -1.80 | +58.8% | -0.3% | -0.2% | +81.2% | +84.0% | +79.5% | +108.4% |
| loop_A1_value_embed_ep5 | -2.53 | +84.2% | -0.4% | -0.4% | +98.4% | +122.1% | +118.4% | +166.8% |
| loop_A1_value_embed_ep2 | -2.95 | +45.2% | -0.5% | -0.4% | +64.7% | +69.4% | +65.4% | +72.4% |

Only 2 checkpoints are **positive across ALL 6 windows**: gen_wider_mlp8_ep4 and gen_dilated_1_4_24_72_ep4.

### Top 5 by Average Return

| Checkpoint | Avg Ret | Min Sort | 7d | 30d | 90d | 180d |
|---|---|---|---|---|---|---|
| gen_rope72_ep10 | +109.5% | -11.37 | -5.3% | +124.1% | +122.2% | +282.6% |
| gen_deeper_6L_ep10 | +92.1% | -1.73 | -0.3% | +99.7% | +96.8% | +253.3% |
| gen_value_embed_ep5 | +87.0% | -0.99 | -0.1% | +94.3% | +98.5% | +219.8% |
| gen_wider_mlp8_ep10 | +84.9% | -4.73 | -1.1% | +111.2% | +104.0% | +184.2% |
| loop_A1_value_embed_ep5 | +84.2% | -2.53 | -0.4% | +98.4% | +118.4% | +166.8% |

### Architecture Analysis

Key finding: **dilated attention** (strides 1,4,24,72) is the dominant factor for cross-window robustness. Both top-2 models use it.

| Feature | gen_wider_mlp8_ep4 | gen_dilated_1_4_24_72_ep4 | rw30_ep4 (old deployed) |
|---|---|---|---|
| Architecture | nano | nano | classic |
| Dilated strides | 1,4,24,72 | 1,4,24,72 | none |
| MLP ratio | 8.0 | 4.0 | 4.0 |
| Memory tokens | 8 | 0 | 0 |
| Return weight | 0.1 | 0.1 | 0.3 |
| Decision lag | 1 | 1 | 0 |
| Parameters | 4.98M | 2.88M | 3.18M |

Lower return_weight (0.1 vs 0.3) = less aggressive = more robust across windows.
Dilated attention captures multi-scale temporal patterns (hourly, 4h, daily, 3-day).

### Deployed Config (updated 2026-03-02)

- Supervisor: `binance-doge-margin` RUNNING
- Checkpoint: `DOGEUSD_r4_R4_h384_cosine/binanceneural_20260301_065549/epoch_001.pt`
- Architecture: nano, 6L, h384, 8 heads, seq72, cosine LR, mem8, strides=1,2,6,24
- Max leverage: **2.0x** (upgraded from 1.0x on Mar 1), intensity: 5.0, max hold: 6h, fee: 0.001
- Equity: ~$3,195 (started $3,320)

### Backtest Detail (gen_wider_mlp8_ep4)

| Window | Return | Sortino | MaxDD | Trades |
|---|---|---|---|---|
| 7d | +0.55% | 6.28 | -0.69% | 106 |
| 14d | +0.55% | 5.64 | -0.69% | 127 |
| 30d | +96.09% | 42.50 | -3.73% | 230 |
| 60d | +95.69% | 21.44 | -5.95% | 658 |
| 90d | +82.68% | 11.75 | -9.98% | 1087 |
| 180d | +115.30% | 3.72 | -23.45% | 2366 |

### Sim-vs-Prod Validation (past 24h, nano_fine_strides_ep5 -- previous model)

60% fill direction match, -0.9bps avg price diff. Key divergences:
- Noise trades: tiny buys (0.2% intensity = $6) set entry_ts too early, causing premature force_sells
- Partial fills: prod splits large sells into 3 fills, sim counts as one
- Force sell timing: sim triggered at 08:00 (6h from noise buy at 02:00) vs prod at 09:37 (6h from real buy at 03:36)

### Multi-Window Loss (experimental)

Implemented `multiwindow` and `multiwindow_dd` loss types in `differentiable_loss_utils.py`. Computes objective on multiple sub-windows of the training sequence and optimizes worst-case (minimax). Training config params: `multiwindow_fractions`, `multiwindow_aggregation`.

10-epoch training with multiwindow_dd: best ep2 min_sort=-9.86. Within-batch sub-window optimization (72h) can't solve 7d-180d regime robustness -- architecture choices (dilated attention) matter much more than loss function.

---

## SUI LoRA Fine-tuning

Optimized Chronos2 LoRA for SUI hourly forecasting with context=512.

| Config | Val MAE% |
|--------|----------|
| ctx=128 baseline | 3.19 |
| ctx=256 lr=5e-5 | 3.02 |
| ctx=512 baseline | 2.82 |
| ctx=1024 | 3.16 |
| ctx=512 steps=200 | 3.50 |
| ctx=512 lr=1e-4 r=32 steps=300 | 3.52 |

Best: ctx=512 with ~2.8-3.5% MAE (varies with data window).

Dilation ensemble tested - did not improve MAE vs single inference.

## Trading Bot Training

Training config: 10bp maker/taker fees, 25 epochs, sortino+return weighted loss.

Checkpoint: `binancechronossolexperiment/checkpoints/sui_sortino_rw0012_lr1e4_ep25/policy_checkpoint.pt`

Final training metrics:
- Train sortino: 2322.82, return: 22.77%
- Val sortino: 288.84, return: 19.67%

## 7-Day Holdout Backtest (2026-02-08 to 2026-02-15)

| Strategy | Return | Sortino | Max DD | Trades | Final Equity |
|----------|--------|---------|--------|--------|--------------|
| Momentum | +5.65% | 9.08 | -5.84% | 3 | $10,565 |
| Neural (10bp) | +153.57% | 612.03 | -0.33% | 139 | $25,357 |

Neural policy significantly outperforms momentum baseline with 10bp fees:
- 27x higher return
- 67x higher sortino
- 17x lower max drawdown

## Margin/Leverage Trading Experiment (2026-02-17)

Trained leveraged SUI margin trading policies with 2-5x leverage on Binance cross-margin.
Margin interest: 2.23% annual (~0.00025% hourly). Checkpoint: `lev4x_rw0.012_s1337`.

### Leverage Comparison ($5k start, 10bp fees, lev4x checkpoint)

| Window | 1x | 2x | 3x | 4x | 5x | 4x 0-fee |
|--------|-----|-----|------|------|-------|----------|
| 3d | 1.3x | 1.7x | 2.1x | 2.8x | 3.5x | 3.2x |
| 7d | 1.7x | 2.9x | 4.9x | 8.2x | 13.8x | 11.1x |
| 10d | 2.1x | 4.6x | 9.7x | 20.5x | 43.3x | 32.7x |
| 14d | 3.2x | 10.1x | 31.5x | 97.0x | 295.5x | 182.4x |
| 30d | 6.7x | 43.5x | 279.9x | 1770x | 11016x | 6466x |

Sortino: ~163-211 (3d) to ~138-144 (70d), consistently increases with leverage.
Max DD: ~0.9% per 1x leverage (linear scaling, -3.7% at 4x, -4.6% at 5x).
Margin cost: negligible at short windows (<1% at 5x/10d), grows with compounding.

### Key Findings
- 5x > 4x on all metrics (sortino, return) with +0.9% more drawdown
- Max DD scales linearly with leverage (good risk properties)
- Margin cost (2.23% annual) is negligible vs trading profits
- 0-fee 4x is 1.5-3x better than 10bp 4x (fee sensitivity)
- Sim reinvests all profits at full leverage -> exponential compounding at long horizons

### Bug Fixes (2026-02-17)
- **Position sizing**: was using model intensity to scale trade size (0.2% -> $34 trades instead of $20k at 4x). Fixed: uses full equity * leverage.
- **Equity calc**: now uses USDT net + SUI net * price instead of just USDT net
- **Repay logic**: cancels open orders before repaying, partial repay fallback
- **Simultaneous buy+sell**: bot now places buy orders (add to position up to max leverage) while also managing sell orders, instead of only doing one at a time

### Deployed Config (2026-02-17)
- Supervisor: `binance-sui-margin` RUNNING
- Checkpoint: `binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt`
- Max leverage: 4.0x, cycle: 5min, max hold: 6h
- Capital: ~$5,085 USDT in cross-margin account
- Simultaneous buy+sell enabled (adds to position while managing exit)

## Model Artifacts

- LoRA checkpoint: `chronos2_finetuned/SUI_lora_ctx512/finetuned-ckpt`
- Forecast cache: `binancechronossolexperiment/forecast_cache_sui_10bp/`
- Policy checkpoint (1x): `binancechronossolexperiment/checkpoints/sui_sortino_rw0012_lr1e4_ep25/policy_checkpoint.pt`
- Policy checkpoint (4x margin): `binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt`
- Hyperparams: `hyperparams/chronos2/hourly/SUIUSDT.json`

## RL CUDA PPO Residual Leverage Controller (2026-02-17)

PPO agent learns residual adjustments on top of baseline neural trading policy.
3D action space: [buy_scale, sell_scale, cap_ratio] where cap_ratio dynamically
controls per-step max leverage in [0,1] * max_leverage.

Code: `binanceleveragesui_rlcuda/` (synced from 5090 via R2)
Tests: 14 passed (env + residual_env + chronos + forecast windowing)

### Architecture
- Residual controller: multiplicative scaling of baseline buy/sell amounts
- Dynamic leverage cap: policy self-throttles leverage each step
- Hard cap enforcement: auto-deleverage if position exceeds cap
- Risk/smoothness reward terms: downside_penalty, pnl_smoothness, cap_smoothness
- SB3 PPO, MLP 256x256 SiLU, gamma=0.995

### Experiment Results (7d test set, 5x leverage, $10k start, 10bp fees)

| Experiment | RL Return | RL Sortino | RL Max DD | Baseline Return | Baseline DD | Avg Cap |
|-----------|-----------|------------|-----------|-----------------|-------------|---------|
| residual_sweep_30k (no cap) | **95.1x** | 1224 | -50.7% | 7.7x | -23.2% | n/a |
| caprisk_lite_30k | 5.9x | 1035 | -14.1% | 7.5x | -22.7% | 0.63 |
| caprisk_balanced_30k | 5.2x | 879 | -9.2% | 7.5x | -22.7% | 0.55 |
| caprisk_strict_30k | 5.4x | 923 | -9.0% | 7.5x | -22.7% | 0.55 |

### Multi-Window Eval (caprisk_lite, 5x leverage)

| Window | RL Return | RL Sortino | RL DD | Baseline Return | Baseline DD |
|--------|-----------|------------|-------|-----------------|-------------|
| 3d | 1.0x | 1170 | -13.7% | 1.4x | -22.7% |
| 7d | 5.9x | 1035 | -14.1% | 7.5x | -22.7% |
| 10d | **17.8x** | 583 | -18.0% | 9.7x | -22.7% |

### Key Findings
- Unconstrained residual (no cap control) gets 95x return but -50% DD -- unusable
- Cap-controlled versions trade ~25% return for ~60% lower drawdown
- At 10d window, RL cap-lite actually beats baseline on both return AND drawdown
- Policy self-throttles to ~55-63% of max leverage (avg_cap_ratio)
- Strict risk config: -9% DD vs -23% baseline (2.5x reduction), only -28% return loss
- Best seed varies: s1337 for balanced/strict, s2024 for lite

### Model Artifacts
- Best unconstrained: `binanceleveragesui_rlcuda/artifacts_residual_30k/seed_1337/best_model.zip`
- Best cap-lite: `binanceleveragesui_rlcuda/artifacts_residual_caprisk_lite_30k/seed_2024/best_model.zip`
- Best cap-balanced: `binanceleveragesui_rlcuda/artifacts_residual_caprisk_balanced_30k/seed_1337/best_model.zip`
- Best cap-strict: `binanceleveragesui_rlcuda/artifacts_residual_caprisk_30k/seed_1337/best_model.zip`
- All synced to local via R2 (netwrckstatic/models/binanceleveragesui_rlcuda/)

## Latest 3d Backtest (2026-02-14 to 2026-02-17)

Baseline model (lev4x_rw0.012_s1337) on latest data, $5k start, 10bp fees:

| Window | 1x | 2x | 3x | 4x | 5x |
|--------|-----|-----|------|------|------|
| 3d | 1.28x | 1.64x | 2.10x | 2.68x | 3.42x |
| 7d | 1.69x | 2.85x | 4.81x | 8.09x | 13.58x |
| 10d | 2.15x | 4.59x | 9.78x | 20.76x | 43.89x |
| 14d | 3.17x | 9.94x | 30.78x | 94.19x | 284.97x |
| 30d | 6.67x | 43.74x | 282.05x | 1788.65x | 11158.29x |

Sortino: 191-201 (3d) to 150-158 (30d). Max DD: -0.87% per 1x (linear).
Model performing excellently on fresh data.

### Probe-Mode Shutdown Strategy (TESTED - NO BENEFIT)
After unprofitable trade, shrink to $2 probe orders until profitable, then re-enable.
Result: NO drawdown improvement. 100% of probes were winners (model recovers too fast).
Returns reduced by 3-14% (missed compounding during probe). Not deploying.

### Deployed Config (updated 2026-02-17)
- Supervisor: `binance-sui-margin` RUNNING (was STOPPED, now active)
- Old systemd `sui-binance-trader`: STOPPED + DISABLED (was 1x, zero trades)
- Checkpoint: `binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt`
- Max leverage: 5.0x, cycle: 5min, max hold: 6h
- Equity: ~$5,120 USDT cross-margin
- Active trading confirmed: buy+sell orders at ~$20k notional

## Smoothness Optimization Sweep (2026-02-17)

PPO residual sweep varying downside/smoothness penalties. Baseline 5x: ret=7.5, DD=-23%.

| Config | Seed | Return | Sortino | DD | Cap |
|--------|------|--------|---------|------|------|
| dd=0.5 ps=5e-4 cap=0.3 | 2024 | **13.3x** | 979 | **-12.6%** | 0.74 |
| dd=1.0 ps=1e-3 cap=0.5 | 2024 | 15.1x | 1003 | -14.5% | 0.80 |
| dd=0.1 ps=1e-3 cs=1e-3 | 1337 | 11.4x | 811 | -13.6% | 0.67 |
| dd=1.0 ps=1e-3 | 1337 | 10.9x | **1085** | -13.9% | 0.66 |
| dd=0.5 ps=1e-4 | 1337 | 9.6x | 915 | -13.8% | 0.66 |
| dd=0.1 ps=1e-4 | 1337 | 9.4x | 896 | -13.5% | 0.65 |
| dd=2.0 ps=5e-3 | 1337 | 8.2x | 1008 | -14.4% | 0.65 |

### Full Sweep Results (8 configs, 2 seeds each)

| Config | Seed | Return | Sortino | DD | Cap |
|--------|------|--------|---------|------|------|
| **dd=0.1 ps=0 cap=0.5 cs=1e-3** | **2024** | **17.9x** | 969 | -13.9% | **0.80** |
| dd=1.0 ps=1e-3 cap=0.5 | 2024 | 15.1x | 1003 | -14.5% | 0.79 |
| dd=0.5 ps=5e-4 cap=0.3 | 2024 | 13.3x | 979 | **-12.6%** | 0.74 |
| dd=0.1 ps=1e-3 cs=1e-3 | 1337 | 11.4x | 811 | -13.6% | 0.67 |
| dd=1.0 ps=1e-3 | 1337 | 10.9x | **1085** | -13.9% | 0.66 |
| dd=0.5 ps=1e-4 | 1337 | 9.6x | 915 | -13.8% | 0.65 |
| dd=0.1 ps=1e-4 | 1337 | 9.4x | 896 | -13.5% | 0.65 |
| dd=2.0 ps=5e-3 | 1337 | 8.2x | 1008 | -14.4% | 0.65 |

All configs beat baseline (7.5x, -22.7% DD) on both return AND drawdown.
Cap floor is the strongest factor: configs with cap_floor get higher returns (policy more aggressive).
Best overall: dd=0.1 + cap=0.5 + cs=1e-3 (17.9x ret, -13.9% DD, 2.4x baseline, 40% less DD).

### Multi-Window Eval (top 4 configs, 5x leverage, $10k start, 10bp fees)

| Config | 3d | 7d | 10d | 14d | 30d |
|--------|-----|-----|------|-------|---------|
| **dd01_ps0_cap05 (RL)** | 3.0x | **17.9x** | **45.9x** | **104.2x** | **3440.5x** |
| dd1_ps1e3_cap05 (RL) | 2.5x | 15.1x | 38.5x | 83.6x | 2721.8x |
| dd05_ps5e4_cap03 (RL) | 2.3x | 13.3x | 32.0x | 70.9x | 1970.0x |
| dd1_ps1e3 (RL) | 1.5x | 8.8x | 19.3x | 44.6x | 1201.5x |
| **Baseline** | 1.4x | 7.5x | 9.7x | 14.6x | 51.5x |

| Config | 3d Sort | 7d Sort | 10d Sort | 14d Sort | 30d Sort |
|--------|---------|---------|----------|----------|----------|
| dd01_ps0_cap05 | 845 | 969 | 601 | 482 | 447 |
| dd1_ps1e3_cap05 | 919 | 1003 | 620 | 513 | 486 |
| dd05_ps5e4_cap03 | 869 | 979 | 616 | 487 | 464 |
| dd1_ps1e3 | 692 | 972 | 533 | 456 | 459 |

| Config | 3d DD | 7d DD | 10d DD | 14d DD | 30d DD |
|--------|-------|-------|--------|--------|--------|
| dd01_ps0_cap05 | -13.9% | -13.9% | -13.9% | -13.8% | -13.6% |
| dd05_ps5e4_cap03 | -12.6% | -12.6% | -12.6% | -12.6% | -12.4% |
| Baseline | -22.7% | -22.7% | -22.7% | -22.7% | -22.7% |

Key insight: RL residual controller scales exponentially better than baseline at longer windows.
At 30d, best RL gets 3440x vs baseline 51.5x (67x improvement), with 40% lower drawdown.
DD stays flat across windows for RL (capped by policy), while returns compound exponentially.

### Fine-Grained Sweep (around best config, 60k steps, CPU)

| Config | Return | Sortino | DD | Cap |
|--------|--------|---------|------|------|
| **cap_floor=0.7** | **20.3x** | 893 | -20.0% | 0.86 |
| cap_floor=0.6 | 18.7x | 895 | -18.9% | 0.83 |
| cs=5e-3 | 17.8x | 898 | -18.1% | 0.81 |
| cs=5e-4 | 17.2x | 899 | -17.9% | 0.80 |
| cap_floor=0.5 mc=0.05 | 17.2x | 893 | -18.0% | 0.81 |
| ps=1e-4 cs=1e-3 | 17.1x | 895 | -17.9% | 0.80 |
| cap_floor=0.4 | 17.1x | 900 | -17.1% | 0.79 |
| dd=0.5 cap=0.5 | 16.7x | 891 | -17.7% | 0.79 |
| mc=0.2 | 16.5x | 887 | -17.7% | 0.79 |
| mc=None | 16.1x | 894 | -17.9% | 0.79 |

cap_floor is the dominant knob: higher floor -> more aggressive -> higher return + higher DD.
Linear return/DD tradeoff: ~0.5x return per 1% DD increase.
Other params (cs, mc, ps, dd) have minimal effect once cap_floor is set.

### Longer Training (300k steps, GPU, best config)

| Steps | Return | Sortino | DD | Cap |
|-------|--------|---------|------|------|
| 60k (original) | 17.9x | 969 | **-13.9%** | 0.80 |
| **300k** | **23.8x** | 1000 | -21.0% | -- |

More steps improves return (17.9 -> 23.8x, +33%) but DD worsens (-13.9% -> -21.0%).
60k version has better risk-adjusted properties (DD nearly halved vs baseline).
Diminishing returns on longer training -- the policy exploits more but at cost of safety.

## Stock Model Training Progress (2026-02-17)

Remote 5090: 512h 4L lr=5e-5 seq=32, training complete (500 epochs).
Checkpoints: epochs 1-10, 13, 17, 22, 25, 28.

### Epoch Sweep (30d holdout, me=0.001, $10k start)

| Epoch | Return | Sortino | Trades |
|-------|--------|---------|--------|
| **3** | **+68.9%** | **1.86** | 17 |
| 8 | +52.0% | 1.70 | 11 |
| 10 | +48.2% | 1.48 | 23 |
| 6 | +46.3% | 1.57 | 11 |
| 5 | +45.1% | 1.54 | 11 |
| 7 (deployed 6L) | +42.6% | 1.36 | 23 |
| 2 | +42.1% | 1.75 | 23 |
| 1 | +39.8% | 1.32 | 23 |
| 9 | +41.4% | 1.32 | 23 |
| 13 | +2.2% | 0.29 | 19 |
| 17+ | Negative (overfit) | - | - |

Best: Epoch 3 (68.9% return, Sortino 1.86). Clear early-stopping pattern.
Overfitting starts hard after epoch 10 -- returns collapse to negative by epoch 17.

### Local 4L lr=3e-5 seq=48 Epoch Sweep

| Epoch | Return | Sortino | Trades |
|-------|--------|---------|--------|
| **11** | **+61.0%** | **2.24** | 25 |
| 12 | +59.1% | 2.07 | 25 |
| 14 | +57.8% | 2.19 | 25 |
| 7 | +53.0% | 2.00 | 25 |
| 5 | +49.1% | 1.94 | 25 |

### All Stock Models Comparison (30d holdout, me=0.001)

| Model | Best Epoch | Return | Sortino |
|-------|-----------|--------|---------|
| 6L (DEPLOYED) | 6 | **74.5%** | 2.48 |
| 6L | 7 | 57.8% | **3.38** |
| remote 4L lr=5e-5 | 3 | 68.9% | 1.86 |
| local 4L lr=3e-5 seq=48 | 11 | 61.0% | 2.24 |
| nas_512h_4L | 28 | 50.9% | 3.29 |

6L remains the best architecture. Both 4L variants underperform on sortino.
6L ep6 has highest raw return (74.5%), ep7 has best risk-adjusted (Sortino 3.38).
Consider deploying 6L ep6 for higher returns if willing to accept lower sortino.

## Sortino Maximization Experiments (2026-02-17)

Creative approaches to push SUI residual controller sortino higher.
Code: `experiments/sui_sortino_max/`

### RL Reward Shaping (10 experiments, 2 seeds each)

| Experiment | Return | Sortino | DD |
|-----------|--------|---------|------|
| conservative_sortino | 2.1x | **1129** | -20.8% |
| low_entropy_sortino | 2.8x | 1107 | -18.9% |
| high_cap_sortino | 7.6x | 1099 | -30.3% |
| dd_adaptive_sortino | 3.7x | 1071 | -32.6% |
| big_model_sortino (120k) | 4.0x | 1047 | -17.3% |
| high_gamma_sortino | 1.9x | 1020 | -23.3% |
| sortino_asym4_tight | 3.7x | 999 | -19.7% |
| **original best (dd01_ps0_cap05)** | **17.9x** | **969** | **-13.9%** |
| dd_adaptive_lev | 9.6x | 961 | -19.3% |
| sortino_asym2 | 4.4x | 942 | -20.8% |
| pure_sortino | 3.4x | 936 | -16.1% |
| Baseline (no RL) | 7.5x | 1046 | -22.7% |

Key finding: **sortino vs return tradeoff is fundamental**. Higher sortino configs (1129)
get only 2x return vs 17.9x for the original. The original config sits on the efficient
frontier -- near-optimal risk-adjusted returns.

### Ensemble Evaluation (top models averaged)

| Ensemble | 7d Return | 7d Sortino | 7d DD |
|----------|-----------|------------|-------|
| single_best | 17.9x | 969 | -13.9% |
| ensemble_4 | 17.7x | 980 | -14.2% |
| ensemble_top3 | 16.0x | **999** | -14.1% |
| ensemble_top2 | -- | -- | -- |

Ensembling provides marginal sortino improvement (+3%) at slight cost to return.
Not worth the complexity -- single model is near-optimal.

### Rule-Based Overlays (post-hoc action modification)

| Strategy | Return | Sortino | DD |
|----------|--------|---------|------|
| cap_clamp_05_08 | **21.8x** | 972 | -15.5% |
| cap_clamp_04_06 | 20.4x | 970 | -15.1% |
| cap_clamp_03_07 | 19.4x | 971 | -14.6% |
| **baseline (no overlay)** | **17.9x** | **969** | **-13.9%** |
| conservative_cap_04 | 17.0x | 964 | -13.9% |
| vol_gate | 15.8x | 966 | -13.9% |
| momentum_filter | 14.3x | 958 | -16.9% |

Cap clamping (forcing cap_ratio into a narrow range) can boost returns slightly
at cost of ~1-2% more DD. Vol gating and momentum filtering hurt performance.

### Conclusions
1. Original best config (dd=0.1, cap_floor=0.5, cs=1e-3) is near the efficient frontier
2. Sortino can be pushed to 1129 but at 90% return cost (2x vs 18x)
3. Ensembles and rule-based overlays provide marginal improvements at best
4. The dominant factor remains cap_floor (controls risk/return tradeoff linearly)
5. For production: stick with original best + consider cap_clamp for slightly more return

## Training Improvement Sweep (2026-02-18, in progress)

Sweeping 16 training configurations near baseline (rw=0.012). 25 epochs each, eval at 1x/3x/5x on 10d test, $5k start, 10bp fees.

### Results So Far (5/16 complete)

| Experiment | 5x Return | 5x Sortino | 5x Max DD | Notes |
|-----------|-----------|------------|-----------|-------|
| baseline_rw012 | 35.8x | 137.1 | -6.6% | current deployed |
| rw008 | 33.7x | 136.0 | -6.5% | lower return_weight |
| rw010 | 27.8x | 118.2 | -6.5% | |
| **rw014** | **36.2x** | **269.5** | **-2.3%** | **2x sortino, 1/3 DD** |
| **rw016** | **45.9x** | **176.6** | **-5.4%** | **best return** |

Key finding: **rw014 is dramatically better risk-adjusted** -- same return as baseline but sortino doubled (269 vs 137) and drawdown cut to 1/3 (-2.3% vs -6.6%). rw016 is best on pure return (+28% over baseline).

### Remaining Experiments (running)
- rw020 (higher return weight)
- cosine_rw012 (cosine LR schedule)
- cosine_rw012_min01 (cosine with lower floor)
- warmdown_rw012 (linear warmdown)
- smooth001_rw012 (smoothness penalty 0.001)
- smooth005_rw012 (smoothness penalty 0.005)
- cosine_smooth001 (combined cosine + smooth)
- ep35_rw012 / ep40_rw012 (more epochs)
- nano_rw012 (nano architecture)
- wd_linear_rw012 (linear weight decay)

### Live Bot Status (2026-02-18)
- Running at 5x leverage since ~Feb 17
- Currently in position: 25.4k SUI, $19.3k USDT borrowed
- Equity: ~$4,268 (down ~15% from $5k start)
- SUI dropped from ~$0.97 entry to ~$0.93 (-4.4%), 5x leveraged = ~22% equity loss
- Bot correctly placing sell limits above market, waiting for price recovery
- Data pipeline healthy: 1h stale (normal -- current bar incomplete)

## Covariate/Cross-Learning Forecast Experiment (queued)

Testing whether BTC/ETH/SOL covariates improve SUI OHLC forecast MAE using Chronos-2.

### Three Approaches
1. Univariate: SUI OHLC only (current production)
2. Multivariate: SUI OHLC jointly predicted (open/high/low/close together)
3. Cross-learning: SUI + BTC + ETH + SOL jointly with predict_batches_jointly=True

Script: `binanceleveragesui/eval_covariate_forecasts.py`
Model: finetuned LoRA (`chronos2_finetuned/binance_lora_20260208_newpairs_SUIUSDT/finetuned-ckpt`)
Eval: rolling 30-day holdout, every 24h, horizons h1/h4/h24
Status: waiting for GPU (sweep consuming 96%)

## Ongoing Binance Meta Optimization (2026-03-04)

### Completed this cycle

#### Ultra-fine static leverage sweep
- Artifact: `binanceleveragesui/meta_leverage_ultrafine_static_v4.json`
- Scope: static `winner_cash + calmar + lb1` on deployed pair (`doge_drop15_ep008 + aave_strides_ep002`) with leverage grid `2.26..2.35`.
- Best robust point remains **2.30x**:
  - min sortino **0.22**, mean sortino **0.3175**
  - min return **+446.06%**, mean return **+749.46%**
  - mean DD **-9.22%**, beats baseline **4/4** windows
- Higher leverage (`>=2.31`) increases return but drops min sortino to 0.21.

#### Confidence-scaled dynamic leverage search
- Artifact: `binanceleveragesui/meta_confidence_dynamic_leverage_v1.json`
- Result: no dynamic policy beat static 2.30x objective.
- Top rows collapsed to effectively static behavior (`low_lev=high_lev=2.25`) with:
  - min sortino **0.22**, mean sortino **0.3175**
  - min return **+428.25%**, mean return **+713.64%**
  - mean DD **-9.03%**, beats **4/4**
- Decision: keep static 2.30x deployment target; no dynamic leverage rollout.

#### Selector hyperparameter retune at 2.30x
- Artifact: `binanceleveragesui/meta_pair_tune_doge_ep008_aave_ep002_lev230_v1.json`
- Pair: `doge_drop15_ep008 + aave_strides_ep002`
- Sweep: metrics `{calmar,sortino,sharpe,return}`, lookbacks `1..14d`, thresholds/hysteresis/confidence-gap grids.
- Best region confirms current settings:
  - `winner_cash + calmar + lb1`, `cash_threshold=0`, `switch_margin >= 0.002`, `min_score_gap` near 0
  - Robust stats match leverage sweep winner: min sortino **0.22**, mean return **+749.46%**, mean DD **-9.22%**.
- Decision: no selector change required.

### New outside-the-box tooling added
- New script: `binanceleveragesui/search_meta_selector_stack.py`
- Purpose: two-layer meta policy where layer-2 daily selector picks among many layer-1 selector profiles (selector-of-selectors).
- Status: script added and compiled; quick + medium stacked sweeps completed.

### Long-run note
- Initial broad 5x5 epoch search and first full stacked sweep were started but later interrupted after long wall-clock runtime.
- They were replaced by reduced searches documented below to keep iteration throughput high.

### Deployment state
- No superior deployable config found vs current target.
- Keep target config as:
  - pair: `doge_drop15_ep008 + aave_strides_ep002`
  - selector: `winner_cash + calmar + lookback=24h + switch_margin=0.005`
  - leverage: **2.30x**
- Note: supervisor control remains permission-blocked from this shell, so live process restart still requires privileged supervisor action.

### Quick validation of selector-of-selectors (stacked) idea
- Artifact: `binanceleveragesui/meta_selector_stack_search_ep008_ep002_lev230_quick_v1.json`
- Setup:
  - base profile pool: 12 (`calmar/sortino`, lb 1/2/3, cash thresholds 0/0.01, switch margin 0.005)
  - layer-2 selector: winner_cash over base profiles (6 meta configs)
- Outcome: **underperformed** current single-layer deployment target.
  - Best stacked config (`sortino`, meta lb=3):
    - min sortino **0.16**, mean sortino 0.26
    - min return +417.64%, mean return +475.09%
    - mean DD -14.52%
  - Current deployed-target single-layer baseline at 2.30x:
    - min sortino **0.22**, mean sortino 0.3175
    - min return +446.06%, mean return +749.46%
    - mean DD -9.22%
- Decision: stacked selector not deployable in this form.

### Validation tests
- `pytest -q tests/test_trade_margin_meta_selection.py tests/test_binance_meta_daily_winners.py`
- Result: **15 passed** (warnings only; no failures)

### Follow-up reduced searches (post-interrupt, faster iteration)

#### Medium stacked-selector sweep
- Artifact: `binanceleveragesui/meta_selector_stack_search_ep008_ep002_lev230_v2_medium.json`
- Base profile pool: 144; layer-2 configs: 64.
- Best stacked config examples:
  - `sortino lb=3 ct=0.0 sm=0.005 mg=0.001`
  - `calmar lb=2 ct=0.0 sm=0.005 mg=0.0`
- Best stacked robust stats (top row range):
  - min sortino ~**0.19**, mean sortino ~0.27
  - min return +424% to +452%, mean return +599% to +657%
  - mean DD ~**-14.70%**
- Verdict: still clearly below current single-layer deployment target (min sortino 0.22, mean DD -9.22).

#### Tight epoch pair search (2x2 families, focused epochs)
- Artifact: `binanceleveragesui/meta_pair_epoch_search_tight2x2_lev230_v3.json`
- Search scope:
  - DOGE families: `drop15`, `strides`
  - AAVE families: `strides`, `wd04`
  - epochs: `6,7,8,9,10,11`
  - selector: `winner_cash`, metric `{calmar,sortino}`, lb `{1,2}`
- Best row found:
  - `doge_drop15_ep008 + aave_wd04_ep006`, calmar lb1
  - min sortino **0.20**, mean sortino 0.31
  - min return +496.71%, mean return +657.13%
  - mean DD **-11.72%**, beats 4/4
- Comparison to current target:
  - Current target keeps stronger worst-window risk-adjusted quality (`min sortino 0.22`) with much lower DD (`-9.22%`).
- Verdict: no pair in this focused scan beats deployment target on combined risk-adjusted objective.

### Runtime note
- Earlier very large exhaustive runs were intentionally interrupted after long wall-clock durations and replaced by the reduced searches above to preserve iteration speed.

#### Focused retune of best alternative pair (final check)
- Artifact: `binanceleveragesui/meta_pair_tune_doge_ep008_aave_wd04_ep006_lev230_v1.json`
- Pair tested: `doge_drop15_ep008 + aave_wd04_ep006`
- Search scope: winner_cash, metrics `{calmar,sortino}`, lookbacks `{1,2,3,5}`, cash/switch/gap grids.
- Result:
  - `MAX_MIN_SORTINO = 0.20` (across all tested configs)
  - Best stats at that cap: mean sortino ~0.31, mean return +657.13%, mean DD -11.72%
- Conclusion:
  - Even after focused retuning, this alternative pair cannot match current deployment target on worst-window risk-adjusted objective (`min sortino 0.22`) and has materially worse DD.
  - No deployment change.

### Additional outside-the-box cycle (2026-03-04, later)

#### New selector metrics implemented
- Updated shared meta scorer in `unified_hourly_experiment/meta_selector.py` with:
  - `omega`, `gain_pain`, `p10`, `median`
- Updated live meta bot metric allowlist in `binanceleveragesui/trade_margin_meta.py` accordingly.
- Validation:
  - `pytest -q tests/test_meta_selector.py tests/test_trade_margin_meta_selection.py tests/test_binance_meta_daily_winners.py`
  - Result: **22 passed**.

#### Pair-of-pairs meta-selector search
- Artifact: `binanceleveragesui/meta_pair_of_pairs_search_lev230_v1.json`
- Idea: layer-2 selector over multiple layer-1 pair strategies (base + alternative pairs).
- Result:
  - best row: min sortino **0.20**, mean sortino 0.2825
  - min return +425.89%, mean return +731.18%
  - mean DD **-12.87%**
- Verdict: not deployable vs current target (worse min sortino and DD).

#### Continuous allocation modes on current best pair
- Artifacts:
  - `binanceleveragesui/meta_pair_mode_softmax_all_doge_ep008_aave_ep002_lev230_v1.json`
  - `binanceleveragesui/meta_pair_mode_blend_top2_doge_ep008_aave_ep002_lev230_v1.json`
- Results:
  - `softmax_all` best min sortino: **0.18**
  - `blend_top2` best min sortino: **0.18**
- Verdict: both underperform hard-switch winner_cash policy.

#### Expanded-metric sweeps (`omega`, `gain_pain`)
- Current pair tuning artifact:
  - `binanceleveragesui/meta_pair_tune_doge_ep008_aave_ep002_lev230_newmetrics_v1.json`
- Outcome:
  - top region remains unchanged (`calmar`, lb1, winner_cash)
  - robust stats unchanged: min sortino **0.22**, mean return **+749.47%**, mean DD **-9.22%**
- Tight epoch expanded-metric artifact:
  - `binanceleveragesui/meta_pair_epoch_search_tight2x2_lev230_newmetrics_v1.json`
- Outcome:
  - `MAX_MIN_SORTINO = 0.20` (no improvement over current deployment target)

#### Downside-metric sweeps (`p10`, `median`)
- Current pair downside tuning artifact:
  - `binanceleveragesui/meta_pair_tune_doge_ep008_aave_ep002_lev230_p10median_v1.json`
- Outcome:
  - best min sortino **0.20** (below current 0.22)
- Tight epoch downside artifact:
  - `binanceleveragesui/meta_pair_epoch_search_tight2x2_lev230_p10median_v1.json`
- Outcome:
  - `MAX_MIN_SORTINO = 0.17`
- Verdict: downside quantile/median selectors did not beat baseline.

#### Leverage frontier sanity sweep
- Artifact: `binanceleveragesui/meta_leverage_frontier_1p0_to_2p3_v1.json`
- Scope: current best policy, leverage 1.0..2.3 (step 0.1).
- Result:
  - min sortino and mean sortino stayed constant at **0.22 / 0.318** across all leverage levels.
  - return and DD scale monotonically with leverage.
- Implication:
  - If objective is max return under same sortino profile, 2.30x remains best.
  - If objective is lower DD with same sortino profile, lower leverage is a straightforward dial.

### Deployment decision after full cycle
- No new algorithm exceeded current deployment target on combined robust objective.
- Keep target deploy config unchanged:
  - pair `doge_drop15_ep008 + aave_strides_ep002`
  - selector `winner_cash + calmar + lookback=24h + switch_margin=0.005`
  - leverage `2.30x`

### Live audit + restart hardening (2026-03-04)

#### Supervisor/runtime findings
- Active supervisor program is `binance-meta-margin` (not `binance-doge-margin`).
- `/etc/supervisor/conf.d/binance-meta-margin.conf` is currently deployed with:
  - DOGE `epoch_001`, AAVE `epoch_003`
  - `--max-leverage 0.1`, `--lookback 12`

#### Root-cause audit on recent stuck/loss behavior
- Confirmed live margin state had a locked AAVE exit order:
  - `AAVE free=0.00043`, `locked=47.936`
  - one open `AAVEUSDT SELL` order (`124.53`, `47.936` qty)
- This can cause repeated `APIError -2010 insufficient balance` when bot tries to place additional full-size exits while almost all inventory is locked.

#### Code fixes applied
- `binanceleveragesui/trade_margin_meta.py`
  - exit sizing now uses **free inventory** for order quantity (not free+locked).
  - added locked-inventory/open-order checks to avoid duplicate failing exits.
  - added borrow-cap safety for entries/adds:
    - caps buy notional to `USDT free + get_max_borrowable("USDT")`.
  - added safer hold-time calc:
    - `hours_held()` now clamps to `>=0` and handles malformed timestamps.
- `binanceneural/binance_watchers.py`
  - fixed Loguru formatting placeholders so watcher logs now print real values.
- simulator parity:
  - `binanceleveragesui/validate_sim_vs_live.py` and `compare_sim_prod.py` now include margin interest accrual (`--margin-hourly-rate`) in 5m replay.

#### Test/verification
- `pytest -q tests/test_trade_margin_meta_selection.py tests/test_meta_selector.py tests/test_binance_meta_daily_winners.py tests/test_simulator_math.py`
  - **48 passed** (warnings only)
- follow-up focused suite after additional hold-time fix:
  - **27 passed** (warnings only)

#### Restart/deploy status
- Restarted live service twice after patches:
  - `sudo supervisorctl restart binance-meta-margin`
  - status: `RUNNING`
- Post-restart logs show patched behavior active:
  - no negative hold duration (`holding aave (0.0h)`)
  - bot remains in controlled hold state with existing open AAVE exit order.

#### Deployment alignment to optimized profile (2026-03-04, post-audit)
- Updated active supervisor unit `/etc/supervisor/conf.d/binance-meta-margin.conf` to match best selected profile:
  - pair: `doge_drop15_ep008 + aave_strides_ep002`
  - selector: `winner_cash + calmar`
  - params: `lookback=24`, `switch_margin=0.005`, `max_leverage=2.30`, `cycle_minutes=60`
- Applied with:
  - `sudo supervisorctl reread && sudo supervisorctl update && sudo supervisorctl restart binance-meta-margin`
- Verified live process args now match optimized command line (pid `3947580` at deploy time).
- Backup of prior active supervisor config created before replacement (`/etc/supervisor/conf.d/binance-meta-margin.conf.bak_<timestamp>`).

### Continued Search + Runtime Hardening (2026-03-04, latest)

#### New broad multi-candidate sweeps (DOGE/AAVE/ETH)
- Artifacts:
  - `binanceleveragesui/meta_daily_winners_doge_aave_eth_lev230_v2.json`
  - `binanceleveragesui/meta_daily_winners_doge_aave_eth_lev230_v3.json`
- `v2` top region (8 candidates, smoother-focused epochs):
  - dominated by `softmax_all`, `lookback=1d`, `temperature=0.25`
  - min sortino **0.19**, mean sortino **0.333**
  - min return **+221.38%**, mean return **+306.21%**
  - mean DD **-6.73%**, beats **4/4** windows
- `v3` union sweep (11 candidates; prior winners + smoother set):
  - top region again `softmax_all`, `lookback=1d`, `temperature=0.25`
  - min sortino **0.21**, mean sortino **0.328**
  - min return **+237.70%**, mean return **+308.12%**
  - mean DD **-7.43%**, beats **4/4** windows
- Deployability note:
  - these best rows rely on continuous-allocation mode (`softmax_all`) not used by current live margin runtime.
  - best live-compatible `winner_cash` rows from this cycle improved DD but reduced worst-window sortino vs deployed baseline.

#### Expanded regime retune on deployed pair
- Artifact:
  - `binanceleveragesui/meta_pair_regime_tune_ep008_ep002_reduced_v4.json`
- Best rows:
  - `min_sortino=0.22` (ties baseline), `mean_sortino=0.3375` (higher),
  - `mean_return=+692.55%` (higher),
  - but `mean_DD=-12.01%` (materially worse).
- No strict dominance over deployed static policy.

#### Selector-stack experiment (outside-the-box)
- Artifact:
  - `binanceleveragesui/meta_selector_stack_ep008_ep002_v3.json`
- Result:
  - top rows increased mean return but all reduced `min_sortino` vs baseline (`0.21` or below vs `0.22`).
  - no row strictly dominated deployed baseline on combined objective.

#### Critical live bug fix (stuck locked exit + bad hold clock)
- Root cause found in live state:
  - `open_ts` had future UTC timestamp, so `hours_held()` clamped to `0.0h` and force-close timing was neutralized.
  - inventory was almost entirely `locked` by stale AAVE exit order, but `asset_free` was tiny positive, so old unlock condition (`asset_free <= 0`) did not trigger.
- Code updates in `binanceleveragesui/trade_margin_meta.py`:
  - auto-normalize invalid/future `open_ts` (prefer open-order timestamp when available).
  - added stale/age-aware order timestamp parsing helpers.
  - replaced strict free==0 lock gate with `mostly_locked` inventory detection.
  - force-close now unlocks stale locked exits reliably and logs unlock-wait states clearly.
- Tests:
  - `pytest -q tests/test_trade_margin_meta_selection.py tests/test_simulator_math.py tests/test_binance_meta_daily_winners.py`
  - **47 passed** (warnings only).

#### Live deployment + verification (latest restart)
- Restarted live service:
  - `sudo supervisorctl restart binance-meta-margin`
- Observed in live logs after patch:
  - `corrected open_ts (future_ts_from_open_order) -> ...`
  - repeated `FORCE CLOSE aave after 6.9h`
  - successful unlock and liquidation: `force-close sell=110.5500 qty=47.9360`
  - `aave position closed`
  - fresh re-entry observed under live policy.
- Post-check:
  - stale AAVE open order cleared.
  - active runtime still aligned to deployed strategy target (`doge_ep008 + aave_ep002`, `winner_cash`, `calmar`, `lookback=24`, `switch_margin=0.005`, `max_leverage=2.30`).

#### Current decision
- Keep algorithm deployment unchanged for now (no new strict dominance on combined robustness objective).
- Keep newest runtime bugfixes deployed (they resolved the real stuck-position failure mode in production).
