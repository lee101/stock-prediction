# 5-Minute Stock RL Challenge — Failed Experiments & Hard Lessons (Part 5)

This is a systematic record of what does not work, organized by category. It is the companion to stockprogress.md and the series of `alpacaprogress*.md` and `binanceprogress*.md` files. Read this before starting a new sweep or re-attempting a dead end. Dates are given where known.

---

## 1. The Big Lessons

**Lesson 1 — Look-ahead bias is catastrophic (2026-03-12)**
The C trading env had a 1-bar observation lag bug: the agent could see the CLOSE price it was about to execute into. "Record" returns (crypto12 2,658x per 30d) were inflated by this. The fix: strict 1-bar obs lag + OPEN price execution. Always verify temporal causality before trusting in-sample numbers. Every new env change must be audited for future-data leakage.

**Lesson 2 — More training steps = more overfitting OOS**
200M-step in-sample returns are spectacular. 200M-step OOS returns on held-out data are -3% to -16% per 30d. The 5-minute timeboxed (~8M steps) models generalize; the long-trained ones do not. Track OOS from the very first checkpoint. The autoresearch methodology (short training + diverse configs + OOS eval) is the correct workflow. Do not attempt to close the gap by training longer.

**Lesson 3 — Long-only fails in sustained downtrends**
The deployed long-only crypto hourly policy lost -13% to -38% when crypto crashed 30-45% (Feb-Mar 2026). A policy that has never learned to be flat or short during a bear market will ride losses all the way down. Any long-only strategy must be paired with a hard exit rule or have its deployment paused when the index is in a sustained drawdown.

**Lesson 4 — Training friction (slippage) produces better OOS generalisation**
Adding 5bps fill slippage during RL training forces the agent to find wider edges. The slip_5bps config is the #1 hourly OOS result (+5.3%/30d, 96% profitable). Zero-slippage training produces agents that exploit tiny in-sample patterns that vanish in live execution. Always train with at least 5bps slippage for hourly, and 8-12bps for daily stock RL.

**Lesson 5 — Joint multivariate forecasting never beats per-symbol independent**
1,079 experiments over multiple weeks conclusively showed that cross-symbol joint Chronos2 forecasting does not outperform training each symbol's forecast independently. The `ForecastConfig.crypto_multi_ohlc.use_cross_learning=False` commit locked this in. Never re-open joint forecasting without a new and compelling theoretical reason.

---

## 2. Failed Approaches by Category

### A. Reward Shaping Failures

**Drawdown penalty**
- Tested: `--drawdown-penalty` > 0 combined with standard PPO+sortino loss
- Result: 19.9x vs 42.75x baseline (over 2x degradation)
- Root cause: The penalty creates conflicting gradient signals. The agent learns to avoid all positions rather than avoid bad ones. Drawdown is an emergent property of good position-level decisions; punishing it directly corrupts the credit assignment.
- Status: Do not use `--drawdown-penalty` with sortino loss.

**Smoothness/downside penalties without tuning temperature**
- Tested: `smooth_downside_penalty` with default `smooth_downside_temperature=0.02`
- Result in stock autoresearch: the bottom-ranked runs (holdout_robust_score -118 to -160) all used combinations of downside penalty + small hidden size (h256). The penalty terms interact badly with the capacity-limited policy.
- The winning stock config (`random_mut_2272`) used `smooth_downside_temperature=0.01` specifically. The value is sensitive; the wrong temperature makes the penalty counterproductive.

**High entropy coefficient (> 0.1)**
- Result: unstable training, policy collapses or oscillates
- 0.05 is the validated sweet spot for crypto hourly RL
- For daily stock RL the winning entropy coef was 0.03 (lower exploitation needed with cleaner daily signals)

**Low entropy coefficient (0.01)**
- Result: premature convergence, policy stops exploring, OOS fails
- The agent locks into a narrow strategy that worked on training data and never generalises

**Symmetric reward clipping without trade penalty**
- The ret*10 + clip[-5,5] formulation without any trade penalty causes over-trading in daily RL
- trade_penalty=0.05 is needed to impose trade discipline on daily timeframes

**Gamma = 0.999 with ResidualMLP**
- Tested combo: ResidualMLP architecture + gamma=0.999
- Result: only 9.9x return (vs ~140x baseline at same step count)
- Root cause: high discount factor kills credit assignment in daily trading. The agent can not attribute a today's entry to tomorrow's exit. gamma=0.99 is the validated value.

### B. Architecture Failures

**h4096 and h2048**
- Under-converges relative to h1024 at the same 50M-step budget (too many parameters for the available data)
- h4096: 58.9x vs h1024: 68x at 50M steps
- The additional capacity overfits noise before the useful signal is learned

**h256 in stock autoresearch**
- The bottom quartile of the 80-trial stock autoresearch sweep is dominated by h256 runs
- h256 hits a capacity ceiling in the first 2 minutes of training; the remaining 3 minutes adds noise not signal
- For stocks (12 symbols, daily, multi-window robust eval) h256 is definitively too small

**Depth recurrence without matching step budget**
- Adding LSTM / GRU layers requires more steps to converge; at the 5-minute timebox they under-converge relative to MLP

**Cross-symbol transformer attention**
- Joint multivariate Chronos2: 1,079 experiments, never beats per-symbol. See Lesson 5.

### C. Leverage Failures

**2x leverage in RL**
- Only 48% of evaluation episodes are profitable vs 100% at 1x leverage
- The compounding variance at 2x swamps any return uplift
- The same observation holds in crypto backtests: 1x leverage consistently Pareto-dominates 2x on Sortino

**Long+short stocks during a bull market val period**
- Long+short stock RL yielded -41% annualized OOS on the Jun-Dec 2025 val set
- The val period was a sustained bull market; the short leg was simply wrong for that regime
- Long-only is strictly better for stock RL in non-bear conditions

**Leverage > 1x with high fee rates**
- At 10bps fees (USDT pairs), leverage multiplies both returns and fee drag
- At 2x leverage the annual fee drag doubles; the edge rarely survives

### D. Position Sizing Failures

**Kelly criterion**
- Tested as an overlay on RL signals
- Result: worse than end-to-end neural policy sizing
- Root cause: Kelly requires an accurate estimate of win probability and edge size. The RL policy's implicit edge estimate is already baked into the action probabilities; imposing an external formula over-leverages the confident wrong predictions.

**Fixed uniform position sizes**
- Can not adapt to conviction level; misses the implicit signal in the policy's action probability distribution

**Over-sizing individual positions (> 25% of account per symbol)**
- Correlated drawdowns across crypto symbols wipe account equity when the whole market moves against the position
- Validated cap: 25% per position (--max-position-fraction 0.25)

### E. Training Infrastructure Failures

**Retraining "verified" checkpoints and expecting the same results**
- RL training is non-deterministic even with the same seed and same hyperparameters (CUDA scheduling variance)
- The trade_pen_05 daily crypto model: seed=42 gave +20.0% OOS; seeds 123, 2024, 7 gave +1.4%, -10.2%, -18.6% respectively
- Deploy only verified, tested checkpoints. Never retrain and assume the result will match.

**Training without OOS checkpointing from the start**
- If you only checkpoint at the end of a long run, you cannot recover the early generalising weights
- The overfitting detection must happen at every checkpoint, not retroactively

**Missing checkpoint resume on pod termination**
- Long runs lost on cloud pod preemption when no periodic checkpointing was set
- Always use `--checkpoint-interval` with a short enough period to survive pod termination

**Flat-only evaluation (single starting state)**
- The worksteal sweep revealed that a config that looks profitable from flat-cash start can be deeply unprofitable if the starting inventory is BTC or ETH
- The "old flat-only best" worksteal config: worst return -1.41% from seeded starts vs the robust best's worst return +3.06%
- Always evaluate across multiple starting states (flat, long BTC, long ETH at minimum)

**Single holdout window evaluation**
- High variance, not representative of strategy robustness
- The confidence threshold conf_0.7 looked brilliant on Mar 2-9 (+357%) and was catastrophically bad on Mar 7-14 (-40.7%)
- Minimum: 20 rolling holdout windows before trusting any parameter change

### F. Symbol Selection Failures

**NYT (New York Times Co.) in the live trading set**
- Model classified NYT as SHORT; NYT rallied +51.6%
- NYT is a special-situation momentum stock in 2025 (private-equity rumours, earnings surprises); the historical Chronos2 forecasts cannot capture this
- Rule: NYT must be EXCLUDED from live trading even though it is useful as a training symbol (removing it from training degrades other symbol predictions)

**TSLA in the symbol set**
- All OOS periods negative; worst 150d at -10.8%
- TSLA has extremely high idiosyncratic volatility tied to Elon Musk news events; the model cannot forecast these
- Rule: Never add TSLA to the live trading universe

**BNB on Binance**
- Only 49 hourly bars available at the time of the FDUSD sweep
- 49 bars is not enough for any RL policy to learn a meaningful signal
- Rule: minimum 500 bars required before including a symbol

**Highly correlated symbol pairs**
- Including too many correlated symbols (e.g., BTC+ETH+SOL all together with no uncorrelated assets) concentrates risk without diversifying signal
- When BTC dumps 10%, ETH and SOL dump 8-12% simultaneously; the portfolio takes triple the loss

**Too few symbols (crypto3/crypto5)**
- Insufficient opportunity set: the agent goes flat more than it should because no signal meets its threshold
- crypto12 > crypto8 > crypto5 in both in-sample and OOS returns
- For stocks: 12 symbols (AAPL, MSFT, NVDA, GOOG, META, TSLA, AMZN, JPM, V, SPY, QQQ, PLTR) is the validated minimum for daily RL

### G. Evaluation Methodology Failures

**In-sample-only evaluation pre-causal fix (2026-03-12)**
- All crypto12 results before the causal fix were inflated by look-ahead bias
- The 2,658.9x record is not reproducible on truly causal data
- Lesson: always verify the eval harness for temporal leakage before publishing results

**Contaminated train/val split**
- Using the same data for both training and OOS eval is the most dangerous failure mode
- It is easy to accidentally re-use data when re-exporting .bin files with overlapping date ranges

**Not accounting for stock market hours**
- Early stock RL evaluations ran the environment on all hours including nights and weekends
- This created spurious "signals" during periods when no trades could actually execute
- The C env must enforce market session hours for any stock symbol

**Short annualisation from short episodes**
- +5.3%/30d annualises to +87.4% per year. But a 250-day evaluation on the same model gives only +32.5% annualized
- Short-episode annualisation inflates apparent returns by 2-3x
- Minimum eval period: 90 days for crypto, 180 days for stocks

**Timezone misalignment in data joins**
- Hourly loader was silently discarding rows when CSV files contained mixed timezone labels (e.g., BTCUSD.csv containing BTCUSDT rows)
- This reduced effective forecast coverage, explaining some apparent "forecast is useless" results that were actually data join failures

### H. LLM/Prompt-Based Failures

**Volatility context in LLM prompts (Exp03, 2026-03-14)**
- Added ATR%, realised vol, Bollinger bandwidth to Gemini Flash prompt
- Result: Sortino 19.5 vs 30.9 baseline; MaxDD 87.7% vs 28.3%
- Root cause: Gemini Flash Lite is not a quant model. Extra numerical context produces erratic decisions. Keep prompts simple; Chronos2 forecasts are the right level of abstraction.

**Correlation matrix in LLM prompts (Exp04, 2026-03-14)**
- Added 48h rolling BTC/ETH/SOL correlation matrix to per-symbol prompt
- Result: Return -32.4%, Sortino -6.0 — catastrophic
- Cross-asset correlation is useful for portfolio-level risk management but must not appear in per-symbol prompts

**High confidence threshold (MIN_CONFIDENCE = 0.7, Exp02 + Exp07)**
- One 7-day window looked amazing: +357%, Sortino 39.8
- Next 7-day window: -40.7%, Sortino 2.5
- The improvement was window-specific. Multi-window validation is mandatory before deployment.
- Deployed anyway, then reverted after the second window failed.

**Market regime detection filters (Exp09)**
- Tested SMA(12/24/48), 24h return thresholds, drawdown thresholds
- All filters performed worse than unfiltered trailing stop
- Root cause: the trailing stop already handles regime changes by exiting losers within 1-2 bars; regime filters block re-entry during recoveries

**Uncertainty gating from Chronos2 p10/p90 spread**
- Restricted trades to "low uncertainty" according to quantile width
- Result: Sortino dropped from 74.7 to 5.2; trade count dropped to 6 in 3 days
- Root cause: the LLM's own reasoning already incorporates uncertainty; explicit gating is double-counting

**h24 forecast horizon added to h1 prompt**
- h1-only: Sortino 81.85, Return +13.2%
- h1+h24 combined: Sortino 61.7, Return +9.4%
- Adding the longer-horizon forecast makes the LLM more cautious and reduces trade frequency. h1 alone is the correct signal.

**Adaptive/hybrid trailing stop widths (Exp11)**
- Tested start-wide-then-tighten variants (0.8% → 0.3% after +0.3% profit)
- Fixed 0.3% beats all adaptive variants
- Root cause: wider initial trails allow drawdowns that compound into large losses; the tight trail's fast exits enable more compounding re-entries which dominate long-run returns

**Meta-selector trained on raw returns (2026-03-10)**
- Trained softmax selector on raw per-strategy returns
- Result: -8% to -65% on holdout; mode collapse to zero turnover
- Root cause: selector operated on raw returns without Chronos2 price gating; it learned to predict recent momentum which doesn't persist

### I. Hyperparameter Failures (Alpaca Neural Policy)

**Muon optimiser**
- Tested: 5 variants (`opt_muon_s42`, `opt_muon_all_s42`, `opt_muon_cautious_s42`, `opt_muon_warmdown`, `opt_muon_001_r`)
- Best val Sortino: 129 vs 258 for AdamW; qualified on 30d only (+0.21%)
- Conclusion: Muon consistently worse than AdamW for this task. Stick with AdamW.

**Multiwindow loss (mw_short)**
- Run: `mw_short_rw015_wd06_s42` (30 epochs)
- Result: 0.00% on 1d/7d windows for ALL epochs; model stopped trading in short windows
- Root cause: multiwindow loss trains the model to wait for long-term trends; it learns to be idle in 1d/7d windows which is a hard disqualifier
- Conclusion: Do not use multiwindow loss without a fix for short-window trading suppression

**High weight decay (wd >= 0.07)**
- wd=0.07: No multi-period qualified epochs; best 30d=1.20% ep19
- wd=0.08: Qualified on 30d only at 1.43% ep10
- The wd=0.06 + seq=48 combination is the validated sweet spot; going above 0.07 over-regularises

**Low weight decay without seq=48**
- wd=0.03/0.04/0.05 without seq=48: strong 1d (+1.26%) but consistently negative 120d/150d
- Root cause: low WD without sufficient sequence length overfits to recent patterns; the wd=0.06 + seq=48 combo is jointly critical

**Validation days = 60 (vd60)**
- Only 1,049 training samples remain after 60-day validation holdout
- All epochs negative, worst -22%
- 1,049 samples is insufficient for the transformer to learn anything robust
- Rule: vd=30 is the minimum viable training set size for this architecture

**Non-seed-42 runs with identical config**
- Seeds 1337 and 2024 with exact production config: NO epoch qualifies across 30 epochs
- seed=42 is a statistically lucky find; ~95% of seeds fail with the same configuration
- This is a known artefact of the training process, not a systematic edge

**Hourly RL configs applied to daily timeframes**
- slip_5bps (hourly #1 config): becomes the daily leaderboard bottom (-18.6% OOS)
- trade_penalty=0.05 (daily #1): was never in the hourly top-5
- Hyperparameter optimums are timeframe-specific; a single sweep cannot optimise both

### J. Data Pipeline Failures

**Chronos2 cache staleness**
- Stock forecasts ended Feb 13 for most symbols; live LLM prompts were using 5-week-old forecasts
- Stale forecasts actively mislead the LLM; a symbol that has moved significantly since the forecast was generated will have completely wrong directional context
- Rule: refresh Chronos2 caches at least weekly; do not use forecasts older than 7 days in production

**Mixed alias rows in CSV files**
- BTCUSD.csv contained BTCUSDT rows; the loader was dropping mismatched rows silently
- This reduced effective forecast join coverage and explained several cases where "forecast is useless" turned out to be "data is missing"

**Hourly vs daily timeframe mismatch in model evaluation**
- Evaluating an hourly-trained model on daily-aggregated data (or vice versa) gives completely wrong results
- The feature distribution is different: daily bars have 24x less noise, different autocorrelation structure

**Hold period mismatch: stocks**
- hold=4 breaks performance at 120d+; hold=6 breaks performance at 7d
- hold=5 is the only value that satisfies all evaluation periods simultaneously
- This constraint is not visible from a single-window eval

---

## 3. Current Open Questions

These are areas we have not conclusively tested. Do not assume the answer.

1. Whether transformers with cross-symbol attention outperform symbol-independent MLPs at scale (tested at small scale, always lost; untested at large scale with >500M parameters and >10 years of data).
2. Whether test-time training (TTT) or online fine-tuning can recover from regime shifts without retraining from scratch.
3. Whether a meta-ensemble using checkpoint selection (rather than raw-return selection) adds consistent signal. All raw-return ensemble attempts failed; feature-based or uncertainty-based selection is untested.
4. Optimal symbol count beyond 12 for stock daily RL. 12 is the current validated minimum; whether 24 or 50 improves holdout returns is unknown.
5. Whether adding macro features (VIX, rates, sector ETF returns, yield curve slope) meaningfully improves stock RL beyond the current OHLCV-only input.
6. Whether the daily trade_penalty sweet spot (0.05-0.20) is stable across different market regimes or is itself a regime-dependent hyperparameter.
7. Whether combining daily + hourly RL signals (majority vote) in production adds return or just adds correlation.

---

## 4. What NOT to Try Again

These are dead ends. Each entry has been tried at least once and failed with clear root-cause understanding.

| Approach | Why not |
|---|---|
| Train longer than 5-min timebox to "improve" OOS | More steps = more overfitting OOS. The overfitting is monotonic and severe (200M steps: -3% to -16% OOS). |
| Drawdown penalty > 0 with sortino loss | Destroys returns by discouraging all position-taking. 19.9x vs 42.75x baseline. |
| TSLA in live trading universe | All periods negative OOS. High idiosyncratic event risk the model cannot forecast. |
| NYT in live trading | Rallied +51.6% when model called SHORT. Special-situation stock with event-driven moves. |
| Muon optimiser | Consistently 2x worse val Sortino vs AdamW. |
| 2x+ leverage in RL | Only 48% profitable episodes vs 100% at 1x. Variance destroys the edge. |
| Kelly criterion sizing overlay | Worse than end-to-end neural policy. Double-counting the policy's embedded sizing. |
| Cross-symbol joint Chronos2 forecasting | 1,079 experiments, zero wins over per-symbol independent. Committed to False. |
| Volatility/correlation context in LLM prompts | Gemini Flash Lite is not a quant model. Numerical context causes erratic decisions. |
| Long+short RL for stocks in bull market val | -41% annualized OOS. Short leg is wrong during sustained uptrends. |
| Retrain a verified checkpoint and expect same result | RL is non-deterministic. Deploy verified checkpoints; never retrain and redeploy. |
| Regime detection filters on top of trailing stop | Trailing stop already handles regime changes. Filters just block re-entry on bounces. |
| Wider adaptive trailing stops (> 0.3%) | Fixed 0.3% beats all adaptive variants. Wider trails compound into larger losses. |
| Symbols with < 500 historical bars | Cannot learn a meaningful signal; includes BNB at time of FDUSD sweep. |
| vd=60 (validation days) | Only 1,049 train samples; all epochs negative. |
| Seeds other than 42 with the current stock neural policy config | ~95% fail rate. seed=42 is a statistical artefact, not a generally-robust config. |
| High confidence threshold (conf_0.7) for LLM signals | Window-dependent. Looked great one week (-40.7% the next). |
| Adding h24 Chronos2 horizon to h1 LLM prompts | Reduces Sortino from 81.85 to 61.7. h1 alone is sharper. |
| Per-symbol rolling Sortino gating | Gates never trigger when trailing stop is active. Redundant mechanism. |
| Flat-only evaluation for worksteal configs | Multi-start eval (flat+seeded BTC+seeded ETH) reveals configs that appear profitable but fail from non-flat starts. |

---

## 5. Experiment Graveyard

Specific named runs with dates and failure modes.

| Date | Run Name | Config | Val / OOS | Failure Mode |
|------|----------|--------|-----------|--------------|
| 2026-02-08 | rl_trading v1 | h256, 10M steps, crypto3 | 0.80x val (−42% DD) | Feature gap vs supervised; no Chronos2 features |
| 2026-02-08 | rl_trading v2 | h512, 50M steps, crypto3 | 0.88x val (−36% DD) | Same root cause; more steps made it marginally worse OOS |
| 2026-02-11 | r0_longonly_cash001 | PPO long-only, cash penalty 0.01 | −94% annualized | Cash penalty too aggressive; agent learned to stay flat |
| 2026-02-11 | r4_longshort_down10 | PPO long+short, downside=0.10 | −99% annualized | Short side + downside penalty killed all profitable trades |
| 2026-03-03 | opt_muon_s42 | AdamW replaced with Muon | Best val Sort=129 vs 258 baseline | Muon is sub-optimal for this architecture/task |
| 2026-03-03 | mw_short_rw015_wd06_s42 | Multiwindow loss | 0.00% all 1d/7d epochs | Model learned to wait; trading suppressed in short windows |
| 2026-03-03 | wd_0.07_s42 | WD=0.07, seq=48 | No multi-period qualified epochs | Over-regularised; sweet spot is 0.06 |
| 2026-03-03 | vd60_wd006_seq48_s42 | vd=60 days | All epochs negative, worst -22% | Only 1,049 training samples; too few to generalise |
| 2026-03-03 | s1337_vd30_wd006_seq48 | seed=1337, else production config | No qualifying epoch across 30 epochs | seed=42 is a statistical outlier; other seeds consistently fail |
| 2026-03-10 | learned_meta_selector | Softmax on raw returns | −8% to −65% on holdout | Mode collapse; no Chronos2 gating; raw returns don't persist |
| 2026-03-12 | crypto12_ppo_v8 (pre-causal) | All crypto12 pre-fix runs | "2,658x" (inflated) | Look-ahead bias from 1-bar obs lag bug. Not real. |
| 2026-03-13 | sortino_dd_penalty2_s1337 | DD penalty + sortino | All periods negative, worst -11.1% (120d) | DD penalty + sortino loss are incompatible |
| 2026-03-14 | uncertainty_gated (Exp03) | Chronos p10/p90 gating | Sort=5.19 vs 74.7 baseline | Over-conservative; LLM already handles uncertainty |
| 2026-03-14 | correlation_context (Exp04) | Correlation matrix in LLM prompt | -32.4% return, Sort=-6.0 | LLM confused by cross-asset numerical context |
| 2026-03-14 | conf_0.7 (Exp02) | MIN_CONFIDENCE=0.7 | +357% one week, -40.7% next | Window-dependent; not robust across market regimes |
| 2026-03-14 | regime_filter_sma48 (Exp09) | SMA48 entry gate | +5,202% vs +28,865% no-filter | Blocks profitable re-entries after drops |
| 2026-03-14 | hybrid_trail_08_03 (Exp11) | Wide-to-tight adaptive trail | Sort=53 vs 95 fixed 0.3% | Wide initial trail compounds losses before tightening |
| 2026-03-15 | baseline_daily_50M | Daily RL, 50M steps, no timebox | -22.9% OOS at best checkpoint | Long training = severe overfitting on daily timeframe |
| 2026-03-15 | longshort_stock_tp05 | Long+short stocks, daily RL | -41% annualized OOS | Short leg wrong during Jun-Dec 2025 bull market |
| 2026-03-15 | daily_tp_04 | trade_penalty=0.04 | -24.6% OOS, 0% profitable | Below the 0.05 threshold; agent overtrades |
| 2026-03-15 | daily_tp_03 | trade_penalty=0.03 | -23.2% OOS, 0% profitable | Same failure mode as tp_04 |
| 2026-03-15 | daily_wd_1 | weight_decay=0.1, daily RL | -6.43% OOS, 2% profitable | Over-regularisation kills policy diversity |
| 2026-03-15 | ep_120 | Episode length=120 days | -17.2% OOS | Does not generalise beyond 90-day horizon |
| 2026-03-15 | cosine_fee2x_combo | Cosine LR + double fee combined | -18.0% OOS | Combining too many regularisers hurts convergence |
| 2026-03-21 | random_mut_9095 | h256, smooth_downside+slippage | Robust score -118 | h256 too small; penalty terms interact destructively |
| 2026-03-21 | random_mut_7067 | h256, drawdown_penalty=0.01 | Robust score -157 | Drawdown penalty + h256 = catastrophic |
| 2026-03-21 | random_mut_257 | h256, lr=1e-4 | Robust score -159 | Low LR + small model = severe underfitting |

---

*Document compiled 2026-03-22. Sources: failedalpacaprogress.md, failedalpacaprogress4.md, failedalpacaprogress5.md, failedbinanceprogress5.md, binancefailedexperiments.md, dailyvshourly.md, alpacaprogress3-5.md, binanceprogress4-7.md, autoresearch_stock_daily_leaderboard.csv, pufferlibbest.md, project memory.*
