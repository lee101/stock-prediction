# 2026-04-26 Algorithm Research Directions

Production target remains the repo gate: median monthly return >= 27% on
100+ unseen days, worst slippage cell, decision_lag >= 2, binary fills, no
negative windows, and drawdown below the live risk cap. Nothing below is a
production change until it clears that gate.

## Current Research Read

- Time-series foundation models in finance are moving toward risk-aware
  evaluation rather than pure forecast error. The useful implication for this
  repo is to keep PnL, max drawdown, negative-window count, slippage, and
  decision lag as first-class promotion metrics.
  Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5570099
- Foundation models look more useful after financial-domain adaptation than
  as zero-shot signals. A 2025 volatility paper found incremental fine-tuning
  of TimesFM materially improved volatility forecasting versus the pretrained
  baseline and econometric benchmarks.
  Source: https://arxiv.org/abs/2505.11163
- Multivariate financial TSFM tests report transfer benefits from pretrained
  models, including TTM and Chronos, especially when data is limited. That
  supports cross-asset / covariate Chronos2 LoRA work instead of isolated
  per-symbol tuning.
  Source: https://arxiv.org/abs/2507.07296
- Limit-order-book transformer work is active, but it is most applicable here
  as an execution/slippage model, not as a daily stock selector replacement.
  Source: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full
- Recent DRL portfolio papers still emphasize strict walk-forward evaluation
  and explicit risk constraints. That matches the pufferlib path: distributional
  value / CVaR / drawdown-aware rewards are experiments, not reasons to relax
  binary-fill validation.
  Source: https://www.nature.com/articles/s41598-026-35902-x

## Experiments To Run

1. XGB+Cat wide-universe risk sizing:
   `scripts/xgbcat_risk_parity_wide_120d.sh`
   tests top_n 2/3/4 over the wider photonics/AI universe, with SPY volatility
   targeting, per-pick inverse-vol sizing, cross-sectional skew gates, score
   uncertainty penalties, and fail-fast DD gates.

2. Chronos2 / TSFM volatility gate:
   train or fine-tune a volatility/risk head, then use it only as an exposure
   scale or day-level gate first. This is safer than replacing the selector
   because the current XGB edge is already known and the research benefit is
   strongest on adaptive volatility.

3. Multivariate cross-asset Chronos2 LoRA:
   favor paired/covariate training across related equities, ETFs, crypto, and
   market-regime proxies. Evaluate as extra features in XGB/Cat and as a
   separate ensemble member; do not deploy until the ablation beats the
   current model under worst slippage.

4. Execution model:
   use quote/spread/LOB-style features to choose limit offsets and timeout
   behavior. The production rule should stay explicit-priced limit orders; the
   research question is whether the offset can reduce missed fills without
   adding adverse slippage.

5. Pufferlib risk-aware RL:
   run distributional or CVaR-style heads and drawdown/ulcer penalties against
   the same `scripts/eval_100d.py` gate. This should target smoother equity
   first; raw monthly return alone is not enough.
