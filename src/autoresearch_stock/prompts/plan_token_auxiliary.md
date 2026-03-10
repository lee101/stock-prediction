Focus on a compact, cost-aware action-plan target instead of only raw return regression.

Motivation:

- `src/autoresearch_stock/prepare.py` already converts predictions into a long/short/flat decision after spread, slippage, and fees.
- The current model still optimizes only the three return outputs, so it can lower `val_loss` while still overtrading.
- We already have strong prior art in this repo for compact plan representations:
  - `rl-trading/src/rl_trading_lab/strategy_language.py`
  - `rl-trading/src/rl_trading_lab/token_language.py`

What to try:

- derive deterministic labels from the existing training targets plus round-trip-cost logic,
- add an auxiliary structured head for action intent, for example:
  - long / flat / short,
  - or action edge buckets such as `flat`, `weak_long`, `strong_long`, `weak_short`, `strong_short`,
- make the labels line up with the same cost-aware decision boundary used by `build_action_frame`,
- use the auxiliary head to suppress low-edge trades and ambiguous windows,
- keep the final benchmark command and simulator unchanged.

Constraints:

- do not add any LLM dependency,
- do not change `src/autoresearch_stock/prepare.py`,
- if this becomes multi-file, isolate it under `src/autoresearch_stock/experiments/<experiment_slug>/`,
- keep the experiment deterministic and simple enough to replay exactly.

Success criteria:

- better `robust_score` than the current hourly best,
- ideally lower trade count or better average edge quality,
- `val_loss` can improve, but only if trading results improve too.
