# Ambiguity Plan Gate

Hypothesis:
- The hourly planner is still overtrading because the regression head sees many apparently cost-positive windows, even when upside and downside are too balanced to deserve a real position.
- A compact five-bucket action-plan head (`weak_short`, `strong_short`, `flat`, `weak_long`, `strong_long`) should make those ambiguous windows explicit and gate low-conviction outputs before the fixed simulator sees them.

Exact files changed:
- `src/autoresearch_stock/train.py`
- `src/autoresearch_stock/experiments/ambiguity_plan_gate/__init__.py`
- `src/autoresearch_stock/experiments/ambiguity_plan_gate/plan_head.py`
- `tests/test_autoresearch_stock_plan_head.py`

Exact benchmark command:
- `/nvme0n1-disk/code/stock-prediction/.venv312/bin/python -m autoresearch_stock.train --frequency hourly > /nvme0n1-disk/code/stock-prediction/analysis/autoresearch_stock_agent/turn_0004_codex_hourly_20260310T031448Z/train_hourly.log 2>&1`

Borrowed repo ideas:
- Compact plan-token supervision from `rl-trading/src/rl_trading_lab/strategy_language.py`
- Structured, trade-oriented compression from `rl-trading/src/rl_trading_lab/token_language.py`
- Chronos-style preference for simple, explicit state-to-plan structure instead of unconstrained outputs

Why this should help the realistic simulator:
- The simulator already prices spread, slippage, and fees. The new plan head uses the same cost inputs plus ambiguity-derived buckets to damp predictions on weak or two-sided windows.
- That should reduce low-edge churn without touching `prepare.py` or weakening the fixed execution harness.
