# Autoresearch Stock Prompt Packs

These prompt packs are for manual or scheduled agent turns that try to improve the
stock planner under the fixed five-minute training budget.

Use them with:

```bash
python -m autoresearch_stock.agent_scheduler \
  --analysis-dir analysis/autoresearch_stock_agent \
  --experiment-bundle-root experiments/autoresearch_stock_agent \
  --repo-root . \
  --python .venv312/bin/python \
  --backends codex \
  --frequencies hourly \
  --max-turns 1 \
  --prompt-file src/autoresearch_stock/prompts/isolated_experiment_rules.md \
  --prompt-file src/autoresearch_stock/prompts/hourly_selectivity.md
```

Recommended combinations:

- Hourly realism and churn control:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `chronos_plan_language.md`
  - `plan_token_auxiliary.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
- Daily robustness:
  - `isolated_experiment_rules.md`
  - `daily_robustness.md`
  - `meta_and_cross_algo_reuse.md`
- Cross-algorithm synthesis:
  - `isolated_experiment_rules.md`
  - `chronos_plan_language.md`
  - `meta_and_cross_algo_reuse.md`
- Structured plan supervision:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `chronos_plan_language.md`
  - `plan_token_auxiliary.md`
- Plan-head calibration:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `plan_token_auxiliary.md`
  - `cost_margin_calibration.md`
- Portfolio ranking:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
- Dynamic rank budget:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
- Dynamic score floor:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
  - `dynamic_score_floor.md`
- Soft rank sizing:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
  - `soft_rank_sizing.md`
- Timestamp budget head:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
  - `soft_rank_sizing.md`
  - `timestamp_budget_head.md`
- Budget guided keep count:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
  - `timestamp_budget_head.md`
  - `budget_guided_keep_count.md`
- Continuous budget thresholds:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
  - `timestamp_budget_head.md`
  - `budget_guided_keep_count.md`
  - `continuous_budget_thresholds.md`
- Budget entropy confidence:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
  - `timestamp_budget_head.md`
  - `budget_guided_keep_count.md`
  - `continuous_budget_thresholds.md`
  - `budget_entropy_confidence.md`
- Budget consensus dispersion:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
  - `timestamp_budget_head.md`
  - `budget_guided_keep_count.md`
  - `continuous_budget_thresholds.md`
  - `budget_consensus_dispersion.md`
- Budget broad support quorum:
  - `isolated_experiment_rules.md`
  - `hourly_selectivity.md`
  - `cost_margin_calibration.md`
  - `cross_sectional_rank_gate.md`
  - `dynamic_rank_budget.md`
  - `timestamp_budget_head.md`
  - `budget_guided_keep_count.md`
  - `continuous_budget_thresholds.md`
  - `budget_broad_support_quorum.md`

Prompt pack goals:

- keep experiments deterministic and replayable,
- encourage isolated code paths under `src/autoresearch_stock/experiments/`,
- reuse strong ideas from existing repo work without weakening the simulator,
- optimize for `robust_score` first and `val_loss` second.
