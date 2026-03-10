Reuse the best ideas from other repo algorithms without inheriting their unrealistic assumptions.

Relevant families to borrow from:

- `unified_hourly_experiment/` meta-selection and candidate ranking,
- `marketsimlong/` and Chronos2 forecast reuse,
- `pufferlib_market/` discrete action grids and turnover-aware rewards,
- `rl-trading/` compact plan DSL and structured action/state rendering,
- honest simulation patterns from Binance neural and ETH risk PPO work.

What to preserve:

- live spread costs,
- adverse fills,
- volume caps,
- multi-window holdout scoring,
- fixed five-minute training budget.

What to avoid:

- any assumption closer to limitless liquidity or instant fills,
- importing giant subsystems when a distilled idea is enough,
- changes that make replay or attribution harder.

Good synthesis ideas:

- rank symbols the way meta selectors rank candidates,
- make predictions more action-structured like RL action grids,
- use Chronos2-style normalization or multi-horizon consistency,
- distill plan-DSL concepts into compact supervised targets.
