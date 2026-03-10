Borrow ideas from `rl-trading` and Chronos-style modeling, but only in ways that fit this fixed harness.

Relevant files:

- `rl-trading/src/rl_trading_lab/strategy_language.py`
- `rl-trading/src/rl_trading_lab/token_language.py`
- `rl-trading/src/rl_trading_lab/market_simulator.py`
- `rl-trading/src/rl_trading_lab/evaluation.py`

What to borrow:

- compact trading-plan representations,
- normalized multi-scale market-state encoding,
- explicit action concepts like flat / long / short / allocation / threshold / max hold,
- transfer-friendly tokenization ideas that improve structure without requiring a huge new vocabulary.

How to apply that here:

- treat the current outputs as a compressed plan proxy,
- add auxiliary objectives for direction, asymmetry gap, opportunity, or confidence,
- build small structured heads that better align predictions with trade decisions,
- favor plan-like interpretability over raw unconstrained regression.

Constraint:

- do not turn this into an LLM training project,
- do not weaken the realistic simulator,
- only use Chronos-style ideas that can be distilled into the current five-minute planner workflow.
