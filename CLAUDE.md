# HARD RULES (production safety — do not bypass)

1. **27%/month PnL target on unseen data.** Every new model / knob is
   judged by `scripts/eval_100d.py` at `decision_lag=2` binary fills,
   worst slip cell, median monthly ≥ 0.27. Under-target experiments do
   NOT enter prod. Use fail-fast eval (`--fail-fast-max-dd 0.20`) so
   duds cost seconds.

2. **Exactly one LIVE Alpaca writer at a time.** Everything that trades
   real money must import `alpaca_wrapper`, which calls
   `src/alpaca_singleton.py::enforce_live_singleton` at import and takes
   an fcntl lock on `<state>/account_locks/alpaca_live_writer.lock`. A
   second live import exits 42 with the holder's PID/host. Paper mode
   (`ALP_PAPER=1`) bypasses this — paper can run unlimited instances.
   Do NOT add a second path to Alpaca's write API; make new entry
   points import `alpaca_wrapper` so they inherit the gate.
   Break-glass: `ALPACA_SINGLETON_OVERRIDE=1` (human-only, never in systemd).

3. **No death-spiral sells.** `alpaca_wrapper.alpaca_order_stock` calls
   `guard_sell_against_death_spiral` before every order. Any sell
   priced >50 bps below the most recent buy for the same symbol raises
   `RuntimeError` and crashes the loop — that's the desired behaviour.
   Buy prices are tracked on disk at
   `<state>/alpaca_singleton/alpaca_live_writer_buys.json` (3-day TTL).
   Break-glass: `ALPACA_DEATH_SPIRAL_OVERRIDE=1` (human-only, loudly logged).

4. **Tests in `tests/test_alpaca_singleton.py` must stay green.**
   Change these and AGENTS.md in the same commit if you change the guard.

---

dont use git branches all on one branch is fine
use uv pip NEVER just pip

try not use uv run though just activate the python env then use normal python/pytest

this is a monorepo for trading experiments

we have a few python envs .venv .venv312 etc we try to get them all working as ideally we would be on latest as we can able to use latest tech but sometimes we cant for some experiments

dont use timeouts as we want to train long

fully finish tasks eg if it means install uv pip packages, write the tests and run them then run the related benchmarks for real with long timeouts - dont give up

code is requiring a lot of thought here as its a production trading bot

try do as much work as you can so dont just give up on installing packages - add them to pyproject.toml uv sync and install -e toto/ too just do things and get stuff tested then simulated properly all the way done

write tests/test a lot while developing - use tools 100s of tool calls is great

Ensure every code modification strictly preserves correctness, minimality of change, and robustly handles edge/corner cases related to the problem statement. ok use simple code structures like functions not complex inheritence.

Avoid blanket or “quick fix” solutions that might hide errors or unintentionally discard critical information; always strive to diagnose and address root-causes, not merely symptoms or side-effects.

Where input normalization is necessary - for types, iterables, containers, or input shapes - do so only in a way that preserves API contracts, allows for extensibility, and maintains invariance across all supported data types, including Python built-ins and major library types. can put any re usable utils in src/ and test them

All error/warning messages, exceptions, and documentation updates must be technically accurate, actionable, match the conventions of the host codebase, and be kept fully in sync with new or changed behavior.

Backwards and forwards compatibility: Changes must account for code used in diverse environments (e.g., different Python versions, framework/ORM versions, or platforms), and leverage feature detection where possible to avoid breaking downstream or legacy code.

Refactorings and bugfixes must never silently discard, mask, or change user data, hooks, plugin registrations, or extension points; if a migration or transformation is required, ensure it is invertible/idempotent where possible

use latest tactics in terms of machine learning can see nanochat/ for some good practice

instead of reconfirming with me just do it - you are probably right and yea i can always roll back thats fine lets just do it.

Creating new experiment directories is expected and safe; keep experiments reproducible so we can rerun them and match marketsimulator PnL closely to production.

if you find unexpected changes you should be thorough with resolving them yourself and git commit and push work as you go is expected, work end to end autonomously

aiming for the best pnl and smoothness in pnl sortino etc low max drawdown trading strategy we can find methodically on our marketsimulator
we control everything here so forking projects and C cuda kernels etc is fine we own this and can do whatever we want to succeed
we have a few projects like nanochat autoresearch modded-nanogpt etc that are here as examples of code and the chronos2 project is important as we train loras and fit hyperparameters/preaugmentations of that too

## Production
see alpacaprod.md for whats running, marketsim scores, deploy commands, and monitoring
always update alpacaprod.md when deploying or changing production systems
validate with binary-fill marketsim at lag>=2 before deploying any neural model
soft sigmoid fills have lookahead bias -- never trust training sortino alone

## Marketsim Realism
- fee=10bps, margin=6.25%, fill_buffer=5bps, max_hold=6h, decision_lag>=2
- validation_use_binary_fills=True (default in config.py)
- fill_temperature=0.01 (reduced from 5e-4 to limit gradient leakage)
- test at slippage 0/5/10/20 bps before deploying
- pufferlib C sim with binary fills is ground truth -- binanceneural soft sim is for training gradients only
