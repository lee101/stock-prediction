a dedicated trading server boundary in `src/trading_server/`.

## Why

- Paper trading should not depend on Alpaca paper behavior.
- Account state should be owned by one server process, not reconstructed by each bot.
- A checked-in bot ID plus a writer lease prevents multiple schedulers from mutating the same account.
- Basic invariants like "do not sell below entry inside the protection window" should be enforced at the server boundary, not only inside strategies.
perhaps alert or failover if its doing bad behaviour like this

## Design

- Account registry: `config/trading_server/accounts.json`
- Server-owned account files: `strategy_state/trading_server/accounts/<account>.json`
- Fill ledger: `strategy_state/trading_server/events/<account>.fills.jsonl`
- Rejections ledger: `strategy_state/trading_server/events/<account>.rejections.jsonl`
- Audit ledger: `strategy_state/trading_server/events/<account>.audit.jsonl`

Each account has:

- `mode`: `paper` or `live`
- `allowed_bot_id`: checked into source control
- `starting_cash` for paper accounts
- `sell_loss_cooldown_seconds`
- `min_sell_markup_pct`

The server loads the registry once at startup. Changing `allowed_bot_id` requires a server restart by design.

## Configuration

The trading server can be configured entirely at process start without code changes.
Env-backed defaults are resolved when the engine or FastAPI app is constructed.
If a numeric env var is malformed, the server falls back to the documented safe default
instead of crashing during import.
An already-constructed engine keeps its resolved settings snapshot, so later env changes do
not silently change background-refresh or quote-cache behavior underneath it.

- `TRADING_SERVER_REGISTRY_PATH`
  Path to the account registry JSON. Defaults to `config/trading_server/accounts.json`.
- `STATE_DIR`
  Base strategy-state root. Trading server state lives under `STATE_DIR/trading_server/`.
  Defaults to `strategy_state`.
- `TRADING_SERVER_QUOTE_STALE_SECONDS`
  Quote cache freshness window. Defaults to `300`.
- `TRADING_SERVER_WRITER_TTL_SECONDS`
  Default writer-lease TTL used by API clients unless they override it in the request. The
  repo client and in-memory test adapter use the same shared default. Values are clamped to
  the API contract range `10..3600` seconds. Defaults to `120`.
- `TRADING_SERVER_BACKGROUND_POLL_SECONDS`
  Background refresh loop interval. Defaults to `300`.
- `TRADING_SERVER_QUOTE_FETCH_WORKERS`
  Max parallel quote fetches per refresh cycle. Defaults to `4`.
- `ALLOW_ALPACA_LIVE_TRADING`
  Must be `1` for live execution mode.

Example startup:

```bash
export TRADING_SERVER_REGISTRY_PATH=config/trading_server/accounts.json
export STATE_DIR=/var/lib/stock-prediction
export TRADING_SERVER_QUOTE_FETCH_WORKERS=8
uvicorn src.trading_server.server:app --host 0.0.0.0 --port 8000
```

## Writer Ownership

Mutating requests must include:

- `account`
- `bot_id`
- `session_id`

Startup flow:

1. Bot claims a writer lease for the target account.
2. Server allows only one active `session_id` per account at a time.
3. Every order must use the same `session_id`.

Identifier rules:

- account names must be 1 to 64 characters and may use letters, digits, `_`, `-`, and `.`
- symbols are normalized to uppercase and must fit within 20 characters after `/` and `-`
  separators are removed

That gives two independent protections:

- checked-in bot ownership via `allowed_bot_id`
- runtime single-writer lease via `session_id`

## Safety Checks

Paper and live orders both pass through the same guardrail layer.

- Wrong `bot_id` is rejected.
- Missing or stale writer lease is rejected.
- Live orders require `execution_mode=live`, `live_ack=LIVE`, and `ALLOW_ALPACA_LIVE_TRADING=1`.
- Sells below the server-computed safety floor are rejected unless `allow_loss_exit=true` and `force_exit_reason` is provided.
- Within the cooldown window, the safety floor is above entry by `min_sell_markup_pct`.

## Diagnostics

- Fills and rejection reasons are durable JSONL logs per account.
- The audit ledger records writer claims, heartbeat renewals, order submissions, order rejections, and price refresh summaries.
- Public API responses redact `allowed_bot_id`, `session_id`, and per-order writer credentials; the audit ledger remains the server-side place to debug ownership and sequencing problems.

## Paper Execution

- Quotes are cached and refreshed by the server.
- A background refresh loop updates prices and attempts fills every 5 minutes by default.
- Marketable paper limit orders fill immediately against the current quote.
- Non-marketable paper limit orders remain open until a later quote crosses them.

## Example

```python
from src.trading_server import TradingServerClient

client = TradingServerClient(
    account="paper_main",
    bot_id="paper_main_v1",
    execution_mode="paper",
)
client.claim_writer()
client.submit_limit_order(
    symbol="ETHUSD",
    side="buy",
    qty=0.5,
    limit_price=2000.0,
)
```
