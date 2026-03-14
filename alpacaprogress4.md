# Alpaca Progress 4

## Goal

Test whether the hourly LLM trader improves when we stop over-constraining the model and when we calibrate forecast uncertainty using realized historical forecast error instead of the current Chronos quantile spread heuristic.

## New prompt variants

### `freeform`

- Keeps the same market context, constraints, and Chronos forecasts.
- Removes the strict/gated uncertainty rules.
- Tells the model to use the data however it thinks is best, as long as it clears fees and respects long-only + 6h max hold.

### `mae_bands`

- Uses the same freer prompt style as `freeform`.
- Replaces the dedicated uncertainty block with historical MAE-based error bands around the Chronos `p50` close forecast.
- The MAE band is causal in backtests:
  - It is computed from resolved historical forecasts only.
  - At decision timestamp `t`, only forecasts with `target_timestamp <= t` are used.
  - Default window is the last 30 days, falling back to all prior resolved forecasts when sample count is still small.

This mirrors the `../btcmarketsbot` homepage idea more closely than using the instantaneous `p10/p90` spread as a proxy for forecast reliability.

## Implementation notes

- Added `llm_hourly_trader/historical_error_bands.py`.
- Wired `mae_bands` into:
  - `llm_hourly_trader/experiment_runner.py`
  - `llm_hourly_trader/backtest.py`
- Added freer prompt variants to `llm_hourly_trader/gemini_wrapper.py`:
  - `freeform`
  - `mae_bands`
- Made `gemini_wrapper.py` import-safe for prompt-only/test contexts when `google-genai` is not installed.

## Verification

- `pytest -q tests/test_llm_hourly_trader_historical_error_bands.py`
- `pytest -q tests/test_llm_forecast_pipeline.py`

Both passed after making `gemini_wrapper.py` resilient to missing SDK imports during prompt-only tests.

## Pilot run

Sequential 1-day pilot on `BTCUSD`, comparing:

- `uncertainty_strict`
- `freeform`
- `mae_bands`

Command:

```bash
source .venv/bin/activate
python -m llm_hourly_trader.experiment_runner \
  --symbols BTCUSD \
  --days 1 \
  --variants uncertainty_strict freeform mae_bands \
  --model gemini-3.1-flash-lite-preview \
  --rate-limit 4.2
```

Status:

- Attempted.
- Did not complete in this shell session because the provider call stayed in a long-lived HTTPS session upstream and never returned a comparison summary in reasonable wall clock time.
- No backtest metrics recorded yet from this pilot.

## Real-data smoke check

Built a `mae_bands` prompt on current `BTCUSD` cache data without hitting the LLM provider:

- Decision timestamp: `2026-03-13 22:00:00+00:00`
- 1h historical MAE band: `0.4167%` from `721` resolved samples
- 24h historical MAE band: `2.8245%` from `721` resolved samples
- Prompt length: `3341` chars

This confirms the causal MAE-band path works on real forecast caches and current hourly history.
