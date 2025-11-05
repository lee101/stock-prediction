# Batched Target Optimization Proposal

## Current Bottleneck

**Line 2316**: Loop over 4 targets sequentially
```python
for key_to_predict in ['Close', 'Low', 'High', 'Open']:
    # ... prepare data for this target
    toto_predictions, toto_band, toto_abs = _compute_toto_forecast(...)  # Line 2345
```

Each `_compute_toto_forecast()` makes **7 GPU calls** (walk-forward horizons 1-7).

**Total: 28 GPU calls per simulation** (4 targets × 7 horizons)

## Proposed Optimization

Batch all 4 targets together:

### Step 1: Prepare all targets upfront
```python
# Prepare all 4 targets before loop
target_data = {}
for key_to_predict in ['Close', 'Low', 'High', 'Open']:
    data = pre_process_data(simulation_data, key_to_predict)
    price = data[["Close", "High", "Low", "Open"]]
    price = price.rename(columns={"Date": "time_idx"})
    price["ds"] = pd.date_range(start="1949-01-01", periods=len(price), freq="D").values
    target_series = price[key_to_predict].shift(-1)
    price["y"] = target_series.to_numpy()
    price = price.dropna()
    target_data[key_to_predict] = price
```

### Step 2: Create batched forecast function
```python
def _compute_toto_forecast_batched(
    symbol: str,
    target_keys: List[str],
    price_frames: Dict[str, pd.DataFrame],
    current_last_prices: Dict[str, float],
    toto_params: dict,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Batch predictions across multiple targets (Close, Low, High, Open).

    Returns dict mapping target_key -> (predictions, bands, predicted_absolute_last)
    """
    max_horizon = 7
    results = {}

    # Walk-forward over horizons (still need 7 steps)
    for pred_idx in reversed(range(1, max_horizon + 1)):
        # Stack all targets into a batch
        contexts = []
        valid_targets = []

        for key in target_keys:
            price_frame = price_frames[key]
            if len(price_frame) <= pred_idx:
                continue
            current_context = price_frame[:-pred_idx]
            if current_context.empty:
                continue

            context = torch.tensor(current_context["y"].values, dtype=torch.float32)
            contexts.append(context)
            valid_targets.append(key)

        if not contexts:
            continue

        # BATCH: Stack into [batch_size, seq_len]
        batched_context = torch.stack(contexts)
        if torch.cuda.is_available() and batched_context.device.type == 'cpu':
            batched_context = batched_context.to('cuda', non_blocking=True)

        # Single batched GPU call for all targets!
        batched_forecast = cached_predict(
            batched_context,
            1,
            num_samples=requested_num_samples,
            samples_per_batch=requested_batch,
            symbol=symbol,
        )

        # Un-batch results
        for idx, key in enumerate(valid_targets):
            if key not in results:
                results[key] = {'predictions': [], 'bands': []}

            tensor = batched_forecast[idx]  # Get this target's forecast
            # ... process distribution, percentiles, etc.
            results[key]['predictions'].append(prediction_value)
            results[key]['bands'].append(band_width)

    # Convert lists to tensors
    final_results = {}
    for key in target_keys:
        if key in results:
            predictions = torch.tensor(results[key]['predictions'], dtype=torch.float32)
            bands = torch.tensor(results[key]['bands'], dtype=torch.float32)
            predicted_absolute_last = current_last_prices[key] * (1.0 + predictions[-1].item())
            final_results[key] = (predictions, bands, predicted_absolute_last)

    return final_results
```

### Step 3: Call batched function
```python
# Replace the loop at line 2316 with:
batched_results = _compute_toto_forecast_batched(
    symbol,
    ['Close', 'Low', 'High', 'Open'],
    target_data,
    {k: float(simulation_data[k].iloc[-1]) for k in ['Close', 'Low', 'High', 'Open']},
    toto_params,
)

# Then iterate for validation/strategy only:
for key_to_predict in ['Close', 'Low', 'High', 'Open']:
    toto_predictions, toto_band, toto_abs = batched_results.get(key_to_predict, (None, None, None))
    # ... rest of strategy logic
```

## Expected Speedup

**Current**: 28 GPU calls per simulation
**Optimized**: 7 GPU calls per simulation (4x batched)

**Expected speedup**: 3-4x (accounting for batching overhead)

## Validation Required

1. ✅ Ensure `pipeline.predict()` supports batched context input
2. ✅ Verify forecast quality unchanged (MAE should be identical)
3. ✅ Test that batch dimension is correctly handled in return structure
4. ✅ Confirm no CUDA OOM with 4x batch size

## Implementation Notes

- Keep walk-forward logic (7 horizons) - only batch across targets
- Maintain async GPU transfers (`non_blocking=True`)
- Handle variable-length contexts gracefully (some targets may have different data lengths)
- Error handling: if batch fails, fallback to sequential per-target
