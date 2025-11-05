# Forecast Validation Integration Guide

## Overview

The `src/forecast_validation.py` module provides validation and correction for OHLC price forecasts, ensuring they maintain logical ordering: `low_price <= close_price <= high_price`.

This mirrors the production retry logic from `trade_stock_e2e.py:1436-1515`.

## Key Components

### 1. OHLCForecast Dataclass

```python
from src.forecast_validation import OHLCForecast

forecast = OHLCForecast(
    open_price=100.0,
    high_price=105.0,
    low_price=98.0,
    close_price=102.0,
)

# Check validity
if forecast.is_valid():
    print("Forecast is valid!")
else:
    # Get violations
    violations = forecast.get_violations()
    print(f"Violations: {violations}")

    # Auto-correct
    corrected = forecast.correct()
```

### 2. Retry Logic

```python
from src.forecast_validation import forecast_with_retry

def my_forecast_function():
    """Your model prediction function that returns OHLCForecast."""
    # ... call Kronos/Toto model
    return OHLCForecast(...)

# Automatic retry with correction fallback
forecast, retry_count = forecast_with_retry(
    my_forecast_function,
    max_retries=2,
    symbol="AAPL",
)

print(f"Got valid forecast after {retry_count} retries")
```

### 3. Simple Validation & Correction

```python
from src.forecast_validation import validate_and_correct_forecast

o, h, l, c = validate_and_correct_forecast(
    open_price=100.0,
    high_price=98.0,  # Invalid
    low_price=105.0,  # Invalid
    close_price=102.0,
    symbol="AAPL",
)

# Returns corrected prices that maintain low <= close <= high
```

## Integration with Model Wrappers

### Example: Kronos Wrapper Integration

```python
# In src/models/kronos_wrapper.py

from src.forecast_validation import OHLCForecast, forecast_with_retry

class KronosForecastingWrapper:
    def predict_ohlc_with_validation(self, symbol: str) -> OHLCForecast:
        """Predict OHLC with automatic validation and retry."""

        def _forecast_fn():
            # Your existing prediction logic
            predictions = self.predict(...)

            return OHLCForecast(
                open_price=predictions['open'],
                high_price=predictions['high'],
                low_price=predictions['low'],
                close_price=predictions['close'],
            )

        # Automatically retry and correct if needed
        forecast, retries = forecast_with_retry(
            _forecast_fn,
            max_retries=2,
            symbol=symbol,
        )

        return forecast
```

### Example: Toto Wrapper Integration

```python
# In src/models/toto_wrapper.py

from src.forecast_validation import validate_and_correct_forecast

class TotoPipeline:
    def forecast_next_day(self, symbol: str):
        """Forecast with automatic validation."""

        # Your existing prediction logic
        raw_predictions = self._predict(...)

        # Validate and correct
        o, h, l, c = validate_and_correct_forecast(
            open_price=raw_predictions['open'],
            high_price=raw_predictions['high'],
            low_price=raw_predictions['low'],
            close_price=raw_predictions['close'],
            symbol=symbol,
        )

        return {
            'open': o,
            'high': h,
            'low': l,
            'close': c,
        }
```

## Batched OHLC Prediction

For better performance, you could batch all OHLC predictions together:

```python
from src.forecast_validation import OHLCForecast
import torch

def batch_predict_ohlc(model, inputs):
    """
    Predict all OHLC values in a single forward pass.

    Args:
        model: Your forecasting model
        inputs: Batched inputs

    Returns:
        List of validated OHLCForecast objects
    """
    # Single model call for all predictions
    with torch.no_grad():
        outputs = model(inputs)  # Shape: [batch, 4] for O, H, L, C

    forecasts = []
    for i in range(outputs.shape[0]):
        forecast = OHLCForecast(
            open_price=float(outputs[i, 0]),
            high_price=float(outputs[i, 1]),
            low_price=float(outputs[i, 2]),
            close_price=float(outputs[i, 3]),
        )

        # Validate and correct
        if not forecast.is_valid():
            forecast = forecast.correct()

        forecasts.append(forecast)

    return forecasts
```

## Benefits

1. **Production Parity**: Mirrors the exact validation logic used in `trade_stock_e2e.py`
2. **Automatic Retry**: Attempts to get valid forecasts by retrying the model
3. **Graceful Fallback**: Applies corrections if retries fail
4. **Comprehensive Logging**: Tracks violations and corrections
5. **Well-Tested**: 20 unit tests covering all edge cases
6. **Type-Safe**: Uses dataclasses for clear interfaces

## Running Tests

```bash
source .venv/bin/activate
python test_forecast_validation.py
```

All 20 tests should pass, validating:
- Valid forecast detection
- Invalid forecast detection (inverted, close out of bounds, open out of bounds)
- Correction logic
- Retry mechanism
- Exception handling
- Edge cases
