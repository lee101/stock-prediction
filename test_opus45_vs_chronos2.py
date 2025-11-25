"""
Test comparing Claude Opus 4.5 vs Chronos2 for time series forecasting.
Measures MAE for line/return forecasting on BTCUSD data.
"""
import os
import re
import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import anthropic

from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.cache import async_cache_decorator

# Get API key - check both common environment variable names
api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
if not api_key:
    raise RuntimeError("API key required: set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable")

# Initialize Anthropic client for Opus 4.5
client = anthropic.Anthropic(api_key=api_key)

# Load data
base_dir = Path(__file__).parent
data_path = base_dir / "trainingdata" / "BTCUSD.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Expected dataset not found at {data_path}")

data = pd.read_csv(data_path)

# Identify close price column, support multiple naming conventions
close_column = next(
    (col for col in ["Close", "close", "Adj Close", "adj_close", "Price", "price", "close_price"] if col in data.columns),
    None
)

if close_column is None:
    raise KeyError("Unable to locate a close price column in the dataset.")

# Ensure chronological order if timestamp present
if "timestamp" in data.columns:
    data = data.sort_values("timestamp")

data = data.reset_index(drop=True)

# Convert to returns for forecasting
data['returns'] = data[close_column].astype(float).pct_change()
data = data.dropna()

# Define forecast periods - last 8 predictions for testing
end_idx = len(data) - 1
start_idx = len(data) - 9  # last 8 predictions

# Initialize Chronos2 model
chronos2_wrapper = Chronos2OHLCWrapper.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    target_columns=("close",),  # We'll use close prices for return calculation
    default_context_length=512,
)


def analyse_prediction(pred: str) -> float:
    """
    Extract the final numeric value from a model response.
    Claude occasionally wraps the answer in prose, so we always take
    the last numeric token that appears in the string.
    """
    if pred is None:
        logger.error(f"Failed to extract number from string: {pred}")
        return 0.0

    if isinstance(pred, (int, float)):
        return float(pred)

    pred_str = str(pred).strip()
    if not pred_str:
        logger.error(f"Failed to extract number from string: {pred}")
        return 0.0

    try:
        matches = re.findall(r'-?\d*\.?\d+', pred_str)
        if matches:
            return float(matches[-1])
        logger.error(f"Failed to extract number from string: {pred}")
        return 0.0
    except Exception as exc:
        logger.error(f"Failed to extract number from string: {pred} ({exc})")
        return 0.0


@async_cache_decorator(typed=True)
async def query_opus45_async(prompt: str, system_message: str) -> Optional[str]:
    """Query Claude Opus 4.5 for predictions with caching"""
    try:
        message = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=20000,
            temperature=1,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        if message.content:
            content_block = message.content[0]
            if hasattr(content_block, 'text'):
                return content_block.text
        return None
    except Exception as e:
        logger.error(f"Error in Opus 4.5 query: {e}")
        return None


def predict_chronos2_return(context_df: pd.DataFrame, close_col: str) -> float:
    """
    Use Chronos2 to predict the next close price and calculate expected return.
    """
    # Prepare data for Chronos2 - needs timestamp and target columns
    predict_df = context_df.copy()
    if 'timestamp' not in predict_df.columns:
        # Create synthetic timestamps if not present
        predict_df['timestamp'] = pd.date_range(
            end=pd.Timestamp.now(tz='UTC'),
            periods=len(predict_df),
            freq='D'
        )

    # Rename close column to match Chronos2 expectations
    if close_col != 'close':
        predict_df['close'] = predict_df[close_col].astype(float)

    try:
        batch = chronos2_wrapper.predict_ohlc(
            predict_df,
            symbol="BTCUSD",
            prediction_length=1,
            context_length=min(512, len(predict_df)),
        )

        # Get median prediction for close
        median_df = batch.quantile_frames.get(0.5)
        if median_df is not None and 'close' in median_df.columns:
            predicted_close = float(median_df['close'].iloc[0])
            current_close = float(predict_df['close'].iloc[-1])
            if current_close != 0:
                predicted_return = (predicted_close - current_close) / current_close
                return predicted_return
    except Exception as exc:
        logger.warning(f"Chronos2 prediction failed: {exc}")

    return 0.0


# Storage for results
chronos2_forecasts = []
opus45_forecasts = []

chronos2_abs_error_sum = 0.0
opus45_abs_error_sum = 0.0
prediction_count = 0

print("Generating forecasts with Opus 4.5 vs Chronos2...")
print(f"Testing {end_idx - start_idx} predictions")

with tqdm(range(start_idx, end_idx), desc="Forecasting") as progress_bar:
    for t in progress_bar:
        context = data.iloc[:t].copy()
        context_returns = data['returns'].iloc[:t]
        actual = data['returns'].iloc[t]

        # Chronos2 forecast using the wrapper
        chronos2_pred = predict_chronos2_return(context, close_column)

        # Opus 4.5 return forecast
        recent_returns = context_returns.tail(10).tolist()
        prompt = (
            f"Given these recent return values: {recent_returns}, predict the next return value as a decimal number. "
            "End your response with the numeric prediction alone on the last line."
        )
        opus45_response = asyncio.run(
            query_opus45_async(
                prompt,
                system_message=(
                    "You are a number guessing system. Provide minimal reasoning if needed, "
                    "and ensure the final line of your reply is just the numeric prediction with no trailing text."
                ),
            )
        )
        opus45_pred = analyse_prediction(opus45_response)

        chronos2_forecasts.append({
            'date': data.index[t],
            'actual': actual,
            'predicted': chronos2_pred
        })

        opus45_forecasts.append({
            'date': data.index[t],
            'actual': actual,
            'predicted': opus45_pred
        })

        prediction_count += 1
        chronos2_abs_error_sum += abs(actual - chronos2_pred)
        opus45_abs_error_sum += abs(actual - opus45_pred)

        progress_bar.set_postfix(
            chronos2_mae=chronos2_abs_error_sum / prediction_count,
            opus45_mae=opus45_abs_error_sum / prediction_count,
        )

chronos2_df = pd.DataFrame(chronos2_forecasts)
opus45_df = pd.DataFrame(opus45_forecasts)

# Calculate error metrics
chronos2_mape = mean_absolute_percentage_error(chronos2_df['actual'], chronos2_df['predicted'])
chronos2_mae = mean_absolute_error(chronos2_df['actual'], chronos2_df['predicted'])

opus45_mape = mean_absolute_percentage_error(opus45_df['actual'], opus45_df['predicted'])
opus45_mae = mean_absolute_error(opus45_df['actual'], opus45_df['predicted'])

print("\n" + "="*60)
print("RESULTS: Opus 4.5 vs Chronos2 Forecasting Comparison")
print("="*60)
print(f"\nChronos2 MAPE: {chronos2_mape:.6f}")
print(f"Chronos2 MAE: {chronos2_mae:.6f}")
print(f"\nOpus 4.5 MAPE: {opus45_mape:.6f}")
print(f"Opus 4.5 MAE: {opus45_mae:.6f}")

# Determine winner
print("\n" + "-"*60)
if chronos2_mae < opus45_mae:
    winner = "Chronos2"
    improvement = ((opus45_mae - chronos2_mae) / opus45_mae) * 100
    print(f"WINNER (lowest MAE): {winner}")
    print(f"Chronos2 MAE is {improvement:.2f}% better than Opus 4.5")
else:
    winner = "Opus 4.5"
    improvement = ((chronos2_mae - opus45_mae) / chronos2_mae) * 100
    print(f"WINNER (lowest MAE): {winner}")
    print(f"Opus 4.5 MAE is {improvement:.2f}% better than Chronos2")
print("-"*60)

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(chronos2_df.index, chronos2_df['actual'], label='Actual Returns', color='blue')
plt.plot(chronos2_df.index, chronos2_df['predicted'], label='Chronos2 Predicted Returns', color='red', linestyle='--')
plt.plot(opus45_df.index, opus45_df['predicted'], label='Opus 4.5 Predicted Returns', color='green', linestyle='--')
plt.title('Return Predictions: Opus 4.5 vs Chronos2 on BTCUSD')
plt.legend()
plt.tight_layout()
plt.savefig('opus45_vs_chronos2_returns.png', dpi=150)
print("\nPlot saved to opus45_vs_chronos2_returns.png")
