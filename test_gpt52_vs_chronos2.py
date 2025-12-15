"""
Test comparing GPT-5.2 (with thinking tokens) vs Chronos2.
Measures MAE for return forecasting on BTCUSD data.

GPT-5.2 uses the new Responses API with reasoning effort settings.
"""
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
from openai import OpenAI

import os
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.cache import async_cache_decorator, sync_cache_decorator

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# GPT-5.2 model IDs
GPT52_MODEL = "gpt-5.2"  # GPT-5.2 Thinking model
GPT52_INSTANT_MODEL = "gpt-5.2-chat-latest"  # GPT-5.2 Instant (faster, less reasoning)

# Load data
base_dir = Path(__file__).parent
data_path = base_dir / "trainingdata" / "BTCUSD.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Expected dataset not found at {data_path}")

data = pd.read_csv(data_path)

# Identify close price column
close_column = next(
    (col for col in ["Close", "close", "Adj Close", "adj_close", "Price", "price", "close_price"] if col in data.columns),
    None
)

if close_column is None:
    raise KeyError("Unable to locate a close price column in the dataset.")

if "timestamp" in data.columns:
    data = data.sort_values("timestamp")

data = data.reset_index(drop=True)
data['returns'] = data[close_column].astype(float).pct_change()
data = data.dropna()

# Define forecast periods - last 8 predictions for testing
end_idx = len(data) - 1
start_idx = len(data) - 9

# Initialize Chronos2 model
chronos2_wrapper = Chronos2OHLCWrapper.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    target_columns=("close",),
    default_context_length=512,
)


def analyse_prediction(pred: str) -> float:
    """Extract the final numeric value from a model response."""
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


def _extract_text_from_gpt52_response(response) -> Optional[str]:
    """Extract text from GPT-5.2 Responses API output."""
    # Try output_text first
    text_out = getattr(response, "output_text", None)
    if isinstance(text_out, str) and text_out.strip():
        return text_out.strip()

    # Traverse output blocks
    collected_parts = []
    try:
        output_blocks = getattr(response, "output", None)
        if output_blocks:
            for block in output_blocks:
                block_content = getattr(block, "content", None)
                if block_content is None and isinstance(block, dict):
                    block_content = block.get("content")
                if not block_content:
                    continue
                for item in block_content:
                    candidate = None
                    if hasattr(item, "text"):
                        candidate = getattr(item, "text")
                    elif isinstance(item, dict):
                        candidate = item.get("text") or item.get("value")
                    if candidate and isinstance(candidate, str):
                        collected_parts.append(candidate)
    except Exception as exc:
        logger.error(f"Failed to traverse GPT-5.2 response: {exc}")

    if collected_parts:
        return "\n".join(collected_parts).strip()
    return None


@sync_cache_decorator(typed=True)
def query_gpt52_thinking(prompt: str, system_message: str, reasoning_effort: str = "high") -> Optional[str]:
    """Query GPT-5.2 with thinking tokens using the Responses API."""
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        response = openai_client.responses.create(
            model=GPT52_MODEL,
            input=messages,
            max_output_tokens=4096,
            reasoning={"effort": reasoning_effort},
        )

        return _extract_text_from_gpt52_response(response)
    except Exception as e:
        logger.error(f"Error in GPT-5.2 Thinking query: {e}")
        return None


@sync_cache_decorator(typed=True)
def query_gpt52_xhigh(prompt: str, system_message: str) -> Optional[str]:
    """Query GPT-5.2 with xhigh reasoning effort (maximum thinking)."""
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        response = openai_client.responses.create(
            model=GPT52_MODEL,
            input=messages,
            max_output_tokens=8192,
            reasoning={"effort": "xhigh"},
        )

        return _extract_text_from_gpt52_response(response)
    except Exception as e:
        logger.error(f"Error in GPT-5.2 xhigh query: {e}")
        return None


@sync_cache_decorator(typed=True)
def query_gpt52_instant(prompt: str, system_message: str) -> Optional[str]:
    """Query GPT-5.2 Instant (faster, minimal reasoning)."""
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        response = openai_client.responses.create(
            model=GPT52_INSTANT_MODEL,
            input=messages,
            max_output_tokens=2048,
            reasoning={"effort": "none"},
        )

        return _extract_text_from_gpt52_response(response)
    except Exception as e:
        logger.error(f"Error in GPT-5.2 Instant query: {e}")
        return None


def predict_chronos2_return(context_df: pd.DataFrame, close_col: str) -> float:
    """Use Chronos2 to predict the next return."""
    predict_df = context_df.copy()
    if 'timestamp' not in predict_df.columns:
        predict_df['timestamp'] = pd.date_range(
            end=pd.Timestamp.now(tz='UTC'),
            periods=len(predict_df),
            freq='D'
        )

    if close_col != 'close':
        predict_df['close'] = predict_df[close_col].astype(float)

    try:
        batch = chronos2_wrapper.predict_ohlc(
            predict_df,
            symbol="BTCUSD",
            prediction_length=1,
            context_length=min(512, len(predict_df)),
        )

        median_df = batch.quantile_frames.get(0.5)
        if median_df is not None and 'close' in median_df.columns:
            predicted_close = float(median_df['close'].iloc[0])
            current_close = float(predict_df['close'].iloc[-1])
            if current_close != 0:
                return (predicted_close - current_close) / current_close
    except Exception as exc:
        logger.warning(f"Chronos2 prediction failed: {exc}")

    return 0.0


# Storage for results
chronos2_forecasts = []
gpt52_high_forecasts = []
gpt52_xhigh_forecasts = []
gpt52_instant_forecasts = []

print("="*70)
print("Comparing: GPT-5.2 (high/xhigh/instant reasoning) vs Chronos2")
print("="*70)
print(f"Testing {end_idx - start_idx} predictions\n")

system_message = (
    "You are an expert financial forecasting system. Given recent return values, "
    "predict the next return value as a decimal number. Analyze the pattern carefully. "
    "The final line of your response must be ONLY the numeric prediction with no other text."
)

with tqdm(range(start_idx, end_idx), desc="Forecasting") as progress_bar:
    for t in progress_bar:
        context = data.iloc[:t].copy()
        context_returns = data['returns'].iloc[:t]
        actual = data['returns'].iloc[t]

        # Chronos2 forecast
        chronos2_pred = predict_chronos2_return(context, close_column)

        # GPT-5.2 forecasts
        recent_returns = context_returns.tail(10).tolist()
        prompt = (
            f"Given these recent daily return values: {recent_returns}, predict the next return value as a decimal number. "
            "Consider trends, mean reversion, and volatility patterns. "
            "End your response with the numeric prediction alone on the last line."
        )

        # GPT-5.2 with high reasoning
        gpt52_high_response = query_gpt52_thinking(prompt, system_message, reasoning_effort="high")
        gpt52_high_pred = analyse_prediction(gpt52_high_response)

        # GPT-5.2 with xhigh reasoning (most thorough)
        gpt52_xhigh_response = query_gpt52_xhigh(prompt, system_message)
        gpt52_xhigh_pred = analyse_prediction(gpt52_xhigh_response)

        # GPT-5.2 Instant (fastest)
        gpt52_instant_response = query_gpt52_instant(prompt, system_message)
        gpt52_instant_pred = analyse_prediction(gpt52_instant_response)

        chronos2_forecasts.append({'actual': actual, 'predicted': chronos2_pred})
        gpt52_high_forecasts.append({'actual': actual, 'predicted': gpt52_high_pred})
        gpt52_xhigh_forecasts.append({'actual': actual, 'predicted': gpt52_xhigh_pred})
        gpt52_instant_forecasts.append({'actual': actual, 'predicted': gpt52_instant_pred})

        # Running MAE
        n = len(chronos2_forecasts)
        chronos2_mae = sum(abs(f['actual'] - f['predicted']) for f in chronos2_forecasts) / n
        gpt52_high_mae = sum(abs(f['actual'] - f['predicted']) for f in gpt52_high_forecasts) / n
        gpt52_xhigh_mae = sum(abs(f['actual'] - f['predicted']) for f in gpt52_xhigh_forecasts) / n
        gpt52_instant_mae = sum(abs(f['actual'] - f['predicted']) for f in gpt52_instant_forecasts) / n

        progress_bar.set_postfix(
            chronos2=f"{chronos2_mae:.4f}",
            gpt52_high=f"{gpt52_high_mae:.4f}",
            gpt52_xhigh=f"{gpt52_xhigh_mae:.4f}",
            gpt52_instant=f"{gpt52_instant_mae:.4f}",
        )

chronos2_df = pd.DataFrame(chronos2_forecasts)
gpt52_high_df = pd.DataFrame(gpt52_high_forecasts)
gpt52_xhigh_df = pd.DataFrame(gpt52_xhigh_forecasts)
gpt52_instant_df = pd.DataFrame(gpt52_instant_forecasts)

# Calculate final metrics
chronos2_mae = mean_absolute_error(chronos2_df['actual'], chronos2_df['predicted'])
gpt52_high_mae = mean_absolute_error(gpt52_high_df['actual'], gpt52_high_df['predicted'])
gpt52_xhigh_mae = mean_absolute_error(gpt52_xhigh_df['actual'], gpt52_xhigh_df['predicted'])
gpt52_instant_mae = mean_absolute_error(gpt52_instant_df['actual'], gpt52_instant_df['predicted'])

chronos2_mape = mean_absolute_percentage_error(chronos2_df['actual'], chronos2_df['predicted'])
gpt52_high_mape = mean_absolute_percentage_error(gpt52_high_df['actual'], gpt52_high_df['predicted'])
gpt52_xhigh_mape = mean_absolute_percentage_error(gpt52_xhigh_df['actual'], gpt52_xhigh_df['predicted'])
gpt52_instant_mape = mean_absolute_percentage_error(gpt52_instant_df['actual'], gpt52_instant_df['predicted'])

print("\n" + "="*70)
print("FINAL RESULTS: GPT-5.2 (Thinking) vs Chronos2")
print("="*70)

results = [
    ("Chronos2", chronos2_mae, chronos2_mape),
    ("GPT-5.2 + High Thinking", gpt52_high_mae, gpt52_high_mape),
    ("GPT-5.2 + xHigh Thinking", gpt52_xhigh_mae, gpt52_xhigh_mape),
    ("GPT-5.2 Instant", gpt52_instant_mae, gpt52_instant_mape),
]

# Sort by MAE
results.sort(key=lambda x: x[1])

print(f"\n{'Model':<30} {'MAE':>12} {'MAPE':>12}")
print("-"*55)
for name, mae, mape in results:
    print(f"{name:<30} {mae:>12.6f} {mape:>12.6f}")

print("\n" + "-"*70)
winner_name, winner_mae, _ = results[0]
print(f"WINNER (lowest MAE): {winner_name}")

# Calculate improvements
for name, mae, _ in results[1:]:
    improvement = ((mae - winner_mae) / mae) * 100
    print(f"   {winner_name} is {improvement:.2f}% better than {name}")
print("-"*70)

# Visualize results
plt.figure(figsize=(14, 6))
x = range(len(chronos2_df))
plt.plot(x, chronos2_df['actual'], label='Actual Returns', color='blue', linewidth=2)
plt.plot(x, chronos2_df['predicted'], label='Chronos2', color='red', linestyle='--', alpha=0.8)
plt.plot(x, gpt52_high_df['predicted'], label='GPT-5.2 High', color='green', linestyle='--', alpha=0.8)
plt.plot(x, gpt52_xhigh_df['predicted'], label='GPT-5.2 xHigh', color='purple', linestyle='--', alpha=0.8)
plt.plot(x, gpt52_instant_df['predicted'], label='GPT-5.2 Instant', color='orange', linestyle='--', alpha=0.8)
plt.title('Return Predictions: GPT-5.2 (Thinking) vs Chronos2 on BTCUSD')
plt.xlabel('Prediction Index')
plt.ylabel('Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gpt52_vs_chronos2_returns.png', dpi=150)
print("\nPlot saved to gpt52_vs_chronos2_returns.png")
