"""
Test comparing Claude Opus 4.5 (thinking) vs Gemini 3 Pro (thinking HIGH) vs Chronos2.
Measures MAE for return forecasting on BTCUSD data.
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
import anthropic
from google import genai
from google.genai import types

import os
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.cache import async_cache_decorator, sync_cache_decorator
from env_real import GEMINI_API_KEY

# Get Claude API key from environment (loaded via ~/.secretbashrc)
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY or CLAUDE_API_KEY must be set")

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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


@async_cache_decorator(typed=True)
async def query_opus45(prompt: str, system_message: str) -> Optional[str]:
    """Query Claude Opus 4.5 with extended thinking"""
    try:
        message = anthropic_client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000,
            },
            messages=[
                {"role": "user", "content": f"{system_message}\n\n{prompt}"}
            ]
        )
        if message.content:
            for content_block in message.content:
                if hasattr(content_block, 'text'):
                    return content_block.text
        return None
    except Exception as e:
        logger.error(f"Error in Opus 4.5 query: {e}")
        return None


@sync_cache_decorator(typed=True)
def query_gemini(prompt: str, system_message: str) -> Optional[str]:
    """Query Gemini 3 Pro with thinking HIGH"""
    try:
        full_prompt = f"{system_message}\n\n{prompt}"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=full_prompt),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH",
            ),
        )

        response = gemini_client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=contents,
            config=generate_content_config,
        )

        if response.text:
            return response.text
        return None
    except Exception as e:
        logger.error(f"Error in Gemini query: {e}")
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
opus45_forecasts = []
gemini_forecasts = []

print("="*70)
print("Comparing: Opus 4.5 (thinking) vs Gemini 3 Pro (thinking) vs Chronos2")
print("="*70)
print(f"Testing {end_idx - start_idx} predictions\n")

system_message = (
    "You are a number guessing system. Provide minimal reasoning if needed, "
    "and ensure the final line of your reply is just the numeric prediction with no trailing text."
)

with tqdm(range(start_idx, end_idx), desc="Forecasting") as progress_bar:
    for t in progress_bar:
        context = data.iloc[:t].copy()
        context_returns = data['returns'].iloc[:t]
        actual = data['returns'].iloc[t]

        # Chronos2 forecast
        chronos2_pred = predict_chronos2_return(context, close_column)

        # Opus 4.5 forecast
        recent_returns = context_returns.tail(10).tolist()
        prompt = (
            f"Given these recent return values: {recent_returns}, predict the next return value as a decimal number. "
            "End your response with the numeric prediction alone on the last line."
        )
        opus45_response = asyncio.run(query_opus45(prompt, system_message))
        opus45_pred = analyse_prediction(opus45_response)

        # Gemini forecast
        gemini_response = query_gemini(prompt, system_message)
        gemini_pred = analyse_prediction(gemini_response)

        chronos2_forecasts.append({'actual': actual, 'predicted': chronos2_pred})
        opus45_forecasts.append({'actual': actual, 'predicted': opus45_pred})
        gemini_forecasts.append({'actual': actual, 'predicted': gemini_pred})

        # Running MAE
        n = len(chronos2_forecasts)
        chronos2_mae = sum(abs(f['actual'] - f['predicted']) for f in chronos2_forecasts) / n
        opus45_mae = sum(abs(f['actual'] - f['predicted']) for f in opus45_forecasts) / n
        gemini_mae = sum(abs(f['actual'] - f['predicted']) for f in gemini_forecasts) / n

        progress_bar.set_postfix(
            chronos2=f"{chronos2_mae:.4f}",
            opus45=f"{opus45_mae:.4f}",
            gemini=f"{gemini_mae:.4f}",
        )

chronos2_df = pd.DataFrame(chronos2_forecasts)
opus45_df = pd.DataFrame(opus45_forecasts)
gemini_df = pd.DataFrame(gemini_forecasts)

# Calculate final metrics
chronos2_mae = mean_absolute_error(chronos2_df['actual'], chronos2_df['predicted'])
opus45_mae = mean_absolute_error(opus45_df['actual'], opus45_df['predicted'])
gemini_mae = mean_absolute_error(gemini_df['actual'], gemini_df['predicted'])

chronos2_mape = mean_absolute_percentage_error(chronos2_df['actual'], chronos2_df['predicted'])
opus45_mape = mean_absolute_percentage_error(opus45_df['actual'], opus45_df['predicted'])
gemini_mape = mean_absolute_percentage_error(gemini_df['actual'], gemini_df['predicted'])

print("\n" + "="*70)
print("FINAL RESULTS: LLMs with Thinking vs Chronos2")
print("="*70)

results = [
    ("Chronos2", chronos2_mae, chronos2_mape),
    ("Opus 4.5 + Thinking", opus45_mae, opus45_mape),
    ("Gemini 3 Pro + Thinking", gemini_mae, gemini_mape),
]

# Sort by MAE
results.sort(key=lambda x: x[1])

print(f"\n{'Model':<25} {'MAE':>12} {'MAPE':>12}")
print("-"*50)
for name, mae, mape in results:
    print(f"{name:<25} {mae:>12.6f} {mape:>12.6f}")

print("\n" + "-"*70)
winner_name, winner_mae, _ = results[0]
print(f"🏆 WINNER (lowest MAE): {winner_name}")

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
plt.plot(x, opus45_df['predicted'], label='Opus 4.5 + Thinking', color='green', linestyle='--', alpha=0.8)
plt.plot(x, gemini_df['predicted'], label='Gemini 3 Pro + Thinking', color='purple', linestyle='--', alpha=0.8)
plt.title('Return Predictions: LLMs with Thinking vs Chronos2 on BTCUSD')
plt.xlabel('Prediction Index')
plt.ylabel('Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('llms_vs_chronos2_returns.png', dpi=150)
print("\nPlot saved to llms_vs_chronos2_returns.png")
