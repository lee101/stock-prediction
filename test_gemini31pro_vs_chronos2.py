"""Gemini 3.1 Pro (via OpenRouter) vs Chronos2 return forecasting comparison on BTCUSD."""

import os
import re
import asyncio
import requests
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.cache import async_cache_decorator

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
GEMINI_MODEL = "google/gemini-2.5-pro"

base_dir = Path(__file__).parent
data_path = base_dir / "trainingdata" / "BTCUSD.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

data = pd.read_csv(data_path)
close_column = next(
    (c for c in ["Close", "close", "Adj Close", "Price", "close_price"] if c in data.columns), None
)
if close_column is None:
    raise KeyError("No close price column found")

if "timestamp" in data.columns:
    data = data.sort_values("timestamp")
data = data.reset_index(drop=True)
data['returns'] = data[close_column].astype(float).pct_change()
data = data.dropna()

end_idx = len(data) - 1
start_idx = len(data) - 9

chronos2_wrapper = Chronos2OHLCWrapper.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    target_columns=("close",),
    default_context_length=512,
)


def analyse_prediction(pred: str) -> float:
    if pred is None:
        return 0.0
    if isinstance(pred, (int, float)):
        return float(pred)
    pred_str = str(pred).strip()
    if not pred_str:
        return 0.0
    matches = re.findall(r"-?\d*\.?\d+", pred_str)
    return float(matches[-1]) if matches else 0.0


@async_cache_decorator(typed=True)
async def query_gemini(prompt: str, system_message: str) -> Optional[str]:
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GEMINI_MODEL,
                "max_tokens": 8000,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        data_resp = resp.json()
        return data_resp["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return None


def predict_chronos2_return(context_df: pd.DataFrame, close_col: str) -> float:
    predict_df = context_df.copy()
    if 'timestamp' not in predict_df.columns:
        predict_df['timestamp'] = pd.date_range(
            end=pd.Timestamp.now(tz='UTC'), periods=len(predict_df), freq='D'
        )
    if close_col != 'close':
        predict_df['close'] = predict_df[close_col].astype(float)
    try:
        batch = chronos2_wrapper.predict_ohlc(
            predict_df, symbol="BTCUSD", prediction_length=1,
            context_length=min(512, len(predict_df)),
        )
        median_df = batch.quantile_frames.get(0.5)
        if median_df is not None and 'close' in median_df.columns:
            predicted_close = float(median_df['close'].iloc[0])
            current_close = float(predict_df['close'].iloc[-1])
            if current_close != 0:
                return (predicted_close - current_close) / current_close
    except Exception as exc:
        logger.warning(f"Chronos2 failed: {exc}")
    return 0.0


chronos2_forecasts = []
gemini_forecasts = []
chronos2_err = 0.0
gemini_err = 0.0
n = 0

print(f"Gemini 3.1 Pro ({GEMINI_MODEL}) via OpenRouter vs Chronos2 -- {end_idx - start_idx} predictions")

with tqdm(range(start_idx, end_idx), desc="Forecasting") as pbar:
    for t in pbar:
        context = data.iloc[:t].copy()
        context_returns = data['returns'].iloc[:t]
        actual = data['returns'].iloc[t]

        c2_pred = predict_chronos2_return(context, close_column)

        recent = context_returns.tail(10).tolist()
        prompt = (
            f"Given these recent return values: {recent}, predict the next return value as a decimal number. "
            "End your response with the numeric prediction alone on the last line."
        )
        gem_response = asyncio.run(query_gemini(
            prompt,
            system_message="You are a number guessing system. Provide minimal reasoning if needed, "
            "and ensure the final line of your reply is just the numeric prediction with no trailing text.",
        ))
        gem_pred = analyse_prediction(gem_response)

        chronos2_forecasts.append({'idx': t, 'actual': actual, 'predicted': c2_pred})
        gemini_forecasts.append({'idx': t, 'actual': actual, 'predicted': gem_pred})

        n += 1
        chronos2_err += abs(actual - c2_pred)
        gemini_err += abs(actual - gem_pred)

        pbar.set_postfix(c2_mae=chronos2_err/n, gem_mae=gemini_err/n)

c2_df = pd.DataFrame(chronos2_forecasts)
gem_df = pd.DataFrame(gemini_forecasts)

c2_mae = mean_absolute_error(c2_df['actual'], c2_df['predicted'])
c2_mape = mean_absolute_percentage_error(c2_df['actual'], c2_df['predicted'])
gem_mae = mean_absolute_error(gem_df['actual'], gem_df['predicted'])
gem_mape = mean_absolute_percentage_error(gem_df['actual'], gem_df['predicted'])

print("\n" + "=" * 60)
print("RESULTS: Gemini 3.1 Pro vs Chronos2")
print("=" * 60)
print(f"Chronos2      MAE: {c2_mae:.6f}  MAPE: {c2_mape:.6f}")
print(f"Gemini 3.1Pro MAE: {gem_mae:.6f}  MAPE: {gem_mape:.6f}")
print("-" * 60)
if c2_mae < gem_mae:
    pct = ((gem_mae - c2_mae) / gem_mae) * 100
    print(f"WINNER: Chronos2 ({pct:.1f}% lower MAE)")
else:
    pct = ((c2_mae - gem_mae) / c2_mae) * 100
    print(f"WINNER: Gemini 3.1 Pro ({pct:.1f}% lower MAE)")

plt.figure(figsize=(12, 6))
plt.plot(c2_df.index, c2_df['actual'], label='Actual', color='blue')
plt.plot(c2_df.index, c2_df['predicted'], label='Chronos2', color='red', linestyle='--')
plt.plot(gem_df.index, gem_df['predicted'], label='Gemini 3.1 Pro', color='purple', linestyle='--')
plt.title('Return Predictions: Gemini 3.1 Pro vs Chronos2 (BTCUSD)')
plt.legend()
plt.tight_layout()
plt.savefig('gemini31pro_vs_chronos2_returns.png', dpi=150)
print(f"\nPlot saved to gemini31pro_vs_chronos2_returns.png")
