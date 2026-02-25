"""Opus 4.6 vs Chronos2 return forecasting comparison on BTCUSD."""

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import anthropic

from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.cache import async_cache_decorator
from env_real import ANTHROPIC_API_KEY

OPUS46_MODEL_ID = os.getenv("OPUS46_MODEL_ID", "claude-opus-4-6")

api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
if not api_key:
    raise RuntimeError("Set ANTHROPIC_API_KEY or CLAUDE_API_KEY")

client = anthropic.Anthropic(api_key=api_key)

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
async def query_opus46(prompt: str, system_message: str) -> Optional[str]:
    try:
        message = client.messages.create(
            model=OPUS46_MODEL_ID,
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            messages=[{"role": "user", "content": f"{system_message}\n\n{prompt}"}],
        )
        if message.content:
            for block in message.content:
                if hasattr(block, 'text'):
                    return block.text
        return None
    except Exception as e:
        logger.error(f"Opus 4.6 error: {e}")
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
opus46_forecasts = []
chronos2_err = 0.0
opus46_err = 0.0
n = 0

print(f"Opus 4.6 ({OPUS46_MODEL_ID}) vs Chronos2 -- {end_idx - start_idx} predictions")

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
        o46_response = asyncio.run(query_opus46(
            prompt,
            system_message="You are a number guessing system. Provide minimal reasoning if needed, "
            "and ensure the final line of your reply is just the numeric prediction with no trailing text.",
        ))
        o46_pred = analyse_prediction(o46_response)

        chronos2_forecasts.append({'idx': t, 'actual': actual, 'predicted': c2_pred})
        opus46_forecasts.append({'idx': t, 'actual': actual, 'predicted': o46_pred})

        n += 1
        chronos2_err += abs(actual - c2_pred)
        opus46_err += abs(actual - o46_pred)

        pbar.set_postfix(c2_mae=chronos2_err/n, o46_mae=opus46_err/n)

c2_df = pd.DataFrame(chronos2_forecasts)
o46_df = pd.DataFrame(opus46_forecasts)

c2_mae = mean_absolute_error(c2_df['actual'], c2_df['predicted'])
c2_mape = mean_absolute_percentage_error(c2_df['actual'], c2_df['predicted'])
o46_mae = mean_absolute_error(o46_df['actual'], o46_df['predicted'])
o46_mape = mean_absolute_percentage_error(o46_df['actual'], o46_df['predicted'])

print("\n" + "=" * 60)
print("RESULTS: Opus 4.6 vs Chronos2")
print("=" * 60)
print(f"Chronos2  MAE: {c2_mae:.6f}  MAPE: {c2_mape:.6f}")
print(f"Opus 4.6  MAE: {o46_mae:.6f}  MAPE: {o46_mape:.6f}")
print("-" * 60)
if c2_mae < o46_mae:
    pct = ((o46_mae - c2_mae) / o46_mae) * 100
    print(f"WINNER: Chronos2 ({pct:.1f}% lower MAE)")
else:
    pct = ((c2_mae - o46_mae) / c2_mae) * 100
    print(f"WINNER: Opus 4.6 ({pct:.1f}% lower MAE)")

plt.figure(figsize=(12, 6))
plt.plot(c2_df.index, c2_df['actual'], label='Actual', color='blue')
plt.plot(c2_df.index, c2_df['predicted'], label='Chronos2', color='red', linestyle='--')
plt.plot(o46_df.index, o46_df['predicted'], label='Opus 4.6', color='green', linestyle='--')
plt.title('Return Predictions: Opus 4.6 vs Chronos2 (BTCUSD)')
plt.legend()
plt.tight_layout()
plt.savefig('opus46_vs_chronos2_returns.png', dpi=150)
print(f"\nPlot saved to opus46_vs_chronos2_returns.png")
