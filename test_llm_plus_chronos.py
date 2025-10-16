from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import transformers
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from tqdm import tqdm
from pathlib import Path
import asyncio
from claude_queries import query_to_claude_async
from src.cache import async_cache_decorator

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

# Convert to returns
data['returns'] = data[close_column].astype(float).pct_change()
data = data.dropna()

# Define forecast periods
# start_idx = int(len(data) * 0.8) # Use last 20% for testing
end_idx = len(data) - 1
start_idx = len(data) -9 # last 8 for now

# Generate forecasts with Chronos
chronos_forecasts = []
claude_plus_forecasts = []

chronos_model = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
import re

def analyse_prediction(pred: str):
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
async def predict_chronos(context_values):
    """Cached prediction function that doesn't include the model in the cache key"""
    with torch.inference_mode():
        transformers.set_seed(42)
        pred = chronos_model.predict(
            context=torch.from_numpy(context_values),
            prediction_length=1,
            num_samples=100
        ).detach().cpu().numpy().flatten()
        return np.mean(pred)

chronos_abs_error_sum = 0.0
claude_plus_abs_error_sum = 0.0
prediction_count = 0

print("Generating forecasts...")
with tqdm(range(start_idx, end_idx), desc="Forecasting") as progress_bar:
    for t in progress_bar:
        context = data['returns'].iloc[:t]
        actual = data['returns'].iloc[t]

        # Chronos forecast - now not passing model as argument
        chronos_pred_mean = asyncio.run(predict_chronos(context.values))

        # Claude forecast
        recent_returns = context.tail(10).tolist()
        prompt = (
            "You are collaborating with the Chronos time-series model to improve number forecasting.\n"
            f"Chronos predicts the next return will be {chronos_pred_mean:.6f}.\n"
            "Chronos benchmark accuracy: MAE 0.0294.\n"
            "Your previous solo performance without Chronos context: MAE 0.0315.\n"
            f"Recent observed numbers leading into this step: {recent_returns}.\n"
            "Provide your updated numeric prediction leveraging Chronos' forecast. "
            "Think thoroughly, ultrathink, but ensure the final line of your reply is only the numeric prediction, you need to improve upon the prediction though we cant keep it."
        )
        claude_plus_pred = analyse_prediction(
            asyncio.run(
                query_to_claude_async(
                    prompt,
                    system_message=(
                        "You are a number guessing system. Provide minimal reasoning if needed, "
                        "and ensure the final line of your reply is just the numeric prediction with no trailing text."
                    ),
                )
            )
        )

        chronos_forecasts.append({
            'date': data.index[t],
            'actual': actual,
            'predicted': chronos_pred_mean
        })

        claude_plus_forecasts.append({
            'date': data.index[t],
            'actual': actual,
            'predicted': claude_plus_pred
        })

        prediction_count += 1
        chronos_abs_error_sum += abs(actual - chronos_pred_mean)
        claude_plus_abs_error_sum += abs(actual - claude_plus_pred)

        progress_bar.set_postfix(
            chronos_mae=chronos_abs_error_sum / prediction_count,
            chronos_plus_claude_mae=claude_plus_abs_error_sum / prediction_count,
        )

chronos_df = pd.DataFrame(chronos_forecasts)
claude_plus_df = pd.DataFrame(claude_plus_forecasts)

# Calculate error metrics
chronos_mape = mean_absolute_percentage_error(chronos_df['actual'], chronos_df['predicted'])
chronos_mae = mean_absolute_error(chronos_df['actual'], chronos_df['predicted'])

chronos_plus_claude_mape = mean_absolute_percentage_error(claude_plus_df['actual'], claude_plus_df['predicted'])
chronos_plus_claude_mae = mean_absolute_error(claude_plus_df['actual'], claude_plus_df['predicted'])

print(f"\nChronos MAPE: {chronos_mape:.4f}")
print(f"Chronos MAE: {chronos_mae:.4f}")
print(f"\nChronos+Claude MAPE: {chronos_plus_claude_mape:.4f}")
print(f"Chronos+Claude MAE: {chronos_plus_claude_mae:.4f}")

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(chronos_df.index, chronos_df['actual'], label='Actual Returns', color='blue')
plt.plot(chronos_df.index, chronos_df['predicted'], label='Chronos Predicted Returns', color='red', linestyle='--')
plt.plot(claude_plus_df.index, claude_plus_df['predicted'], label='Chronos-Aware Claude Predicted Returns', color='green', linestyle='--')
plt.title('Return Predictions for UNIUSD')
plt.legend()
plt.tight_layout()
plt.show()
