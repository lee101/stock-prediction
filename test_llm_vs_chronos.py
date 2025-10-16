from loguru import logger
import warnings
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import transformers
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
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
claude_forecasts = []
claude_binary_forecasts = []

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
claude_abs_error_sum = 0.0
claude_binary_correct = 0
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
            f"Given these recent values: {recent_returns}, predict the next return value as a decimal number. "
            "End your response with the numeric prediction alone on the last line."
        )
        claude_pred = analyse_prediction(
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

        # Claude binary forecast
        binary_context = ['up' if r > 0 else 'down' for r in recent_returns]
        binary_prompt = (
            f"Given these recent price movements: {binary_context}, predict if the next movement will be 'up' or 'down'."
        )
        binary_response = asyncio.run(
            query_to_claude_async(
                binary_prompt,
                system_message="You are a binary guessing system, just best guess the next value nothing else",
            )
        )
        claude_binary_pred = -1.0 if binary_response and 'down' in binary_response.lower() else 1.0

        chronos_forecasts.append({
            'date': data.index[t],
            'actual': actual,
            'predicted': chronos_pred_mean
        })

        claude_forecasts.append({
            'date': data.index[t],
            'actual': actual,
            'predicted': claude_pred
        })

        claude_binary_forecasts.append({
            'date': data.index[t],
            'actual': np.sign(actual),
            'predicted': claude_binary_pred
        })

        prediction_count += 1
        chronos_abs_error_sum += abs(actual - chronos_pred_mean)
        claude_abs_error_sum += abs(actual - claude_pred)
        actual_binary = np.sign(actual)
        claude_binary_correct += int(actual_binary == claude_binary_pred)

        progress_bar.set_postfix(
            chronos_mae=chronos_abs_error_sum / prediction_count,
            claude_mae=claude_abs_error_sum / prediction_count,
            binary_acc=claude_binary_correct / prediction_count,
        )

chronos_df = pd.DataFrame(chronos_forecasts)
claude_df = pd.DataFrame(claude_forecasts)
claude_binary_df = pd.DataFrame(claude_binary_forecasts)

# Calculate error metrics
chronos_mape = mean_absolute_percentage_error(chronos_df['actual'], chronos_df['predicted'])
chronos_mae = mean_absolute_error(chronos_df['actual'], chronos_df['predicted'])

claude_mape = mean_absolute_percentage_error(claude_df['actual'], claude_df['predicted'])
claude_mae = mean_absolute_error(claude_df['actual'], claude_df['predicted'])

claude_binary_accuracy = (claude_binary_df['actual'] == claude_binary_df['predicted']).mean()

print(f"\nChronos MAPE: {chronos_mape:.4f}")
print(f"Chronos MAE: {chronos_mae:.4f}")
print(f"\nClaude MAPE: {claude_mape:.4f}")
print(f"Claude MAE: {claude_mae:.4f}")
print(f"\nClaude Binary Accuracy: {claude_binary_accuracy:.4f}")

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(chronos_df.index, chronos_df['actual'], label='Actual Returns', color='blue')
plt.plot(chronos_df.index, chronos_df['predicted'], label='Chronos Predicted Returns', color='red', linestyle='--')
plt.plot(claude_df.index, claude_df['predicted'], label='Claude Predicted Returns', color='green', linestyle='--')
plt.title('Return Predictions for UNIUSD')
plt.legend()
plt.tight_layout()
plt.show()

# Plot binary predictions
plt.figure(figsize=(12, 6))
plt.plot(claude_binary_df.index, claude_binary_df['actual'], label='Actual Direction', color='blue')
plt.plot(claude_binary_df.index, claude_binary_df['predicted'], label='Claude Predicted Direction', color='orange', linestyle='--')
plt.title('Binary Direction Predictions for UNIUSD')
plt.legend()
plt.tight_layout()
plt.show()
