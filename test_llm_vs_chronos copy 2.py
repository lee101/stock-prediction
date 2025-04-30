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
import openai
from src.cache import async_cache_decorator
import re

# Load data
base_dir = Path(__file__).parent
data_dir = base_dir / "data" / "2024-09-07--03-36-27"
data = pd.read_csv(data_dir / "UNIUSD-2024-12-28.csv")

# Convert to returns
data['returns'] = data['Close'].pct_change()
data = data.dropna()

# Define forecast periods
start_idx = int(len(data) * 0.8)  # Use last 20% for testing
end_idx = len(data) - 1

# Generate forecasts with Chronos
chronos_forecasts = []
test_forecasts = []
test_binary_forecasts = []

chronos_model = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cuda", 
    torch_dtype=torch.bfloat16
)

def analyse_prediction(pred: str):
    """
    Extracts a number from the model response string.
    """
    if not pred:
        logger.error(f"Failed to extract number from string: {pred}")
        return 0.0
    try:
        matches = re.findall(r'-?\d*\.?\d+', pred)
        if matches:
            return float(matches[0])
        logger.error(f"Failed to extract number from string: {pred}")
        return 0.0
    except Exception:
        logger.error(f"Failed to extract number from string: {pred}")
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

async def query_to_o4_mini(prompt: str, system_message: str = None):
    """
    Sends an async chat completion request to OpenAI's o4-mini model.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    response = await openai.ChatCompletion.acreate(
        model="o4-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

print("Generating forecasts...")
for t in tqdm(range(start_idx, end_idx)):
    context = data['returns'].iloc[:t]
    actual = data['returns'].iloc[t]

    # Chronos forecast
    chronos_pred_mean = asyncio.run(predict_chronos(context.values))

    # Test forecast using OpenAI o4-mini
    recent_returns = context.tail(10).tolist()
    prompt = f"Given these recent values: {recent_returns}, predict the next return value as a decimal number."
    system_msg = "You are a number guessing system, just best guess the next value nothing else"
    test_response = asyncio.run(query_to_o4_mini(prompt, system_message=system_msg))
    test_pred = analyse_prediction(test_response)

    # Test binary forecast
    binary_context = ['up' if r > 0 else 'down' for r in recent_returns]
    binary_prompt = f"Given these recent price movements: {binary_context}, predict if the next movement will be 'up' or 'down'."
    binary_system_msg = "You are a binary guessing system, just best guess the next value nothing else"
    binary_response = asyncio.run(query_to_o4_mini(binary_prompt, system_message=binary_system_msg))
    test_binary_pred = -1.0 if binary_response and 'down' in binary_response.lower() else 1.0

    chronos_forecasts.append({
        'date': data.index[t],
        'actual': actual,
        'predicted': chronos_pred_mean
    })

    test_forecasts.append({
        'date': data.index[t],
        'actual': actual,
        'predicted': test_pred
    })

    test_binary_forecasts.append({
        'date': data.index[t],
        'actual': np.sign(actual),
        'predicted': test_binary_pred
    })

chronos_df = pd.DataFrame(chronos_forecasts)
test_df = pd.DataFrame(test_forecasts)
test_binary_df = pd.DataFrame(test_binary_forecasts)

# Calculate error metrics
chronos_mape = mean_absolute_percentage_error(chronos_df['actual'], chronos_df['predicted'])
chronos_mae = mean_absolute_error(chronos_df['actual'], chronos_df['predicted'])

test_mape = mean_absolute_percentage_error(test_df['actual'], test_df['predicted'])
test_mae = mean_absolute_error(test_df['actual'], test_df['predicted'])

test_binary_accuracy = (test_binary_df['actual'] == test_binary_df['predicted']).mean()

print(f"\nChronos MAPE: {chronos_mape:.4f}")
print(f"Chronos MAE: {chronos_mae:.4f}")
print(f"\nTest MAPE: {test_mape:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"\nTest Binary Accuracy: {test_binary_accuracy:.4f}")

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(chronos_df.index, chronos_df['actual'], label='Actual Returns', color='blue')
plt.plot(chronos_df.index, chronos_df['predicted'], label='Chronos Predicted Returns', color='red', linestyle='--')
plt.plot(test_df.index, test_df['predicted'], label='Test Predicted Returns', color='green', linestyle='--')
plt.title('Return Predictions for UNIUSD')
plt.legend()
plt.tight_layout()
plt.show()

# Plot binary predictions
plt.figure(figsize=(12, 6))
plt.plot(test_binary_df.index, test_binary_df['actual'], label='Actual Direction', color='blue')
plt.plot(test_binary_df.index, test_binary_df['predicted'], label='Test Predicted Direction', color='orange', linestyle='--')
plt.title('Binary Direction Predictions for UNIUSD')
plt.legend()
plt.tight_layout()
plt.show()
