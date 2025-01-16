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
data_dir = base_dir / "data" / "2024-09-07--03-36-27"
data = pd.read_csv(data_dir / "UNIUSD-2024-12-28.csv")

# Convert to returns
data['returns'] = data['Close'].pct_change()
data = data.dropna()

# Define forecast periods
start_idx = int(len(data) * 0.8) # Use last 20% for testing
end_idx = len(data) - 1

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
    claude can return a string
     eg. Based on the recent values provided, my best guess for the next return value is: -0.015
     or 
     we need to extract the number from the string
     """
    if not pred:
        logger.error(f"Failed to extract number from string: {pred}")
        return 0.0
    try:
         # Look for a number in the string using regex
        matches = re.findall(r'-?\d*\.?\d+', pred)
        if matches:
            return float(matches[0])
        # log
        logger.error(f"Failed to extract number from string: {pred}")
        return 0.0
    except:
        logger.error(f"Failed to extract number from string: {pred}")
        return 0.0
        
@async_cache_decorator(typed=True)
async def predict_chronos(model, context_values):
    with torch.inference_mode():
        transformers.set_seed(42)
        pred = model.predict(
            context=torch.from_numpy(context_values),
            prediction_length=1, 
            num_samples=100
        ).detach().cpu().numpy().flatten()
        return np.mean(pred)

print("Generating forecasts...")
for t in tqdm(range(start_idx, end_idx)):
    context = data['returns'].iloc[:t]
    actual = data['returns'].iloc[t]
    
    # Chronos forecast
    chronos_pred_mean = asyncio.run(predict_chronos(chronos_model, context.values))
    
    # Claude forecast
    recent_returns = context.tail(10).tolist()
    prompt = f"Given these recent values: {recent_returns}, predict the next return value as a decimal number."
    claude_pred = analyse_prediction(asyncio.run(query_to_claude_async(prompt, system_message="You are a number guessing system, just best guess the next value nothing else")))
    
    # Claude binary forecast
    binary_context = ['up' if r > 0 else 'down' for r in recent_returns]
    binary_prompt = f"Given these recent price movements: {binary_context}, predict if the next movement will be 'up' or 'down'."
    binary_response = asyncio.run(query_to_claude_async(binary_prompt, system_message="You are a binary guessing system, just best guess the next value nothing else"))
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
