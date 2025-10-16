import os

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
from gpt5_queries import query_to_gpt5_async
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
data["returns"] = data[close_column].astype(float).pct_change()
data = data.dropna()

# Define forecast periods
end_idx = len(data) - 1
start_idx = len(data) - 9  # last 8 for now

# Generate forecasts with Chronos
chronos_forecasts = []
chronos_plus_gpt5_forecasts = []

chronos_device = "cuda" if torch.cuda.is_available() else "cpu"
chronos_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
if chronos_device == "cpu":
    logger.warning("CUDA not available; ChronosPipeline will run on CPU with float32 precision. Expect slower forecasts.")

chronos_model = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map=chronos_device,
    torch_dtype=chronos_dtype
)
import re


def _coerce_reasoning_effort(value: str) -> str:
    allowed = {"minimal", "low", "medium", "high"}
    value_norm = (value or "").strip().lower()
    if value_norm in allowed:
        return value_norm
    logger.warning("Unrecognised GPT5_REASONING_EFFORT value '%s'; defaulting to 'high'.", value)
    return "high"


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s='%s'; falling back to %d.", name, raw, default)
        return default


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s='%s'; falling back to %.2f.", name, raw, default)
        return default


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def analyse_prediction(pred: str):
    """
    Extract the final numeric value from a model response.
    GPT-5 may wrap answers in prose, so we always take
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
        matches = re.findall(r"-?\d*\.?\d+", pred_str)
        if matches:
            return float(matches[-1])
        logger.error(f"Failed to extract number from string: {pred}")
        return 0.0
    except Exception as exc:
        logger.error(f"Failed to extract number from string: {pred} ({exc})")
        return 0.0


@async_cache_decorator(typed=True)
async def predict_chronos(context_values):
    """Cached prediction function that doesn't include the model in the cache key."""
    with torch.inference_mode():
        transformers.set_seed(42)
        pred = chronos_model.predict(
            context=torch.from_numpy(context_values),
            prediction_length=1,
            num_samples=100
        ).detach().cpu().numpy().flatten()
        return np.mean(pred)


chronos_abs_error_sum = 0.0
gpt5_abs_error_sum = 0.0
prediction_count = 0

print("Generating forecasts with GPT-5 assistance...")
reasoning_effort = _coerce_reasoning_effort(os.getenv("GPT5_REASONING_EFFORT", "high"))
lock_reasoning = _read_bool_env("GPT5_LOCK_REASONING", True)
max_output_tokens = _read_int_env("GPT5_MAX_OUTPUT_TOKENS", 120_000)
max_output_tokens_cap = _read_int_env("GPT5_MAX_OUTPUT_TOKENS_CAP", 240_000)
token_growth_factor = _read_float_env("GPT5_TOKEN_GROWTH_FACTOR", 1.2)
min_token_increment = _read_int_env("GPT5_MIN_TOKEN_INCREMENT", 20_000)
timeout_seconds = _read_int_env("GPT5_TIMEOUT_SECONDS", 300)
max_retries = _read_int_env("GPT5_MAX_RETRIES", 10)
max_exception_retries = _read_int_env("GPT5_MAX_EXCEPTION_RETRIES", 3)
exception_retry_backoff = _read_float_env("GPT5_EXCEPTION_RETRY_BACKOFF", 5.0)
skip_plot = _read_bool_env("GPT5_SKIP_PLOT", True)

with tqdm(range(start_idx, end_idx), desc="Forecasting") as progress_bar:
    for t in progress_bar:
        context = data["returns"].iloc[:t]
        actual = data["returns"].iloc[t]

        # Chronos forecast - now not passing model as argument
        chronos_pred_mean = asyncio.run(predict_chronos(context.values))

        # GPT-5 forecast
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
        gpt5_pred = analyse_prediction(
            asyncio.run(
                query_to_gpt5_async(
                    prompt,
                    system_message=(
                        "You are a number guessing system. Provide as much reasoning as you require to be maximally accurate. "
                        "Maintain the configured reasoning effort throughout, and ensure the final line of your reply is just the numeric prediction with no trailing text."
                    ),
                    extra_data={
                        "reasoning_effort": reasoning_effort,
                        "lock_reasoning_effort": lock_reasoning,
                        "max_output_tokens": max_output_tokens,
                        "max_output_tokens_cap": max_output_tokens_cap,
                        "token_growth_factor": token_growth_factor,
                        "min_token_increment": min_token_increment,
                        "timeout": timeout_seconds,
                        "max_retries": max_retries,
                        "max_exception_retries": max_exception_retries,
                        "exception_retry_backoff": exception_retry_backoff,
                    },
                    model="gpt-5-mini",
                )
            )
        )

        chronos_forecasts.append({
            "date": data.index[t],
            "actual": actual,
            "predicted": chronos_pred_mean
        })

        chronos_plus_gpt5_forecasts.append({
            "date": data.index[t],
            "actual": actual,
            "predicted": gpt5_pred
        })

        prediction_count += 1
        chronos_abs_error_sum += abs(actual - chronos_pred_mean)
        gpt5_abs_error_sum += abs(actual - gpt5_pred)

        progress_bar.set_postfix(
            chronos_mae=chronos_abs_error_sum / prediction_count,
            chronos_plus_gpt5_mae=gpt5_abs_error_sum / prediction_count,
        )

chronos_df = pd.DataFrame(chronos_forecasts)
chronos_plus_gpt5_df = pd.DataFrame(chronos_plus_gpt5_forecasts)

# Calculate error metrics
chronos_mape = mean_absolute_percentage_error(chronos_df["actual"], chronos_df["predicted"])
chronos_mae = mean_absolute_error(chronos_df["actual"], chronos_df["predicted"])

chronos_plus_gpt5_mape = mean_absolute_percentage_error(
    chronos_plus_gpt5_df["actual"],
    chronos_plus_gpt5_df["predicted"]
)
chronos_plus_gpt5_mae = mean_absolute_error(
    chronos_plus_gpt5_df["actual"],
    chronos_plus_gpt5_df["predicted"]
)

print(f"\nChronos MAPE: {chronos_mape:.4f}")
print(f"Chronos MAE: {chronos_mae:.4f}")
print(f"\nChronos+GPT-5 MAPE: {chronos_plus_gpt5_mape:.4f}")
print(f"Chronos+GPT-5 MAE: {chronos_plus_gpt5_mae:.4f}")

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(chronos_df.index, chronos_df["actual"], label="Actual Returns", color="blue")
plt.plot(chronos_df.index, chronos_df["predicted"], label="Chronos Predicted Returns", color="red", linestyle="--")
plt.plot(
    chronos_plus_gpt5_df.index,
    chronos_plus_gpt5_df["predicted"],
    label="Chronos-Aware GPT-5 Predicted Returns",
    color="green",
    linestyle="--"
)
plt.title("Return Predictions for BTCUSD")
plt.legend()
plt.tight_layout()
if skip_plot:
    plt.close(plt.gcf())
else:
    plt.show()
