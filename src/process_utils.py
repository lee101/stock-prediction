import subprocess
from typing import Optional

from loguru import logger

from src.utils import debounce
from pathlib import Path

cwd = Path.cwd()


@debounce(
    60 * 10, key_func=lambda symbol: symbol
)  # 10 minutes to not call too much for the same symbol
def backout_near_market(symbol):
    command = (
        f"PYTHONPATH={cwd} python scripts/alpaca_cli.py backout_near_market {symbol}"
    )
    logger.info(f"Running command {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Log output asynchronously
    stdout, stderr = process.communicate()
    if stdout:
        logger.info(f"Command output: {stdout.decode()}")
    if stderr:
        logger.error(f"Command error: {stderr.decode()}")


@debounce(60 * 10, key_func=lambda symbol, side: f"{symbol}_{side}")
def ramp_into_position(symbol: str, side: str = "buy"):
    """Ramp into a position over time using the alpaca CLI."""
    command = f"PYTHONPATH={cwd} python scripts/alpaca_cli.py ramp_into_position {symbol} --side={side}"
    logger.info(f"Running command {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Log output asynchronously
    stdout, stderr = process.communicate()
    if stdout:
        logger.info(f"Command output: {stdout.decode()}")
    if stderr:
        logger.error(f"Command error: {stderr.decode()}")
