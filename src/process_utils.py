import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger

from src.utils import debounce

cwd = Path.cwd()


@debounce(
    60 * 10, key_func=lambda symbol: symbol
)  # 10 minutes to not call too much for the same symbol
def backout_near_market(symbol):
    command = (
        f"PYTHONPATH={cwd} python scripts/alpaca_cli.py backout_near_market {symbol}"
    )
    logger.info(f"Running command {command}")
    # Run process in background without waiting
    subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


@debounce(60 * 10, key_func=lambda symbol, side, target_qty=None: f"{symbol}_{side}_{target_qty}")
def ramp_into_position(symbol: str, side: str = "buy", target_qty: Optional[float] = None):
    """Ramp into a position over time using the alpaca CLI."""
    command = f"PYTHONPATH={cwd} python scripts/alpaca_cli.py ramp_into_position {symbol} --side={side}"
    if target_qty is not None:
        command += f" --target-qty={target_qty}"
    logger.info(f"Running command {command}")
    # Run process in background without waiting
    subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


@debounce(60 * 10, key_func=lambda symbol, takeprofit_price: f"{symbol}_{takeprofit_price}")  # only once in 10 minutes
def spawn_close_position_at_takeprofit(symbol: str, takeprofit_price: float):
    command = f"PYTHONPATH={cwd} python scripts/alpaca_cli.py close_position_at_takeprofit {symbol} --takeprofit_price={takeprofit_price}"
    logger.info(f"Running command {command}")
    # Run process in background without waiting
    subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
