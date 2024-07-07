import subprocess

from loguru import logger

from src.utils import debounce


@debounce(60 * 10) # 10 minutes to not call too much
def backout_near_market(symbol):
    command = f"PYTHONPATH=/media/lee/crucial/code/stock python scripts/alpaca_cli.py backout_near_market {symbol}"
    logger.info(f"Running command {command}")
    subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
