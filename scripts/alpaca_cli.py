import traceback
from time import time, sleep

import requests.exceptions
from alpaca_trade_api.rest import APIError
from loguru import logger
from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT
import alpaca_trade_api as tradeapi

alpaca_api = tradeapi.REST(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    ALP_ENDPOINT,
    'v2')


