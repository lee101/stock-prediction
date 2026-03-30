#!/usr/bin/env python3
from __future__ import annotations

import sys

import trade_daily_stock_prod as daily_stock


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or sys.argv[1:])
    default_args = [
        "--checkpoint",
        daily_stock.DEFAULT_SORTINO_SERVER_CHECKPOINT,
        "--no-ensemble",
        "--execution-backend",
        "trading_server",
        "--server-account",
        daily_stock.DEFAULT_SERVER_PAPER_ACCOUNT,
        "--server-bot-id",
        daily_stock.DEFAULT_SERVER_PAPER_BOT_ID,
    ]
    daily_stock.main(default_args + argv)


if __name__ == "__main__":
    main()
