#!/usr/bin/env python3
"""Compatibility wrapper for the Binance crypto LoRA sweep entrypoint."""
from scripts.run_binance_crypto_lora_sweep import *  # noqa: F403
from scripts.run_binance_crypto_lora_sweep import main


if __name__ == "__main__":
    raise SystemExit(main())
