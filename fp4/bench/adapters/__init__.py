"""Trainer adapters for fp4/bench/bench_trading.py.

Each adapter exposes ``run(cfg, steps, seed, ckpt_dir) -> dict`` matching the
record contract used by ``bench_trading._RUNNERS``.
"""
