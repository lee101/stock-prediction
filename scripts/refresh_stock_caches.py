#!/usr/bin/env python3
"""Daemon: refresh Chronos2 forecast caches for all 18 stocks on a loop."""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from unified_hourly_experiment.rebuild_all_caches import BEST_MODELS, build_cache

REFRESH_INTERVAL = 3600  # 1 hour

def main():
    logger.info("Stock cache refresh daemon starting ({} symbols)", len(BEST_MODELS))
    while True:
        for symbol, model in BEST_MODELS.items():
            try:
                build_cache(symbol, model)
            except Exception as e:
                logger.error("Cache refresh failed for {}: {}", symbol, e)
        logger.info("Refresh complete, sleeping {}s", REFRESH_INTERVAL)
        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    main()
