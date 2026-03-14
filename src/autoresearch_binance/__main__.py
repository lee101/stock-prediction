"""Allow `python -m autoresearch_binance` to run the training script."""
import sys
from .train import main

sys.exit(main())
