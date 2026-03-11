"""Allow `python -m autoresearch_crypto` to run the training script."""
import sys
from .train import main

sys.exit(main())
