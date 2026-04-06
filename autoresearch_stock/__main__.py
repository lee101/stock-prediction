"""Allow `python -m autoresearch_stock` to run the training script."""

import sys

from .train import main


sys.exit(main())
