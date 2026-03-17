"""Compatibility wrapper for checkpoint manager imports.

Some training and inference entrypoints import ``src.checkpoint_manager``
while the implementation lives at the repository root. Re-export the root
module here so fresh processes and tests resolve the same class consistently.
"""

from checkpoint_manager import TopKCheckpointManager

__all__ = ["TopKCheckpointManager"]
