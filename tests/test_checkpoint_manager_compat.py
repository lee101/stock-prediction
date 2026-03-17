from checkpoint_manager import TopKCheckpointManager as RootTopKCheckpointManager
from src.checkpoint_manager import TopKCheckpointManager as SrcTopKCheckpointManager


def test_src_checkpoint_manager_reexports_root_manager() -> None:
    assert SrcTopKCheckpointManager is RootTopKCheckpointManager
