import types

from pufferlibtraining.train_ppo import sync_vecnormalize_stats


class DummyVecNormalize:
    def __init__(self):
        self.obs_rms = object()
        self.ret_rms = object()
        self.training = True
        self.set_training_mode_calls = []

    def set_training_mode(self, flag: bool):
        self.set_training_mode_calls.append(flag)


def test_sync_vecnormalize_stats_copies_running_statistics():
    src = DummyVecNormalize()
    dest = DummyVecNormalize()
    dest.obs_rms = "unchanged"
    dest.ret_rms = "unchanged"

    sync_vecnormalize_stats(src, dest)

    assert dest.obs_rms is src.obs_rms
    assert dest.ret_rms is src.ret_rms
    assert dest.training is False
    assert dest.set_training_mode_calls[-1] is False


def test_sync_vecnormalize_stats_no_shared_attributes_is_noop():
    src = types.SimpleNamespace()
    dest = types.SimpleNamespace()

    sync_vecnormalize_stats(src, dest)

    assert not hasattr(dest, "obs_rms")
    assert not hasattr(dest, "ret_rms")
