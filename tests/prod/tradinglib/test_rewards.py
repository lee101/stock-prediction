import numpy as np

from src.tradinglib.rewards import DrawdownTracker, RewardState, RunningMoments, risk_adjusted_reward, sharpe_like_reward


def test_running_moments_basic():
    moments = RunningMoments()
    for val in [1.0, 2.0, 3.0]:
        moments.update(val)
    assert moments.count == 3
    assert np.isclose(moments.mean, 2.0)
    assert moments.std > 0


def test_drawdown_tracker():
    tracker = DrawdownTracker()
    assert tracker.update(100.0) == 0.0
    assert tracker.update(110.0) == 0.0
    drawdown = tracker.update(90.0)
    assert drawdown < 0.0


def test_risk_adjusted_reward_penalizes_drawdown():
    state = RewardState(moments=RunningMoments(), drawdown=DrawdownTracker())
    reward1 = risk_adjusted_reward(step_return=0.01, state=state, equity=100.0, drawdown_penalty=1.0)
    reward2 = risk_adjusted_reward(step_return=0.01, state=state, equity=90.0, drawdown_penalty=1.0)
    assert reward2 < reward1


def test_sharpe_like_reward_clipped():
    moments = RunningMoments()
    reward = sharpe_like_reward(step_return=0.5, state=moments, clip=0.1)
    assert reward <= 0.1
