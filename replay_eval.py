"""Compatibility wrapper for :mod:`pufferlib_market.replay_eval`."""

from pufferlib_market import replay_eval as _maintained_replay_eval
from pufferlib_market.replay_eval import *  # noqa: F403


main = _maintained_replay_eval.main


def __getattr__(name: str):
    return getattr(_maintained_replay_eval, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_maintained_replay_eval)))


if __name__ == "__main__":
    main()
