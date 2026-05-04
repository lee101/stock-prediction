"""Compatibility wrapper for the maintained pufferlib trainer.

The implementation lives in :mod:`pufferlib_market.train`. Keeping this file
as a wrapper prevents the root script from drifting from the package entrypoint.
"""

from pufferlib_market import train as _maintained_train
from pufferlib_market.train import *  # noqa: F403


main = _maintained_train.main


def __getattr__(name: str):
    return getattr(_maintained_train, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_maintained_train)))


if __name__ == "__main__":
    main()
