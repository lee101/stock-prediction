"""Compatibility wrapper for :mod:`pufferlib_market.evaluate_holdout`."""

from pufferlib_market import evaluate_holdout as _maintained_evaluate_holdout
from pufferlib_market.evaluate_holdout import *  # noqa: F403


main = _maintained_evaluate_holdout.main


def __getattr__(name: str):
    return getattr(_maintained_evaluate_holdout, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_maintained_evaluate_holdout)))


if __name__ == "__main__":
    main()
