from __future__ import annotations

def make_market_env(*args, **kwargs):
    from pufferlibtraining.market_env import make_market_env as _make_market_env

    return _make_market_env(*args, **kwargs)


class MarketEnv:  # pragma: no cover - thin compatibility shim
    def __new__(cls, *args, **kwargs):
        from pufferlibtraining.market_env import MarketEnv as _MarketEnv

        return _MarketEnv(*args, **kwargs)

__all__ = ["MarketEnv", "make_market_env"]
