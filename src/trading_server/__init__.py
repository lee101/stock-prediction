from .client import InMemoryTradingServerClient, TradingServerClient, TradingServerClientLike
from .settings import TradingServerSettings

__all__ = [
    "InMemoryTradingServerClient",
    "TradingServerClient",
    "TradingServerClientLike",
    "TradingServerSettings",
    "TradingServerEngine",
    "app",
    "create_app",
]


def __getattr__(name: str):
    if name in {"TradingServerEngine", "app", "create_app"}:
        from .server import TradingServerEngine, app, create_app

        mapping = {
            "TradingServerEngine": TradingServerEngine,
            "app": app,
            "create_app": create_app,
        }
        return mapping[name]
    if name == "TradingServerSettings":
        return TradingServerSettings
    raise AttributeError(name)
