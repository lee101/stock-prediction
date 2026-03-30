from .client import BinanceTradingServerClient
from .settings import BinanceTradingServerSettings

__all__ = ["BinanceTradingServerClient", "BinanceTradingServerSettings", "BinanceTradingServerEngine", "app", "create_app"]


def __getattr__(name: str):
    if name in {"BinanceTradingServerEngine", "app", "create_app"}:
        from .server import BinanceTradingServerEngine, app, create_app
        return {"BinanceTradingServerEngine": BinanceTradingServerEngine, "app": app, "create_app": create_app}[name]
    raise AttributeError(name)
