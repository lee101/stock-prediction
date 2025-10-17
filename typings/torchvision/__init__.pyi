from typing import Any

__all__: list[str] = []

class _PlaceholderModule:
    def __getattr__(self, name: str) -> Any: ...

datasets: Any
models: Any
ops: Any
transforms: Any
utils: Any

def __getattr__(name: str) -> Any: ...
