"""SignalRegistry: register_signal デコレータ + get_signals."""

from .base import Signal

_REGISTRY: dict[str, type] = {}


def register_signal(name: str):
    """シグナルクラスをレジストリに登録するデコレータ."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_all_signals() -> dict[str, type]:
    """登録済みシグナルの辞書を返す."""
    return dict(_REGISTRY)
