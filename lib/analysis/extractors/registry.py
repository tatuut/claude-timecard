"""抽出器のRegistryパターン管理."""

_REGISTRY: dict[str, type] = {}


def register_extractor(name: str):
    """抽出器クラスをRegistryに登録するデコレータ."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_all_extractors() -> dict[str, type]:
    """登録済みの全抽出器を返す."""
    return dict(_REGISTRY)
