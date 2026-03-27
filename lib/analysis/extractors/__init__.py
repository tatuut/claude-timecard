"""キーワード抽出器パッケージ。import時に全抽出器が自動登録される."""

from .registry import register_extractor, get_all_extractors
from .base import TokenExtractor

# 全抽出器をimportしてregistryに登録
from . import unigram  # noqa: F401
from . import bigram   # noqa: F401
from . import branch   # noqa: F401

__all__ = ["register_extractor", "get_all_extractors", "TokenExtractor"]
