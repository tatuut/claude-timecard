"""バイグラム抽出器: 隣接ユニグラムのペア."""

from .registry import register_extractor
from .unigram import UnigramExtractor


@register_extractor("bigram")
class BigramExtractor:
    """隣接ユニグラムのバイグラム。"annotation panel" -> "annotation_panel"."""

    name = "bigram"

    def extract(self, text: str, **context) -> list[str]:
        unigrams = UnigramExtractor().extract(text, **context)
        bigrams = []
        for i in range(len(unigrams) - 1):
            bigrams.append(f"{unigrams[i]}_{unigrams[i+1]}")
        return bigrams
