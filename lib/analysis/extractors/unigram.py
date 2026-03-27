"""ユニグラム抽出器: 既存の_tokenize相当."""

import re

from .registry import register_extractor


@register_extractor("unigram")
class UnigramExtractor:
    """英単語3+、カタカナ2+、漢字2+のユニグラムを抽出."""

    name = "unigram"

    def __init__(self):
        # 遅延importでstopwords/synonymsを取得
        from ..tfidf import _GRAMMAR_STOPS, _SYNONYM_MAP
        self._stops = _GRAMMAR_STOPS
        self._synonyms = _SYNONYM_MAP

    def extract(self, text: str, **context) -> list[str]:
        tokens: list[str] = []
        # 英単語（簡易ステミング: 複数形s, ing除去）
        for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text):
            wl = w.lower()
            if wl.endswith("ing") and len(wl) > 5:
                wl = wl[:-3]
            elif wl.endswith("s") and not wl.endswith("ss") and len(wl) > 4:
                wl = wl[:-1]
            if wl not in self._stops and len(wl) >= 3:
                wl = self._synonyms.get(wl, wl)
                tokens.append(wl)
        # カタカナ語
        for w in re.findall(r"[\u30A0-\u30FF]{2,}", text):
            if w not in self._stops:
                w = self._synonyms.get(w, w)
                tokens.append(w)
        # 漢字語
        for w in re.findall(r"[\u4E00-\u9FFF]{2,}", text):
            if w not in self._stops:
                w = self._synonyms.get(w, w)
                tokens.append(w)
        return tokens
