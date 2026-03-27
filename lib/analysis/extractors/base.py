"""抽出器のProtocol定義."""

from typing import Protocol


class TokenExtractor(Protocol):
    """トークン抽出器のインターフェース."""

    name: str

    def extract(self, text: str, **context) -> list[str]:
        """テキストからトークンを抽出."""
        ...
