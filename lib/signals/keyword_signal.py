"""キーワード分布変化シグナル（ソフト境界）."""

from collections import Counter

from ..parser.events import Event
from ..config import TimecardConfig
from ..analysis.tfidf import _tokenize
from ..analysis.embeddings import cosine_sim
from .base import Boundary
from .registry import register_signal


@register_signal("keyword")
class KeywordSignal:
    """キーワード分布の変化点をソフト境界として検出."""

    name = "keyword"
    confidence = "soft"

    def __init__(self, window_size: int = 5, similarity_threshold: float = 0.3):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold

    def detect(self, events: list[Event], config: TimecardConfig) -> list[Boundary]:
        """スライディングウィンドウでキーワード分布の変化点を検出."""
        if len(events) < self.window_size * 2:
            return []

        # 各イベントのキーワードベクトルをウィンドウで集計
        boundaries: list[Boundary] = []
        for i in range(self.window_size, len(events) - self.window_size + 1):
            left_msgs = [e.content for e in events[i - self.window_size : i]]
            right_msgs = [e.content for e in events[i : i + self.window_size]]

            left_vec = self._tfidf_vector(left_msgs)
            right_vec = self._tfidf_vector(right_msgs)

            if left_vec and right_vec:
                # コサイン類似度を計算
                all_keys = set(left_vec.keys()) | set(right_vec.keys())
                lv = [left_vec.get(k, 0.0) for k in all_keys]
                rv = [right_vec.get(k, 0.0) for k in all_keys]
                sim = cosine_sim(lv, rv)

                if sim < self.similarity_threshold:
                    boundaries.append(
                        Boundary(
                            timestamp=events[i].ts,
                            signal_name=self.name,
                            confidence=self.confidence,
                            reason=f"keyword shift (sim={sim:.2f})",
                        )
                    )

        return boundaries

    def _tfidf_vector(self, messages: list[str]) -> dict[str, float]:
        """メッセージリストから簡易TFベクトルを作成."""
        counts: Counter = Counter()
        for msg in messages:
            for token in _tokenize(msg):
                counts[token] += 1
        return dict(counts)
